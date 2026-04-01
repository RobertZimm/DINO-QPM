"""Aggregate experimental results — config-hash-based grouping.

This module replaces the folder-name-dependent grouping of the original
``aggregate_results.py`` with a **config-content-based** approach:

1. For every result JSON we locate and load the corresponding
   ``config.yaml`` (including ``ft/config.yaml`` merging).
2. We strip *identity* keys (seed, run_id, output paths …) and compute
   a deterministic SHA-256 hash of the remaining config.
3. Runs that share the same hash are grouped for aggregation.
4. **Before** aggregation — while every row still has its own config —
   we compare configs across groups to derive ``changed_parameters``.
5. We aggregate metrics (mean / std / n) and carry forward the
   pre-computed ``changed_parameters``.

This eliminates the dependency on folder-name conventions like
``<job_id>_<run_number>/`` entirely.  Grouping is based solely on
config content.  The public API (``process_folder``,
``load_results_dataframe``, ``load_results_dataframes``) is preserved.
"""

from pathlib import Path
import ast
import hashlib
import json
import re
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

from dino_qpm.configs.core.conf_getter import get_default_save_dir


# ---------------------------------------------------------------------------
# Constants — whitelist of config keys that matter
# ---------------------------------------------------------------------------

# Only these keys (dot-notation) are used for hashing and comparison.
#
# Three formats are supported:
#   "data"                    – include the ENTIRE ``data`` section
#   "data.crop"               – include only ``data.crop``
#   {"dense": ["opt_mode"]}   – include ``dense`` but EXCLUDE ``opt_mode``
#
# To adjust which keys are considered, simply edit this list.
CONFIG_KEYS_OF_INTEREST: list = [
    # "sldd_mode",
    # "model_type",
    "data.crop",
    "data.model_type",
    "model",
    {"dense": ["opt_mode", "opt_goal", "n_trials"]},
    {"finetune": ["opt_mode", "opt_goal", "n_trials",
                  "no_b", "no_r", "mode", "model_type"]},
]

# Prototype-specific keys excluded from hash/comparison. Uses dot-notation
# (``section.key``).
IGNORE_WHEN_NO_PROTOTYPES: list[str] = [
    "dense.cofs_weight", "dense.cofs_k",
    "finetune.cofs_weight", "finetune.cofs_k",
    "dense.rpl_weight", "finetune.rpl_weight",
    "dense.n_prototypes",
    "dense.per_prototype", "model.init_method", "model.n_prototypes",
    "model.pooling_type", "model.proto_init_strat", "model.proto_method",
    "model.proto_similarity_method", "model.proto_softmax_tau", "model.proto_use_feat_vec",
    "model.apply_relu", "model.proto_pre_pooling_mode"
]


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _extract_keys_of_interest(
    config: Dict,
    keys: list | None = None,
) -> Dict:
    """Extract only the whitelisted keys from *config*.

    *keys* is a list that supports three entry formats:

    * ``"data"``  — include the **entire** ``data`` section.
    * ``"data.crop"``  — include only that specific leaf.
    * ``{"dense": ["opt_mode", ...]}``  — include ``dense`` but
      **exclude** the listed sub-keys.

    Parameters
    ----------
    config : Dict
        Full run config — will **not** be mutated.
    keys : list, optional
        Defaults to :data:`CONFIG_KEYS_OF_INTEREST`.

    Returns
    -------
    Dict
        Shallow/nested copy containing only the requested keys.
    """
    if keys is None:
        keys = CONFIG_KEYS_OF_INTEREST

    # Normalise the three entry formats into three buckets:
    #   top_level          – full section includes             {"model"}
    #   exclude_map        – section with excluded sub-keys    {"dense": {"opt_mode"}}
    #   sub_keys           – specific dotted sub-key includes  {"data": {"crop"}}
    top_level: set[str] = set()
    exclude_map: Dict[str, set] = {}  # section → excluded leaf names
    sub_keys: Dict[str, set] = {}     # section → included leaf names

    for entry in keys:
        if isinstance(entry, dict):
            for section, excl_list in entry.items():
                exclude_map.setdefault(section, set()).update(excl_list)
        elif isinstance(entry, str):
            if "." in entry:
                section, rest = entry.split(".", 1)
                sub_keys.setdefault(section, set()).add(rest)
            else:
                top_level.add(entry)
        # silently skip anything else

    for dotted in IGNORE_WHEN_NO_PROTOTYPES:
        parts = dotted.split(".", 1)
        if len(parts) == 2:
            sec, leaf = parts
            # If the section is a full include, demote it to
            # an exclude-map entry so we can drop the leaf.
            if sec in top_level:
                top_level.discard(sec)
                exclude_map.setdefault(sec, set()).add(leaf)
            elif sec in exclude_map:
                exclude_map[sec].add(leaf)
            else:
                # Section was included via sub_keys; remove the leaf
                if sec in sub_keys:
                    sub_keys[sec].discard(leaf)
                    if not sub_keys[sec]:
                        del sub_keys[sec]
        elif len(parts) == 1:
            top_level.discard(parts[0])
            exclude_map.pop(parts[0], None)
            sub_keys.pop(parts[0], None)

    extracted: Dict = {}

    # --- full section includes ---
    for section in top_level:
        if section in config:
            val = config[section]
            extracted[section] = (
                json.loads(json.dumps(val, default=str))
                if isinstance(val, dict) else val
            )

    # --- section-with-exclusions ---
    for section, excl_keys in exclude_map.items():
        if section in top_level:
            # Already grabbed fully; just strip the excluded keys
            if section in extracted and isinstance(extracted[section], dict):
                for ek in excl_keys:
                    extracted[section].pop(ek, None)
            continue
        if section not in config or not isinstance(config[section], dict):
            continue
        # Deep-copy the whole section, then drop the excluded keys
        copied = json.loads(json.dumps(config[section], default=str))
        for ek in excl_keys:
            copied.pop(ek, None)
        if copied:  # only add if anything remains
            extracted[section] = copied

    # --- dotted sub-key includes ---
    for section, leaves in sub_keys.items():
        if section in top_level or section in exclude_map:
            continue  # already handled
        if section not in config or not isinstance(config[section], dict):
            continue
        sub: Dict = {}
        for leaf in leaves:
            parts = leaf.split(".")
            src = config[section]
            dst_parent = sub
            for part in parts[:-1]:
                if not isinstance(src, dict) or part not in src:
                    break
                src = src[part]
                dst_parent = dst_parent.setdefault(part, {})
            else:
                last = parts[-1]
                if isinstance(src, dict) and last in src:
                    dst_parent[last] = src[last]
        if sub:
            extracted[section] = sub

    return extracted


def _config_hash(config: Dict) -> str:
    """Deterministic 16-char hex hash of a config.

    Only the keys listed in :data:`CONFIG_KEYS_OF_INTEREST` are
    considered.  Two configs that differ only outside the whitelist
    (e.g. seed, paths) produce the same hash.
    """
    extracted = _extract_keys_of_interest(config)
    config_str = json.dumps(extracted, sort_keys=True, default=str)
    return hashlib.sha256(config_str.encode()).hexdigest()[:16]


# Sections to exclude from the base hash (shared identity between dense
# and finetuned runs).  Add section names here to strip them.
_FINETUNE_SECTIONS: set[str] = {"finetune"}


def _base_hash(config: Dict) -> str:
    """Like :func:`_config_hash` but **excludes** finetuning sections.

    Dense and finetuned runs from the same experiment share the same
    base hash, which enables pairing them for comparison.
    """
    # Filter CONFIG_KEYS_OF_INTEREST to exclude finetune sections
    base_keys: list = []
    for entry in CONFIG_KEYS_OF_INTEREST:
        if isinstance(entry, dict):
            filtered = {
                k: v for k, v in entry.items()
                if k not in _FINETUNE_SECTIONS
            }
            if filtered:
                base_keys.append(filtered)
        elif isinstance(entry, str):
            section = entry.split(".")[0]
            if section not in _FINETUNE_SECTIONS:
                base_keys.append(entry)
    extracted = _extract_keys_of_interest(config, keys=base_keys)
    config_str = json.dumps(extracted, sort_keys=True, default=str)
    return hashlib.sha256(config_str.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Result-type registry
# ---------------------------------------------------------------------------

# Single source of truth for all result phases.
#
# Each key is a canonical phase name.  The value is a **regex** that is
# matched against the full JSON **filename** (not path).  A file is
# assigned to the *first* phase whose pattern matches.
#
# To add a new phase, simply add an entry here — everything else
# (``res_to_df``, ``_detect_result_type``, ``type_filter``) picks it
# up automatically.
RESULT_PHASES: dict[str, str] = {
    "dense":    r"^Results_DenseModel\.json$",
    "finetune": r"^Results_.*FinetunedModel\.json$",
}

# Pre-compiled for speed (rebuilt if the dict is mutated at import time).
_PHASE_PATTERNS: list[tuple[str, re.Pattern]] = [
    (phase, re.compile(pattern))
    for phase, pattern in RESULT_PHASES.items()
]


def _detect_result_type(filename: str) -> str | None:
    """Return the phase name for *filename*, or ``None`` if no phase matches."""
    for phase, pat in _PHASE_PATTERNS:
        if pat.search(filename):
            return phase
    return None


def find_run_config(json_file_path: Path, base_folder: Path) -> Optional[Path]:
    """Locate the ``config.yaml`` for a given result JSON file.

    Walks upward from the JSON file's directory towards *base_folder*,
    returning the first ``config.yaml`` found.  Projection directories
    receive special treatment: the search jumps to the ``ft/`` ancestor
    first.
    """
    json_file_path = Path(json_file_path)
    start_dir = json_file_path.parent if json_file_path.is_file() else json_file_path

    # Special handling for projection directories
    if "projection" in json_file_path.parts:
        current = start_dir
        while current != base_folder and current.name != "ft":
            current = current.parent
        if current.name == "ft":
            cfg = current / "config.yaml"
            if cfg.is_file():
                return cfg

    # Walk upward from the JSON directory to the base folder
    current = start_dir
    while True:
        cfg = current / "config.yaml"
        if cfg.is_file():
            return cfg
        if current == base_folder or current == current.parent:
            break
        current = current.parent

    return None


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """Recursively merge *override* into *base* (in-place) and return *base*."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def load_config_with_ft(config_path: Path) -> Dict:
    """Load a run config, merging in ``ft/config.yaml`` when available.

    For dense runs the config written at training time may still carry the
    base-default values for the ``finetune`` section.  The
    ``ft/config.yaml`` produced during finetuning contains the correct
    swept values.  If the loaded config is not already inside an ``ft/``
    directory and a sibling ``ft/config.yaml`` exists, its values are
    deep-merged on top.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if config_path.parent.name == "ft":
        return config

    ft_config_path = config_path.parent / "ft" / "config.yaml"
    if ft_config_path.is_file():
        try:
            with open(ft_config_path, "r") as f:
                ft_config = yaml.safe_load(f)
            _deep_merge(config, ft_config)
        except Exception:
            pass  # fall back to the primary config

    return config


# Sentinel object to distinguish "key absent" from "key explicitly None/null".
_MISSING = "<MISSING>"


def _get_nested_value(config: Dict, key: str, missing=_MISSING):
    """Get value from nested dict using dot notation (``'dense.learning_rate'``).

    Returns *missing* (default :data:`_MISSING`) when the key path does
    not exist.  This lets callers distinguish a genuinely absent key from
    one that is explicitly set to ``None``.
    """
    keys = key.split(".")
    value = config
    for k in keys:
        if isinstance(value, dict) and k in value:
            value = value[k]
        else:
            return missing
    return value


def _derive_majority_changed_parameters(
    df: pd.DataFrame,
    hash_to_config: Dict[str, Dict],
) -> pd.DataFrame:
    """Build ``changed_parameters`` via majority-based comparison.

    For every whitelisted config leaf key, find the **most common** value
    across all ``group_key`` entries.  A row's ``changed_parameters`` then
    lists only those keys whose value differs from the majority.

    This replaces the old base-config approach and avoids the need for a
    single "reference" config.
    """
    if not hash_to_config or "group_key" not in df.columns:
        return df

    # Prune: keep only configs whose hash is an active group_key
    active_hashes = set(df["group_key"].unique())
    hash_to_config = {h: c for h, c in hash_to_config.items()
                      if h in active_hashes}
    if not hash_to_config:
        return df

    # ---- 1. Collect all leaf keys from whitelisted configs ----
    def _all_leaf_keys(cfg: Dict, prefix: str = "") -> set:
        keys: set = set()
        for k, v in cfg.items():
            cur = f"{prefix}{k}" if prefix else k
            if isinstance(v, dict):
                keys.update(_all_leaf_keys(v, f"{cur}."))
            else:
                keys.add(cur)
        return keys

    all_keys: set = set()
    for cfg in hash_to_config.values():
        extracted = _extract_keys_of_interest(cfg)
        all_keys.update(_all_leaf_keys(extracted))

    # ---- 2. Find majority value per key across group_keys ----
    from collections import Counter
    key_counts: Dict[str, Counter] = {k: Counter() for k in all_keys}

    for h, cfg in hash_to_config.items():
        extracted = _extract_keys_of_interest(cfg)
        for k in all_keys:
            v = _get_nested_value(extracted, k)  # returns _MISSING if absent
            key_counts[k][str(v)] += 1

    majority: Dict[str, str] = {}
    for k, counter in key_counts.items():
        if counter:
            majority[k] = counter.most_common(1)[0][0]

    # ---- 3. For each group_key, list keys differing from majority ----
    hash_to_params: Dict[str, str] = {}
    for h in active_hashes:
        cfg = hash_to_config.get(h)
        if cfg is None:
            hash_to_params[h] = "N/A"
            continue
        extracted = _extract_keys_of_interest(cfg)
        parts = []
        for k in sorted(all_keys):
            v = _get_nested_value(extracted, k)
            v_str = str(v)
            if v_str != majority.get(k):
                display = v if v is not _MISSING else "<absent>"
                parts.append(f"{k}={display}")
        hash_to_params[h] = ", ".join(parts) if parts else "no changes"

    df = df.copy()
    df["changed_parameters"] = df["group_key"].map(
        hash_to_params).fillna("N/A")
    return df


def _extend_changed_parameters(
    df: pd.DataFrame,
    hash_to_config: Dict[str, Dict],
) -> pd.DataFrame:
    """Ensure every varying key appears in every row's ``changed_parameters``.

    After :func:`_derive_majority_changed_parameters` each row only lists
    keys that *differ* from the majority.  This function finds **all** keys
    whose value varies across any pair of rows and writes the complete set
    (with each row's actual value) into ``changed_parameters``.

    The original, compact version is assumed to already be saved as
    ``unext_changed_parameters`` before this function is called.
    """
    if not hash_to_config or "group_key" not in df.columns:
        return df

    # Prune: keep only configs whose hash is an active group_key
    active_hashes = set(df["group_key"].unique())
    hash_to_config = {h: c for h, c in hash_to_config.items()
                      if h in active_hashes}
    if not hash_to_config:
        return df

    def _all_leaf_keys(cfg: Dict, prefix: str = "") -> set:
        keys: set = set()
        for k, v in cfg.items():
            cur = f"{prefix}{k}" if prefix else k
            if isinstance(v, dict):
                keys.update(_all_leaf_keys(v, f"{cur}."))
            else:
                keys.add(cur)
        return keys

    # Collect all leaf keys from whitelisted configs
    all_keys: set = set()
    for cfg in hash_to_config.values():
        extracted = _extract_keys_of_interest(cfg)
        all_keys.update(_all_leaf_keys(extracted))

    # Find keys whose value varies across active hashes
    key_values: Dict[str, set] = {}
    for h, cfg in hash_to_config.items():
        extracted = _extract_keys_of_interest(cfg)
        for k in all_keys:
            v = _get_nested_value(extracted, k)  # returns _MISSING if absent
            key_values.setdefault(k, set()).add(str(v))

    varying_keys = sorted(k for k, vs in key_values.items() if len(vs) > 1)
    if not varying_keys:
        return df

    # Build extended changed_parameters per group_key
    hash_to_ext: Dict[str, str] = {}
    for h in active_hashes:
        cfg = hash_to_config.get(h)
        if cfg is None:
            hash_to_ext[h] = "N/A"
            continue
        extracted = _extract_keys_of_interest(cfg)
        parts = []
        for k in varying_keys:
            v = _get_nested_value(extracted, k)
            display = v if v is not _MISSING else "<absent>"
            parts.append(f"{k}={display}")
        hash_to_ext[h] = ", ".join(parts) if parts else "no changes"

    df = df.copy()
    df["changed_parameters"] = df["group_key"].map(
        hash_to_ext).fillna("N/A")
    return df


# ---------------------------------------------------------------------------
# Parameter string parsing / formatting helpers
# ---------------------------------------------------------------------------

def parse_changed_parameters(param_string: str) -> dict:
    """Parse ``"param1=value1, param2=value2, …"`` into a dict."""
    pattern = re.compile(
        r"([a-zA-Z_][a-zA-Z0-9_.]*?)=(.+?)(?=, [a-zA-Z_][a-zA-Z0-9_.]*?=|$)"
    )
    params: dict = {}
    if not isinstance(param_string, str):
        return {}
    for match in pattern.finditer(param_string):
        key = match.group(1)
        value_str = match.group(2).strip()
        try:
            params[key] = ast.literal_eval(value_str)
        except (ValueError, SyntaxError):
            params[key] = value_str
    return params


def expand_changed_parameters(df: pd.DataFrame) -> pd.DataFrame:
    """Expand ``changed_parameters`` column into individual parameter columns."""
    if "changed_parameters" not in df.columns:
        return df
    df = df.copy()
    first_val = str(df["changed_parameters"].iloc[0]) if not df.empty else ""
    if "=" not in first_val and first_val != "N/A":
        return df
    param_df = df["changed_parameters"].apply(
        parse_changed_parameters).apply(pd.Series)
    if not param_df.empty and len(param_df.columns) > 0:
        df = pd.concat([df, param_df], axis=1)
        print(
            f"Info: Parsed 'changed_parameters' into columns: {param_df.columns.tolist()}")
    return df


# ---------------------------------------------------------------------------
# Loading results from JSON
# ---------------------------------------------------------------------------

def res_to_df(folder: Path, type_filter: str = "all") -> pd.DataFrame:
    """Load results from result JSON files under *folder*.

    Only files whose name matches a regex in :data:`RESULT_PHASES` are
    considered.  This ensures that unrelated JSONs (configs, etc.) are
    never loaded.

    Parameters
    ----------
    type_filter : str
        ``"all"`` keeps every known phase.  Pass any key from
        :data:`RESULT_PHASES` (e.g. ``"dense"``, ``"finetune"``) to
        restrict to that phase.
    """
    if type_filter != "all" and type_filter not in RESULT_PHASES:
        raise ValueError(
            f"Unknown type_filter {type_filter!r}. "
            f"Choose from {list(RESULT_PHASES)} or 'all'."
        )

    results: List[Dict] = []
    for json_file in folder.glob("**/*.json"):
        rtype = _detect_result_type(json_file.name)
        if rtype is None:                       # not a known result file
            continue
        if type_filter != "all" and rtype != type_filter:
            continue
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            data["type"] = rtype
            data["filepath"] = str(json_file.relative_to(folder))
            data["json_file_path"] = str(json_file)
            data["filename"] = json_file.name
            results.append(data)
        except (UnicodeDecodeError, json.JSONDecodeError):
            pass

    df = pd.DataFrame(results)
    if not df.empty and "type" in df.columns:
        cols = ["type", "filepath", "json_file_path", "filename"] + [
            c for c in df.columns
            if c not in ("type", "filepath", "json_file_path", "filename")
        ]
        df = df[cols]
    return df


# ---------------------------------------------------------------------------
# Core: aggregate_runs  (config-hash-based)
# ---------------------------------------------------------------------------

def aggregate_runs(
    df: pd.DataFrame,
    folder: Path = None,
) -> tuple[pd.DataFrame, Dict[str, Dict]]:
    """Group repetitions and compute mean / std per unique configuration.

    Grouping strategy
    -----------------
    1. Load each run's ``config.yaml``, compute ``config_hash`` (full)
       and ``base_hash`` (excluding finetune sections).
    2. **Dense** rows are grouped by ``base_hash``;
       all other rows by ``config_hash``.
    3. Aggregate numeric columns (mean / std / count).

    Only configs with corresponding result rows (matching the upstream
    ``type_filter``) are loaded.  ``hash_to_config`` is pruned to
    active ``group_key`` entries so that configs without results never
    influence downstream processing.

    ``changed_parameters`` is **not** computed here; it is derived once
    at the end of the pipeline in :func:`load_results_dataframe`.

    Rows whose config could not be loaded are dropped with a warning.

    Returns
    -------
    (aggregated_df, hash_to_config)
        ``hash_to_config`` maps every ``group_key`` (the hash used for
        grouping) to its representative config dict.
    """
    if df.empty or "filepath" not in df.columns:
        return df, {}

    df = df.copy()

    # ==================================================================
    # STEP 1 — Load configs, compute hashes
    # ==================================================================
    config_cache: Dict[str, Optional[Dict]] = {}   # json_file_path → config
    # json_file_path → config_hash
    full_hash_cache: Dict[str, Optional[str]] = {}
    # json_file_path → base_hash
    base_hash_cache: Dict[str, Optional[str]] = {}

    for _, row in df.iterrows():
        jfp = row["json_file_path"]
        if jfp in full_hash_cache:
            continue
        cfg_path = find_run_config(Path(jfp), folder) if folder else None
        if cfg_path is not None:
            try:
                cfg = load_config_with_ft(cfg_path)
                config_cache[jfp] = cfg
                full_hash_cache[jfp] = _config_hash(cfg)
                base_hash_cache[jfp] = _base_hash(cfg)
            except Exception:
                config_cache[jfp] = None
                full_hash_cache[jfp] = None
                base_hash_cache[jfp] = None
        else:
            config_cache[jfp] = None
            full_hash_cache[jfp] = None
            base_hash_cache[jfp] = None

    df["config_hash"] = df["json_file_path"].map(full_hash_cache)
    df["base_hash"] = df["json_file_path"].map(base_hash_cache)

    # Drop rows without a config hash (no config found)
    n_missing = df["config_hash"].isna().sum()
    if n_missing:
        print(f"Warning: Dropping {n_missing} rows with no config found.")
        df = df[df["config_hash"].notna()].copy()
    if df.empty:
        return df, {}

    # Grouping key: dense → base_hash, everything else → config_hash
    df["group_key"] = df.apply(
        lambda r: r["base_hash"] if r.get("type") == "dense"
        else r["config_hash"],
        axis=1,
    )

    # ==================================================================
    # STEP 2 — Build hash_to_config mapping (group_key → config)
    # ==================================================================
    hash_to_config: Dict[str, Dict] = {}
    for jfp, cfg in config_cache.items():
        if cfg is None:
            continue
        # Map both full and base hashes
        fh = full_hash_cache[jfp]
        bh = base_hash_cache[jfp]
        if fh is not None and fh not in hash_to_config:
            hash_to_config[fh] = cfg
        if bh is not None and bh not in hash_to_config:
            hash_to_config[bh] = cfg

    # Prune: keep only hashes that are actual group_keys (drop configs
    # with no corresponding result rows).
    active_keys = set(df["group_key"].unique())
    hash_to_config = {h: c for h, c in hash_to_config.items()
                      if h in active_keys}

    # ==================================================================
    # STEP 3 — Aggregate metrics
    # ==================================================================
    internal_cols = {
        "type", "filepath",
        "json_file_path", "filename", "config_hash", "base_hash", "group_key",
    }
    metric_cols = [c for c in df.columns if c not in internal_cols]

    agg_dict: Dict[str, tuple] = {}
    for col in metric_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            agg_dict[f"{col}_mean"] = (col, "mean")
            agg_dict[f"{col}_std"] = (col, lambda x: x.std(ddof=0))
            agg_dict[f"{col}_n"] = (col, "count")

    grouped = df.groupby("group_key").agg(**agg_dict).reset_index()

    # n_samples: use Accuracy count (always present, case-insensitive)
    acc_n_col = next(
        (c for c in grouped.columns
         if c.lower() == "accuracy_n"),
        None,
    )
    if acc_n_col is not None:
        grouped["n_samples"] = grouped[acc_n_col].astype(int)
    else:
        n_cols = [c for c in grouped.columns if c.endswith("_n")]
        if n_cols:
            grouped["n_samples"] = grouped[n_cols].max(axis=1).astype(int)

    # Carry forward auxiliary columns (first occurrence per group)
    for aux in ("type", "json_file_path", "filename", "config_hash", "base_hash"):
        if aux in df.columns:
            first = df.groupby("group_key")[aux].first().reset_index()
            grouped = grouped.merge(first, on="group_key")

    # ==================================================================
    # STEP 4 — Column ordering  (group_key / hashes last)
    # ==================================================================
    front_cols: List[str] = []
    front_cols.append("type")
    if "n_samples" in grouped.columns:
        front_cols.append("n_samples")

    tail_cols = ["group_key", "base_hash", "config_hash"]
    drop_cols: set = set()
    middle = [
        c for c in grouped.columns
        if c not in front_cols and c not in tail_cols and c not in drop_cols
    ]
    cols = front_cols + middle + tail_cols
    grouped = grouped[[c for c in cols if c in grouped.columns]]

    return grouped, hash_to_config


def _reaggregate_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Re-aggregate rows that share the same ``group_key``.

    When the same config (same hash) appears in multiple source folders,
    concatenating per-folder aggregated DataFrames produces duplicate
    ``group_key`` rows.  This function combines them using weighted mean
    and pooled standard deviation so that each ``group_key`` appears
    exactly once.

    Operates on the pre-``combine_mean_std`` DataFrame (columns like
    ``Accuracy_mean``, ``Accuracy_std``, ``Accuracy_n``).
    """
    if "group_key" not in df.columns:
        return df

    dups = df["group_key"].duplicated(keep=False)
    if not dups.any():
        return df

    unique_df = df[~dups].copy()
    dup_df = df[dups].copy()

    mean_cols = [c for c in dup_df.columns if c.endswith("_mean")]
    metric_names = [c.replace("_mean", "") for c in mean_cols]

    reagged_rows: list[dict] = []
    for gk, group in dup_df.groupby("group_key"):
        row: dict = {"group_key": gk}

        combined_n_samples = 0
        for metric in metric_names:
            mc = f"{metric}_mean"
            sc = f"{metric}_std"
            nc = f"{metric}_n"
            if mc not in group.columns or sc not in group.columns:
                continue
            means = group[mc].to_numpy(dtype=float, na_value=np.nan)
            stds = group[sc].to_numpy(dtype=float, na_value=np.nan)
            ns = (
                group[nc].to_numpy(dtype=float, na_value=0)
                if nc in group.columns
                else np.ones(len(group))
            )
            valid = ~np.isnan(means) & ~np.isnan(stds) & (ns > 0)
            if not valid.any():
                row[mc] = np.nan
                row[sc] = np.nan
                if nc in group.columns:
                    row[nc] = 0
                continue
            means, stds, ns = means[valid], stds[valid], ns[valid]
            total_n = ns.sum()
            wmean = (means * ns).sum() / total_n
            # Pooled variance (within + between group variance)
            wvar = (ns * (stds ** 2 + (means - wmean) ** 2)).sum() / total_n
            row[mc] = wmean
            row[sc] = np.sqrt(wvar)
            if nc in group.columns:
                row[nc] = total_n
            combined_n_samples = max(combined_n_samples, int(total_n))

        row["n_samples"] = combined_n_samples

        # Carry forward auxiliary columns (first occurrence)
        for aux in ("type", "json_file_path", "filename", "config_hash", "base_hash",
                    "source_folder"):
            if aux in group.columns:
                row[aux] = group[aux].iloc[0]

        reagged_rows.append(row)

    if reagged_rows:
        reagged_df = pd.DataFrame(reagged_rows)
        result = pd.concat([unique_df, reagged_df], ignore_index=True)
    else:
        result = unique_df

    return result


# ---------------------------------------------------------------------------
# combine / filter / compare helpers  (unchanged logic)
# ---------------------------------------------------------------------------

def combine_mean_std(
    df: pd.DataFrame,
    round_digits: int | None = 4,
    as_percent: bool = False,
    include_n: bool = False,
    exclude_from_percent: list = None,
) -> pd.DataFrame:
    if df.empty:
        return df
    if exclude_from_percent is None:
        exclude_from_percent = [
            "Alignment", "alignment", "NFfeatures", "n_per_class", "PerClass",
        ]
    df = df.copy()

    metric_names = {c.replace("_mean", "")
                    for c in df.columns if c.endswith("_mean")}
    combined_cols: List[str] = []
    if "type" in df.columns:
        combined_cols.append("type")
    if "filename" in df.columns:
        combined_cols.append("filename")
    if "n_samples" in df.columns:
        combined_cols.append("n_samples")
    if "source_folder" in df.columns:
        combined_cols.append("source_folder")

    metrics_to_add: List[str] = []

    def _safe_precision(rd: int | None, default: int = 4) -> int:
        return rd if isinstance(rd, int) and rd >= 0 else default

    def _format_mean_std(mean_val, std_val, scale: int, rd: int | None) -> str:
        if pd.isna(mean_val) or pd.isna(std_val):
            return ""
        prec = _safe_precision(rd)
        return f"{mean_val*scale:.{prec}f} ± {std_val*scale:.{prec}f}"

    for metric in sorted(metric_names):
        mean_col, std_col, n_col = f"{metric}_mean", f"{metric}_std", f"{metric}_n"
        if mean_col in df.columns and std_col in df.columns:
            apply_pct = as_percent and metric not in exclude_from_percent
            scale = 100 if apply_pct else 1
            df[metric] = df.apply(
                lambda r, mc=mean_col, sc=std_col, s=scale, rd=round_digits: _format_mean_std(
                    r[mc], r[sc], s, rd
                ),
                axis=1,
            )
            metrics_to_add.append(metric)
            if include_n and n_col in df.columns:
                metrics_to_add.append(n_col)

    if "accuracy" in metrics_to_add:
        combined_cols.append("accuracy")
        metrics_to_add.remove("accuracy")
    combined_cols.extend(metrics_to_add)

    # Tail: hash columns last
    for c in ("group_key", "base_hash", "config_hash"):
        if c in df.columns:
            combined_cols.append(c)

    return df[combined_cols]


def filter_metrics(
    df: pd.DataFrame,
    metric_names: list = None,
    mapping: dict = None,
) -> pd.DataFrame:
    if df.empty:
        return df
    if mapping is None:
        mapping = {}
    id_cols = {"group_key", "base_hash", "config_hash",
               "type", "filename", "source_folder"}
    all_metrics = [c for c in df.columns if c not in id_cols]
    print(f"All metrics: {all_metrics}")

    cols_to_keep: List[str] = []
    for c in ("type", "filename", "source_folder"):
        if c in df.columns:
            cols_to_keep.append(c)

    if metric_names is None:
        metric_names = [c for c in all_metrics]
        print("No metric selection provided - using all metrics")

    filtered = []
    for m in metric_names:
        if m in df.columns and m not in id_cols:
            if m not in cols_to_keep:
                cols_to_keep.append(m)
                filtered.append(m)
    print(f"Filtered metrics: {filtered}")

    # Tail: hash columns last
    for c in ("group_key", "base_hash", "config_hash"):
        if c in df.columns and c not in cols_to_keep:
            cols_to_keep.append(c)

    result_df = df[cols_to_keep].copy()
    if mapping:
        result_df.rename(columns=mapping, inplace=True)
    return result_df


def compare_dense_finetune(
    dense_df: pd.DataFrame,
    finetune_df: pd.DataFrame,
    round_digits: int | None = 1,
    as_percent: bool = True,
) -> pd.DataFrame:
    """Compare dense vs finetune results, pairing by ``base_hash``.

    Both DataFrames must contain a ``base_hash`` column (produced by
    ``aggregate_runs``).  Rows with the same ``base_hash`` share the
    identical non-finetune config and are therefore comparable.
    """
    if dense_df.empty or finetune_df.empty:
        print("Warning: One or both dataframes are empty")
        return pd.DataFrame()

    dense_df = dense_df.copy()
    finetune_df = finetune_df.copy()

    # Pairing via base_hash (dense has group_key == base_hash already)
    pair_col = "base_hash" if "base_hash" in dense_df.columns else "group_key"
    ft_pair_col = "base_hash" if "base_hash" in finetune_df.columns else "group_key"

    non_metric = {pair_col, ft_pair_col, "group_key", "type",
                  "base_hash", "config_hash", "json_file_path",
                  "changed_parameters", "n_samples"}
    dense_metrics = [
        c for c in dense_df.columns
        if pd.api.types.is_numeric_dtype(dense_df[c]) and c not in non_metric
    ]
    finetune_metrics = [
        c for c in finetune_df.columns
        if pd.api.types.is_numeric_dtype(finetune_df[c]) and c not in non_metric
    ]
    common_metrics = list(set(dense_metrics) & set(finetune_metrics))
    if not common_metrics:
        print("Warning: No matching metric columns found between dense and finetune")
        return pd.DataFrame()

    delta_records: List[Dict] = []
    common_keys = sorted(
        set(dense_df[pair_col]) & set(finetune_df[ft_pair_col])
    )
    for bh in common_keys:
        ds = dense_df[dense_df[pair_col] == bh]
        fs = finetune_df[finetune_df[ft_pair_col] == bh]
        for _, dr in ds.iterrows():
            for _, fr in fs.iterrows():
                rec: Dict[str, Any] = {"base_hash": bh}
                for m in common_metrics:
                    dv, fv = dr[m], fr[m]
                    if pd.notna(dv) and pd.notna(fv) and dv != 0:
                        rec[f"{m}_delta"] = ((fv - dv) / abs(dv)) * 100
                delta_records.append(rec)

    if not delta_records:
        print("Warning: No matching runs found between dense and finetune")
        return pd.DataFrame()

    delta_df = pd.DataFrame(delta_records)
    delta_cols = [c for c in delta_df.columns if c.endswith("_delta")]

    agg_dict: Dict[str, tuple] = {}
    for c in delta_cols:
        base = c.replace("_delta", "")
        agg_dict[f"{base}_delta_%_mean"] = (c, "mean")
        agg_dict[f"{base}_delta_%_std"] = (c, "std")
        agg_dict[f"{base}_delta_%_n"] = (c, "count")

    aggregated = delta_df.groupby("base_hash").agg(**agg_dict).reset_index()

    # Rename base_hash → group_key for consistency
    aggregated.rename(columns={"base_hash": "group_key"}, inplace=True)

    if as_percent:
        def _safe_precision(rd: int | None, default: int = 1) -> int:
            return rd if isinstance(rd, int) and rd >= 0 else default

        id_cols_out: Dict[str, Any] = {"group_key": aggregated["group_key"]}
        mnames = {
            c.replace("_delta_%_mean", "").replace(
                "_delta_%_std", "").replace("_delta_%_n", "")
            for c in aggregated.columns if "_delta_%" in c
        }
        for m in sorted(mnames):
            mc, sc = f"{m}_delta_%_mean", f"{m}_delta_%_std"
            if mc in aggregated.columns and sc in aggregated.columns:
                id_cols_out[f"{m}_delta_%"] = aggregated.apply(
                    lambda r, _mc=mc, _sc=sc, rd=round_digits: (
                        f"{r[_mc]:.{_safe_precision(rd)}f} ± {r[_sc]:.{_safe_precision(rd)}f}"
                        if pd.notna(r[_mc]) and pd.notna(r[_sc]) else ""
                    ),
                    axis=1,
                )
        return pd.DataFrame(id_cols_out)
    return aggregated


def filter_by_changed_parameters(
    df: pd.DataFrame,
    ext_params: Dict[str, Any] | None = None,
    unext_params: Dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Keep rows that satisfy the filter criteria.

    *ext_params* operates on the **expanded DataFrame columns** (the
    extended / individual parameter columns such as ``finetune.fdl``).

    *unext_params* operates on ``unext_changed_parameters`` (the
    compact, non-extended string) so that filtering reflects the
    original pre-extension parameter set.

    Both dicts use the same convention:

    - ``value is None`` → **filter**: require that the parameter *name*
      is present (any value).
    - ``value is not None`` → **fix**: require an exact match.

    Rows with ``"no changes"`` are **always** kept regardless of the
    filter criteria.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain an ``unext_changed_parameters`` column.
    ext_params : dict, optional
        ``{param_name: value | None}`` — checked against expanded
        DataFrame columns.  ``None`` = column must exist & be non-NaN;
        set value = column must equal that value.
    unext_params : dict, optional
        ``{param_name: value | None}`` — checked against the
        ``unext_changed_parameters`` string.  ``None`` = param name
        must appear; set value = param must equal that value.

    Returns
    -------
    pd.DataFrame
        Filtered copy.
    """
    if df.empty or "unext_changed_parameters" not in df.columns:
        return df
    if not ext_params and not unext_params:
        return df

    col = "unext_changed_parameters"

    # Always keep "no changes" rows
    no_change = df[col].str.strip() == "no changes"

    mask = pd.Series(True, index=df.index)

    # --- ext_params: match against expanded DataFrame columns ---
    if ext_params:
        for param, required in ext_params.items():
            if param not in df.columns:
                if required is None:
                    # Column absent → param never present → drop
                    print(f"Warning: ext_params key '{param}' is not a "
                          f"column in the DataFrame. No rows can match.")
                    mask &= False
                else:
                    print(f"Warning: ext_params key '{param}' is not a "
                          f"column in the DataFrame. Dropping all "
                          f"non-'no changes' rows for this constraint.")
                    mask &= False
                continue
            if required is None:
                # Filter: column must be present and non-NaN
                mask &= df[param].notna()
            else:
                mask &= df[param] == required

    # --- unext_params: match against unext_changed_parameters string ---
    if unext_params:
        for param, required in unext_params.items():
            if required is None:
                # Filter: param name must appear
                def _has_param(cp_str, _p=param):
                    parsed = parse_changed_parameters(cp_str)
                    return _p in parsed
                mask &= df[col].apply(_has_param)
            else:
                # Fix: param must equal required value
                def _fix_match(cp_str, _p=param, _r=required):
                    parsed = parse_changed_parameters(cp_str)
                    return parsed.get(_p) == _r
                mask &= df[col].apply(_fix_match)

    filtered = df[mask | no_change].copy()
    n_dropped = len(df) - len(filtered)
    if n_dropped:
        print(f"Info: filter dropped {n_dropped} of {len(df)} rows "
              f"(ext_params={ext_params}, unext_params={unext_params})")
    return filtered


def _filter_tag(
    ext_params: Dict[str, Any] | None = None,
    unext_params: Dict[str, Any] | None = None,
) -> str:
    """Build a short filesystem-safe tag from filter settings for filenames."""
    parts = []
    for label, params in [("ext", ext_params), ("unext", unext_params)]:
        if not params:
            continue
        for k, v in sorted(params.items()):
            short_key = k.rsplit(".", 1)[-1]
            if v is None:
                parts.append(short_key)
            else:
                parts.append(f"{short_key}={v}")
    tag = "_".join(parts)
    tag = re.sub(r"[^\w.=-]", "_", tag)
    return tag


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------

def process_folder(
    folder: Path,
    round_digits: int = 1,
    as_percent: bool = True,
    include_n: bool = False,
    exclude_from_percent: list = None,
    type_filter: str = "all",
    metric_filter: list = None,
    metric_mapping: dict = None,
    return_all: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    raw_df = res_to_df(folder, type_filter=type_filter)
    aggregated_df, hash_to_config = aggregate_runs(raw_df, folder=folder)
    combined_df = combine_mean_std(
        aggregated_df,
        round_digits=round_digits,
        as_percent=as_percent,
        include_n=include_n,
        exclude_from_percent=exclude_from_percent,
    )
    if metric_filter is not None or metric_mapping is not None:
        combined_df = filter_metrics(
            combined_df, metric_names=metric_filter, mapping=metric_mapping)
    if return_all:
        return raw_df, aggregated_df, combined_df, hash_to_config
    return combined_df


def load_results_dataframe(
    folder: Path,
    type_filter: str = "finetune",
    round_digits: int = 1,
    as_percent: bool = True,
    save_to_csv: bool = False,
) -> tuple[pd.DataFrame, Dict[str, Dict]]:
    """Load, aggregate and format results from *folder*.

    Returns ``(combined_df, hash_to_config)``.  ``changed_parameters``
    is **not** derived here — that is done in
    :func:`load_results_dataframes` after all folders have been merged,
    so the majority comparison spans the full dataset.

    Parameters
    ----------
    type_filter
        ``"dense"``, ``"finetune"``, ``"all"``, ``"comparison"``, or
        ``"both"`` (dense + finetune side-by-side).
    """
    folder = Path(folder)

    if type_filter == "comparison":
        # Dense vs finetune delta comparison, paired by base_hash
        _, dense_agg, _, htc_d = process_folder(
            folder, type_filter="dense", round_digits=round_digits,
            as_percent=as_percent, return_all=True)
        _, finetune_agg, _, htc_f = process_folder(
            folder, type_filter="finetune", round_digits=round_digits,
            as_percent=as_percent, return_all=True)
        hash_to_config = {**htc_d, **htc_f}
        combined_df = compare_dense_finetune(
            dense_agg, finetune_agg,
            round_digits=round_digits, as_percent=as_percent)

    elif type_filter == "both":
        # Dense + finetune aggregated separately, then concatenated
        _, dense_agg, _, htc_d = process_folder(
            folder, type_filter="dense", round_digits=round_digits,
            as_percent=as_percent, return_all=True)
        _, finetune_agg, _, htc_f = process_folder(
            folder, type_filter="finetune", round_digits=round_digits,
            as_percent=as_percent, return_all=True)

        unified_agg = pd.concat(
            [dense_agg, finetune_agg], ignore_index=True)
        hash_to_config = {**htc_d, **htc_f}

        combined_df = combine_mean_std(
            unified_agg, round_digits=round_digits, as_percent=as_percent)

    else:
        result = process_folder(
            folder, type_filter=type_filter,
            round_digits=round_digits, as_percent=as_percent,
            return_all=True)
        _, _, combined_df, hash_to_config = result

    if save_to_csv:
        output_file = folder / f"aggregated_results_{type_filter}.csv"
        combined_df.to_csv(output_file, index=False)
        print(f"✓ Saved DataFrame to: {output_file}")

    return combined_df, hash_to_config


def _apply_changed_parameters(
    df: pd.DataFrame,
    hash_to_config: Dict[str, Dict],
    ext_params: Dict[str, Any] | None = None,
    unext_params: Dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Derive ``changed_parameters`` and reorder columns.

    This is the single place where the full changed-parameters pipeline
    runs: majority derivation → save compact copy → extend → expand →
    filter → column reorder.
    """
    if hash_to_config:
        df = _derive_majority_changed_parameters(df, hash_to_config)

    if "changed_parameters" in df.columns:
        df["unext_changed_parameters"] = df["changed_parameters"].copy()

    if hash_to_config:
        df = _extend_changed_parameters(df, hash_to_config)

    df = expand_changed_parameters(df)

    if ext_params or unext_params:
        df = filter_by_changed_parameters(
            df, ext_params=ext_params, unext_params=unext_params)

    # Column ordering: unext_changed_parameters first, hashes last
    front = []
    if "unext_changed_parameters" in df.columns:
        front.append("unext_changed_parameters")
    if "changed_parameters" in df.columns:
        front.append("changed_parameters")
    for c in ("type", "n_samples"):
        if c in df.columns:
            front.append(c)
    for c in ("filename", "source_folder"):
        if c in df.columns:
            front.append(c)
    tail = []
    for c in ("group_key", "base_hash", "config_hash"):
        if c in df.columns:
            tail.append(c)
    middle = [c for c in df.columns if c not in front and c not in tail]
    df = df[front + middle + tail]

    return df


def load_results_dataframes(
    folders: List[Path] | Path,
    type_filter: str = "finetune",
    round_digits: int = 1,
    as_percent: bool = True,
    save_to_csv: bool = False,
    add_source_column: bool = True,
    save_merged: bool = False,
    default_save_dir: Optional[Path] = None,
    ext_params: Dict[str, Any] | None = None,
    unext_params: Dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Load and merge results from multiple experiment folders.

    Aggregated (pre-format) data is collected from each folder so that
    duplicate ``group_key`` entries (same config in different folders)
    can be properly re-aggregated using weighted mean / pooled std.
    ``combine_mean_std`` and ``changed_parameters`` are computed **once**
    on the merged dataset.

    Parameters
    ----------
    ext_params : dict, optional
        ``{param_name: value | None}`` — checked against expanded
        (extended) DataFrame columns.  ``None`` = require presence;
        set value = require exact match.
    unext_params : dict, optional
        ``{param_name: value | None}`` — checked against the compact
        ``unext_changed_parameters`` string.  ``None`` = require
        presence; set value = require exact match.
    """
    if isinstance(folders, (str, Path)):
        folders = [Path(folders)]
    else:
        folders = [Path(f) for f in folders]

    merged_htc: Dict[str, Dict] = {}

    # "comparison" mode produces delta DataFrames that cannot be
    # re-aggregated; fall back to per-folder formatted concatenation.
    if type_filter == "comparison":
        dataframes: List[pd.DataFrame] = []
        for folder in folders:
            df, htc = load_results_dataframe(
                folder, type_filter=type_filter,
                round_digits=round_digits, as_percent=as_percent,
                save_to_csv=save_to_csv)
            if not df.empty:
                if add_source_column:
                    df["source_folder"] = folder.name
                dataframes.append(df)
                merged_htc.update(htc)
            else:
                print(f"Warning: No results found in {folder}")
        if not dataframes:
            print("Warning: No results found in any folder")
            return pd.DataFrame()
        merged_df = pd.concat(dataframes, ignore_index=True)
        merged_df = _apply_changed_parameters(
            merged_df, merged_htc,
            ext_params=ext_params, unext_params=unext_params)
    else:
        # Collect pre-format aggregated data from each folder, then
        # re-aggregate duplicate group_keys and format once.
        type_filters = (
            ["dense", "finetune"] if type_filter == "both"
            else [type_filter]
        )
        agg_dataframes: List[pd.DataFrame] = []
        for folder in folders:
            folder = Path(folder)
            for tf in type_filters:
                result = process_folder(
                    folder, type_filter=tf,
                    round_digits=round_digits, as_percent=as_percent,
                    return_all=True)
                _, agg_df, combined_df, htc = result

                if save_to_csv:
                    out = folder / f"aggregated_results_{tf}.csv"
                    combined_df.to_csv(out, index=False)
                    print(f"✓ Saved DataFrame to: {out}")

                if not agg_df.empty:
                    if add_source_column:
                        agg_df = agg_df.copy()
                        agg_df["source_folder"] = folder.name
                    agg_dataframes.append(agg_df)
                    merged_htc.update(htc)
                else:
                    print(f"Warning: No {tf} results found in {folder}")

        if not agg_dataframes:
            print("Warning: No results found in any folder")
            return pd.DataFrame()

        merged_agg = pd.concat(agg_dataframes, ignore_index=True)
        merged_agg = _reaggregate_duplicates(merged_agg)

        merged_df = combine_mean_std(
            merged_agg, round_digits=round_digits, as_percent=as_percent)

        merged_df = _apply_changed_parameters(
            merged_df, merged_htc,
            ext_params=ext_params, unext_params=unext_params)

    if save_merged:
        save_dir = Path(
            default_save_dir) if default_save_dir else get_default_save_dir()
        tables_dir = save_dir / "tables"
        tables_dir.mkdir(parents=True, exist_ok=True)
        folder_names = "_".join(f.name for f in folders[:3])
        if len(folders) > 3:
            folder_names += f"_and_{len(folders) - 3}_more"
        tag = (f"_{_filter_tag(ext_params, unext_params)}"
               if ext_params or unext_params else "")
        output_file = tables_dir / \
            f"merged_results_{type_filter}{tag}_{folder_names}.csv"
        merged_df.to_csv(output_file, index=False)
        print(f"✓ Saved merged DataFrame to: {output_file}")

    return merged_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    type_filter = "finetune"  # Options: "finetune", "dense", "all", "comparison", "both"

    base_folders = [
        # "/home/zimmerro/tmp/dinov2/CUB2011/CVPR_2026/1-N_f_star-N_f_c",
        # "/home/zimmerro/tmp/dinov2/CUB2011/CVPR_2026/qpm",
        "home/zimmerro/tmp/dinov2/CUB2011/Masterarbeit_Experiments/MAS5-losses",
        "/home/zimmerro/tmp/dinov2/CUB2011/Masterarbeit_Experiments/MAS4-nn",
        "/home/zimmerro/tmp/dinov2/CUB2011/Masterarbeit_Experiments/MAS2-n_features-hidden_size",
    ]

    df = load_results_dataframes(
        folders=base_folders,
        type_filter=type_filter,
        round_digits=1,
        as_percent=True,
        save_to_csv=False,
        save_merged=True,
        ext_params={
            # "finetune.n_features": 50,
            # "finetune.n_per_class": 5,
        },
        unext_params={
            # "model.hidden_size": None,
            # "model.n_features": None,
        },
    )

    print("Completed processing results dataframe.")
