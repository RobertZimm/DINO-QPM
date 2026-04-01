import itertools

import yaml
from dino_qpm.configs.core.hp_sweep_params import full_vals, reduced_vals, param_mapping
from dino_qpm.helpers.dict_op import find_and_update_key_value, update_dict


def update_config(config: dict,
                  param: str,
                  val: str | int | float,
                  mode: str):
    if param == "approach":
        config["model"]["arch_type"] = val[0]
        config["model"]["feat_vec_type"] = val[1]

    elif param in ["ignore_first_n_components", "n_clusters", "rpl_weight" "epochs_per_loop", "gamma"]:
        config["projection"][param] = val

    elif param == "model_type":
        config["model_type"] = val

    elif param == "fitzpatrick_split":
        config["data"][param] = val

    elif param == "best_approaches":
        config["model"]["arch_type"] = val[0]
        config["model"]["feat_vec_type"] = val[1]
        config["model_type"] = val[2]

    elif param == "qpm_sel_pairs":
        config["finetune"]["n_features"] = val[0]
        config["finetune"]["n_per_class"] = val[1]

    elif param in config["model"].keys():
        config["model"][param] = val

        if config["model"]["proto_similarity_method"] == "rbf" or config["model"]["proto_similarity_method"] == "log_l2":
            config["finetune"]["cofs_weight"] = 0.0
            config["dense"]["cofs_weight"] = 0.0

    elif param == "pooling_type":
        if isinstance(val, str):
            config["model"]["pooling_type"] = val
        else:
            pool, init_method = val
            config["model"]["pooling_type"] = pool
            config["model"]["init_method"] = init_method

    elif param == "n_f_star":
        config["finetune"]["n_features"] = val

    elif param == "n_f_c":
        config["finetune"]["n_per_class"] = val

    else:
        find_and_update_key_value(data_dict=config,
                                  target_key=param,
                                  new_value=val,
                                  set_value_flag=True,
                                  discriminator_key=mode,
                                  debug=False)


def determine_arr_string(config_path: str, only_print: bool = True):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    param_names = config.get("sweep_param_names", [None])
    if param_names is None:
        param_names = [None]

    density_mode = config.get("density_mode", "full")
    comb_strat = config.get("comb_strat", "cross")

    elements = determine_arr_elements(param_names=param_names,
                                      density_mode=density_mode,
                                      comb_strat=comb_strat)

    if elements == 0:
        result = "0"
    else:
        result = f"0-{elements - 1}"

    print(result)
    if not only_print:
        return result


def determine_arr_elements(param_names: list[str],
                           density_mode: str = "full",
                           comb_strat: str = "cross"):
    if param_names is None:
        return 0

    if len(param_names) == 1:
        if param_names[0] is None:
            # print(">>> No modes specified, nothing to sweep, returning 0")
            return 0

    param_names, applied_in_modes = process_param_names(param_names)

    combined_vals, _, _ = prod_combined_vals(param_names=param_names,
                                             density_mode=density_mode,
                                             comb_strat=comb_strat,
                                             applied_in_modes=applied_in_modes)

    return len(combined_vals)


def prod_combined_vals(param_names: list[str],
                       density_mode: str = "full",
                       comb_strat: str = "cross",
                       applied_in_modes: list[str] = None):
    if comb_strat == "cross":
        val_lists = []
        params = None

        if applied_in_modes is not None:
            modes = applied_in_modes

    elif comb_strat == "single":
        combined_vals = []
        params = []

        if applied_in_modes is not None:
            modes = []

    else:
        raise ValueError(
            f"Invalid combination strategy {comb_strat}. Choose 'cross' or 'single'.")

    for idx, param in enumerate(param_names):
        is_multi_param = isinstance(param, list)

        current_mode = applied_in_modes[idx] if applied_in_modes else "all"

        if not is_multi_param:
            vals = get_vals(param=param, density_mode=density_mode,
                            applied_mode=current_mode)
        else:
            vals = []
            for sub_idx, sub_param in enumerate(param):
                sub_mode = current_mode[sub_idx] if isinstance(
                    current_mode, list) else current_mode
                vals.append(get_vals(param=sub_param,
                            density_mode=density_mode,
                            applied_mode=sub_mode))

            vals = list(itertools.product(*vals))

        if comb_strat == "cross":
            val_lists.append(vals)

        elif comb_strat == "single":
            combined_vals.extend(vals)
            params.extend([param] * len(vals))

            if applied_in_modes is not None:
                modes.extend([applied_in_modes[idx]] * len(vals))

    if comb_strat == "cross":
        combined_vals = list(itertools.product(*val_lists))

    elif comb_strat == "single":
        pass

    if applied_in_modes is None:
        modes = None

    return combined_vals, params, modes


def sweep_params(run_number: int | None,
                 config: dict,
                 param_names: list[str],
                 mode: str,
                 debug: bool = False,
                 density_mode: str = "full",
                 comb_strat: str = "cross"):
    is_multi_list = any([isinstance(param, list) for param in param_names])

    if len(param_names) == 1:
        if param_names[0] is None:
            # print(
            #     ">>> No modes specified, nothing to sweep, returning config without changes")
            return

    if comb_strat == "cross" and is_multi_list:
        raise ValueError("Combination strategy 'cross' is not compatible with multi-valued parameters. "
                         "Use 'single' combination strategy instead.")

    param_names, applied_in_modes = process_param_names(param_names)

    combined_vals, params, applied_in_modes = prod_combined_vals(param_names=param_names,
                                                                 density_mode=density_mode,
                                                                 comb_strat=comb_strat,
                                                                 applied_in_modes=applied_in_modes)

    try:
        update_val = combined_vals[run_number]

        if debug:
            if comb_strat == "cross":
                print_par = tuple(param_names)
                appl_in = tuple(applied_in_modes)
                print(f">>> Run number: {run_number}; {print_par} = {update_val}"
                      f"; mode={appl_in}")
            else:
                if isinstance(params[run_number], list):
                    print_par = tuple(params[run_number])
                    appl_in = tuple(applied_in_modes[run_number])
                    print(f">>> Run number: {run_number}; {print_par} = {update_val}"
                          f"; mode={appl_in}")
                else:
                    print_par = params[run_number]
                    appl_in = applied_in_modes[run_number]

                    print(f">>> Run number: {run_number}; {print_par} = {update_val}"
                          f"; mode={appl_in}")

        if comb_strat == "cross":
            for param, val, appl in zip(param_names, update_val, applied_in_modes):
                if appl == "all" or appl == mode:
                    update_config(config=config,
                                  param=param,
                                  val=val,
                                  mode=mode, )

        elif comb_strat == "single":
            if not isinstance(params[run_number], list):
                param = params[run_number]
                val = update_val
                appl = applied_in_modes[run_number]

                if appl == "all" or appl == mode:
                    update_config(config=config,
                                  param=param,
                                  val=val,
                                  mode=mode, )

            else:
                mul_params = params[run_number]
                appl_in = applied_in_modes[run_number]

                for param, val, appl in zip(mul_params, update_val, appl_in):
                    if appl == "all" or appl == mode:
                        update_config(config=config,
                                      param=param,
                                      val=val,
                                      mode=mode, )

        return update_val

    except IndexError:
        raise ValueError(
            f"Invalid run number {run_number} for combined parameters with range 0-{len(combined_vals) - 1}")


def get_run_label(run_number: int,
                  param_names: list,
                  density_mode: str = "full",
                  comb_strat: str = "cross") -> str:
    """
    Return a human-readable directory label for *run_number* of the form
    ``key1-val1-key2-val2``, derived from the HP-sweep parameters.

    Used to replace the bare integer run_number in the log_dir path so the
    directory name captures exactly which parameter values are active.
    """
    param_names, applied_in_modes = process_param_names(param_names)
    combined_vals, params, _ = prod_combined_vals(
        param_names=param_names,
        density_mode=density_mode,
        comb_strat=comb_strat,
        applied_in_modes=applied_in_modes,
    )

    try:
        vals = combined_vals[run_number]
    except IndexError:
        raise ValueError(
            f"Invalid run number {run_number} for combined parameters "
            f"with range 0-{len(combined_vals) - 1}")

    parts: list[str] = []

    if comb_strat == "cross":
        # vals is a tuple aligned with param_names
        for param, val in zip(param_names, vals):
            if isinstance(param, list):
                for p, v in zip(param, val):
                    parts += [str(p), str(v)]
            else:
                parts += [str(param), str(val)]
    else:  # single
        param = params[run_number]
        val = vals
        if isinstance(param, list):
            for p, v in zip(param, val):
                parts += [str(p), str(v)]
        else:
            parts += [str(param), str(val)]

    return "-".join(parts)


def process_param_names(param_names: list[str]):
    fresh_param_names = []
    applied_in_modes = []
    for param_name in param_names:
        if isinstance(param_name, list):
            tmp_par = []
            tmp_modes = []

            for p in param_name:
                check_name(p, tmp_par, tmp_modes)

            fresh_param_names.append(tmp_par)
            applied_in_modes.append(tmp_modes)

        else:
            check_name(param_name, fresh_param_names, applied_in_modes)

    return fresh_param_names, applied_in_modes


def check_name(param_name: str, fresh_param_names: list[str], applied_in_modes: list[str]):
    if ";" in param_name:
        parts = param_name.split("; ")
        true_param = parts[0]
        appl_in = parts[1] if len(parts) > 1 else "all"

        if appl_in not in ["dense", "finetune", "all"]:
            appl_in = "all"

        fresh_param_names.append(true_param)
        applied_in_modes.append(appl_in)
    else:
        fresh_param_names.append(param_name)
        applied_in_modes.append("all")


def get_vals(param: str, density_mode: str = "full", applied_mode: str = "all"):
    if density_mode == "reduced":
        vals = update_dict(reduced_vals, full_vals)
    elif density_mode == "full":
        vals = full_vals
    else:
        raise ValueError(
            f"Invalid density mode {density_mode}. Choose 'full' or 'reduced'.")

    mode_specific_key = f"{param}; {applied_mode}" if applied_mode != "all" else None

    if mode_specific_key and mode_specific_key in vals.keys():
        return vals[mode_specific_key]

    if param not in vals.keys():
        mode_specific_param = f"{param}; {applied_mode}" if applied_mode != "all" else None

        for key, value in param_mapping.items():
            if param in value or (mode_specific_param and mode_specific_param in value):
                return vals[key]
        raise ValueError(
            f"Invalid sweep mode {param} for parameter mapping {param_mapping}")

    else:
        return vals[param]


if __name__ == "__main__":
    from pathlib import Path

    configs_dir = Path(__file__).parent.parent / "configs"

    # Collect all yaml config files from imagenet and other folders
    config_files = []
    for folder in ["imagenet", "other"]:
        folder_path = configs_dir / folder
        if folder_path.exists():
            for yaml_file in folder_path.rglob("*.yaml"):
                config_files.append(yaml_file)

    for config_path in sorted(config_files):
        relative_path = config_path.relative_to(configs_dir)
        print(f"\n{'=' * 70}")
        print(f"Config: {relative_path}")
        print('=' * 70)

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        param_names = config.get("sweep_param_names", [None])
        if param_names[0] is None:
            print("  No sweep_param_names defined, skipping...")
            continue

        density_mode = config.get("density_mode", "dense")
        comb_strat = config.get("comb_strat", "product")

        array_elements = determine_arr_elements(param_names=param_names,
                                                density_mode=density_mode,
                                                comb_strat=comb_strat)

        print(f"  Array elements: {array_elements}")
        print(f"  Params: {param_names}")

        for run_number in range(array_elements):
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)

            vals = sweep_params(run_number=run_number,
                                config=config,
                                param_names=param_names,
                                mode="finetune",
                                debug=True,
                                density_mode=density_mode,
                                comb_strat=comb_strat)
