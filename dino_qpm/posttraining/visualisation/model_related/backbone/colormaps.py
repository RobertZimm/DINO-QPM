from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import warnings


# ---------------------------------------------------------------------------
# Deprecated legacy palette — kept for backwards-compatibility only.
# Use get_default_cmaps() (colorblind-safe) instead.
# ---------------------------------------------------------------------------
_LEGACY_SOLID_COLORS: list[list[float]] = [
    [1, 0, 0],                             # red
    [0, 1, 0],                             # green
    [0, 0, 1],                             # blue
    [0, 1, 1],                             # cyan
    [1, 0, 1],                             # magenta
    [0.5, 0, 0.5],                         # purple
    [74 / 255, 4 / 255, 4 / 255],          # dark red
    [72 / 255, 60 / 255, 50 / 255],        # brown
    [0, 0.5, 0.5],                         # teal
    [1, 0.5, 0],                           # orange
]


def get_legacy_cmaps():
    """Return the old solid-colour colormaps.

    .. deprecated::
        This palette does not account for colour-vision deficiencies.
        Use :func:`get_default_cmaps` instead, which draws from the
        colorblind-safe ``COLORBLIND_SAFE_RGB`` palette.
    """
    warnings.warn(
        "get_legacy_cmaps() is deprecated — the legacy colour palette is "
        "not colorblind-safe.  Use get_default_cmaps() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    cmaps = [CustomCmap([1, 1, 1], c) for c in _LEGACY_SOLID_COLORS]
    return [convert_cmap_to_cv(x) for x in cmaps]


def get_default_cmaps(n_features: int | None = None):
    """Return solid-colour colormaps using the colorblind-safe palette.

    This imports :data:`COLORBLIND_SAFE_RGB` from *single_image_viz* so
    that both the individual per-feature heatmaps and the combined
    colour-coded heatmap use identical colours.
    """
    from dino_qpm.posttraining.visualisation.model_related.backbone.single_image_viz import (
        get_colorblind_safe_colors,
    )

    n = n_features if n_features is not None else 10
    palette = get_colorblind_safe_colors(n)
    cmaps = [CustomCmap([1, 1, 1], list(rgb)) for rgb in palette]
    colormaps = [convert_cmap_to_cv(x) for x in cmaps]
    return colormaps


def get_colormap(name: str, n_features: int = 1) -> list:
    """
    Get colormap(s) by name.

    Args:
        name: Colormap name. Options:
            - "solid" (default): Different solid color per feature
            - Any matplotlib colormap name: "viridis", "jet", "plasma", "inferno", etc.
        n_features: Number of features (used to generate enough colormaps)

    Returns:
        List of OpenCV-compatible colormaps
    """
    if name == "solid":
        return get_default_cmaps(n_features=n_features)

    # Use matplotlib colormap
    try:
        mpl_cmap = plt.get_cmap(name)
        cv_cmap = convert_cmap_to_cv(mpl_cmap)
        return [cv_cmap] * n_features
    except ValueError:
        print(f"⚠️ Unknown colormap '{name}', falling back to 'solid'")
        return get_default_cmaps()


def CustomCmap(from_rgb, to_rgb):
    # from color r,g,b
    r1, g1, b1 = from_rgb

    # to color r,g,b
    r2, g2, b2 = to_rgb

    cdict = {'red': ((0, r1, r1),
                     (1, r2, r2)),
             'green': ((0, g1, g1),
                       (1, g2, g2)),
             'blue': ((0, b1, b1),
                      (1, b2, b2))}

    cmap = LinearSegmentedColormap('custom_cmap', cdict)
    return cmap


def convert_cmap_to_cv(cmap, log=True, gamma=1.5):
    sm = plt.cm.ScalarMappable(cmap=cmap)
    color_range = sm.to_rgba(np.linspace(0, 1, 256), bytes=True)[:, 2::-1]
    if log:
        color_range = (color_range ** gamma) / (255 ** gamma)
        color_range = np.uint8(color_range * 255)

    return color_range.reshape(256, 1, 3)
