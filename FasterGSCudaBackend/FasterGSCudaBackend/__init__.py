from pathlib import Path

extension_dir = Path(__file__).parent.parent
__extension_name__ = extension_dir.name
__install_command__ = [
    "pip",
    "install",
    str(extension_dir),
    "--no-build-isolation",
]

try:
    from .torch_bindings.adam import FusedAdam
    from .torch_bindings.densification import add_noise, relocation_adjustment
    from .torch_bindings.filter3d import update_3d_filter
    from .torch_bindings.rasterization import (
        RasterizerSettings,
        diff_rasterize,
        rasterize,
        update_pruning_scores,
    )

    __all__ = [
        "diff_rasterize",
        "rasterize",
        "update_pruning_scores",
        "RasterizerSettings",
        "FusedAdam",
        "update_3d_filter",
        "relocation_adjustment",
        "add_noise",
    ]
except ImportError as e:
    raise ImportError(
        f"Failed to import {__extension_name__}. "
        f"Try reinstalling with: {' '.join(__install_command__)}"
    ) from e
