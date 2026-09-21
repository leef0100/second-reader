"""Load modality plugins from entry points against the model registry.

Entry point group `secondreader.plugins` maps modality id → plugin class exposing
`from_registry(registry_root, model_id, device)`. Which model is active is
resolved per modality from `SR_ACTIVE_MODEL_ID` (cxr for now) or the
`<registry>/ACTIVE_MODEL_ID` file. core never imports a modality package.
"""

from __future__ import annotations

from importlib.metadata import entry_points
from pathlib import Path

import structlog

from secondreader.core.plugins.registry import PluginRegistry

log = structlog.get_logger(__name__)


def resolve_active_model_id(registry_root: Path, configured: str | None) -> str | None:
    if configured:
        return configured
    f = Path(registry_root) / "ACTIVE_MODEL_ID"
    if f.exists():
        v = f.read_text().strip()
        return v or None
    return None


def load_plugins(
    registry_root: Path, active_model_id: str | None, device: str = "auto"
) -> PluginRegistry:
    reg = PluginRegistry()
    mid = resolve_active_model_id(registry_root, active_model_id)
    if mid is None:
        log.warning("no_active_model", registry_root=str(registry_root))
        return reg
    for ep in entry_points(group="secondreader.plugins"):
        cls = ep.load()
        try:
            plugin = cls.from_registry(Path(registry_root), mid, device)
        except FileNotFoundError as exc:
            log.error("plugin_load_failed", plugin=ep.name, model_id=mid, error=str(exc))
            continue
        reg.register(plugin)
        log.info("plugin_loaded", plugin=ep.name, model_version_id=plugin.model_version_id)
    return reg
