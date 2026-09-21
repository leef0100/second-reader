"""Discovers and routes to modality plugins.

Plugins register through the `secondreader.plugins` entry-point group or
explicitly via `PluginRegistry.register`. `core` never imports a modality
package directly.
"""

from __future__ import annotations

from importlib.metadata import entry_points
from typing import Any

from secondreader.core.plugins.protocol import ModalityPlugin


class PluginRegistry:
    def __init__(self) -> None:
        self._plugins: dict[str, ModalityPlugin] = {}

    def register(self, plugin: ModalityPlugin) -> None:
        mid = plugin.info.modality_id
        if mid in self._plugins:
            raise ValueError(f"plugin already registered for modality {mid!r}")
        self._plugins[mid] = plugin

    def get(self, modality_id: str) -> ModalityPlugin:
        return self._plugins[modality_id]

    def all(self) -> list[ModalityPlugin]:
        return list(self._plugins.values())

    def route(self, header: dict[str, Any]) -> ModalityPlugin | None:
        """First plugin whose `accepts` says yes; None means 'no plugin for this'."""
        for p in self._plugins.values():
            if p.accepts(header):
                return p
        return None

    def load_entry_points(self) -> None:
        for ep in entry_points(group="secondreader.plugins"):
            factory = ep.load()
            self.register(factory())


_registry: PluginRegistry | None = None


def get_plugin_registry() -> PluginRegistry:
    global _registry
    if _registry is None:
        _registry = PluginRegistry()
    return _registry
