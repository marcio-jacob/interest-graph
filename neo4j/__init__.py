"""
neo4j/__init__.py
=================
Bridge: re-exports the real installed neo4j library while allowing
project-specific submodules (connection, schema, loader) to live here.

Because this local package directory shadows the installed `neo4j` library
we temporarily swap sys.modules to let the real package initialise, then
restore ourselves and keep all names (including submodules) in our namespace.
"""

from __future__ import annotations

import os
import sys

# ---------------------------------------------------------------------------
# Locate the real installed neo4j in site-packages
# ---------------------------------------------------------------------------
_real_site = next(
    (p for p in sys.path
     if "site-packages" in p
     and os.path.isfile(os.path.join(p, "neo4j", "_meta.py"))),
    None,
)

if _real_site is not None:
    # Temporarily remove ourselves so the real package can load as "neo4j"
    _self_mod = sys.modules.pop("neo4j", None)
    sys.path.insert(0, _real_site)
    try:
        import neo4j as _real_neo4j  # loads from site-packages
    finally:
        sys.path.remove(_real_site)
        # Restore our local package as the canonical "neo4j" entry
        if _self_mod is not None:
            sys.modules["neo4j"] = _self_mod

    # Re-export every public name from the real package so that third-party
    # libraries (e.g. graphdatascience) that do `import neo4j; neo4j.X`
    # find what they need even though 'neo4j' resolves to this local package.
    for _name in dir(_real_neo4j):
        if not _name.startswith("__"):
            try:
                globals()[_name] = getattr(_real_neo4j, _name)
            except Exception:
                pass

    # Explicitly expose submodules that graphdatascience accesses as
    # `neo4j.<submodule>` — these live in sys.modules but aren't always
    # in dir() of the real package.
    import importlib as _importlib
    for _submod in ("exceptions", "auth", "time", "spatial", "graph",
                    "query", "types", "data", "addressing", "meta",
                    "io", "api", "conf", "packstream"):
        _full = f"neo4j.{_submod}"
        if _full in sys.modules:
            globals()[_submod] = sys.modules[_full]
        else:
            try:
                _m = _importlib.import_module(_full)
                globals()[_submod] = _m
            except ImportError:
                pass

    __version__ = getattr(_real_neo4j, "__version__", "5.0.0")

    del _self_mod, _real_neo4j, _real_site, _importlib, _submod, _name
