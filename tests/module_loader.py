from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


PKG_ALIAS = "exp_pkg"
PKG_DIR = Path(__file__).resolve().parents[1]


def ensure_package() -> str:
    """Register the package directory under an importable alias for tests."""
    if PKG_ALIAS in sys.modules:
        return PKG_ALIAS

    spec = importlib.util.spec_from_file_location(
        PKG_ALIAS,
        PKG_DIR / "__init__.py",
        submodule_search_locations=[str(PKG_DIR)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to create spec for test package at {PKG_DIR}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[PKG_ALIAS] = module
    spec.loader.exec_module(module)
    return PKG_ALIAS


def load_module(module_name: str) -> ModuleType:
    """Load a package submodule through the alias so relative imports still work."""
    pkg_name = ensure_package()
    qualified_name = f"{pkg_name}.{module_name}"
    if qualified_name in sys.modules:
        return sys.modules[qualified_name]

    spec = importlib.util.spec_from_file_location(
        qualified_name,
        PKG_DIR / f"{module_name}.py",
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module {module_name} from {PKG_DIR}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified_name] = module
    spec.loader.exec_module(module)
    return module
