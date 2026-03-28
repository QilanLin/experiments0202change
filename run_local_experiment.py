from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


PKG_PATH = Path(__file__).resolve().parent
PKG_NAME = "exp_pkg"


def _load_module():
    pkg_spec = importlib.util.spec_from_file_location(
        PKG_NAME,
        PKG_PATH / "__init__.py",
        submodule_search_locations=[str(PKG_PATH)],
    )
    pkg = importlib.util.module_from_spec(pkg_spec)
    sys.modules[PKG_NAME] = pkg
    assert pkg_spec.loader is not None
    pkg_spec.loader.exec_module(pkg)

    mod_spec = importlib.util.spec_from_file_location(
        f"{PKG_NAME}.run_experiment",
        PKG_PATH / "run_experiment.py",
    )
    mod = importlib.util.module_from_spec(mod_spec)
    sys.modules[f"{PKG_NAME}.run_experiment"] = mod
    assert mod_spec.loader is not None
    mod_spec.loader.exec_module(mod)
    return mod


def main() -> None:
    mod = _load_module()
    mod.main()


if __name__ == "__main__":
    main()
