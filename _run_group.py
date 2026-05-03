from __future__ import annotations

import argparse
import importlib
import sys
from collections.abc import Mapping


def run_group(
    *,
    group_name: str,
    description: str,
    modules: Mapping[str, str],
) -> int:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("script", choices=sorted(modules), help=f"{group_name} script to run.")

    raw_args = sys.argv[1:]
    if not raw_args or raw_args[0] in {"-h", "--help"}:
        parser.print_help()
        return 0

    script_name = raw_args[0]
    if script_name not in modules:
        parser.error(f"invalid choice: {script_name!r} (choose from {', '.join(sorted(modules))})")

    passthrough_args = raw_args[1:]
    module = importlib.import_module(modules[script_name])

    entrypoint = getattr(module, "main", None)
    if entrypoint is None:
        raise AttributeError(f"{modules[script_name]} does not define a main() function.")

    sys.argv = [f"{script_name}.py", *passthrough_args]
    result = entrypoint()
    return result if isinstance(result, int) else 0
