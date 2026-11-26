"""Idempotently wire ``honest_mcp`` into a Halo checkout's ``halo/environments/registry.py``."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


IMPORT_LINE = "from honest_mcp_entry.halo_honest_mcp_env import HonestMCPEnvironment"
REGISTER_LINE = '    register_environment("honest_mcp", lambda c: HonestMCPEnvironment(**c))'


def patch_registry_text(text: str) -> tuple[str, bool]:
    """Insert ``honest_mcp`` import and ``_register_builtins()`` entry if missing."""
    has_import = IMPORT_LINE in text
    has_register = REGISTER_LINE in text
    if has_import and has_register:
        return text, False

    updated = text
    if not has_import:
        anchor = "from halo.environments.protocols.mcp import build_mcp_environment"
        if anchor not in updated:
            raise ValueError(f"Expected anchor {anchor!r} in Halo registry.py")
        updated = updated.replace(anchor, f"{anchor}\n{IMPORT_LINE}", 1)

    if not has_register:
        anchor = '    register_environment("mcp", lambda c: build_mcp_environment(**c))'
        if anchor not in updated:
            raise ValueError(f"Expected anchor {anchor!r} in Halo registry.py")
        updated = updated.replace(anchor, f"{anchor}\n{REGISTER_LINE}", 1)

    return updated, True


def install_into_halo(halo_root: Path, *, dry_run: bool = False, check: bool = False) -> int:
    registry_path = halo_root / "halo" / "environments" / "registry.py"
    if not registry_path.is_file():
        print(f"ERROR: registry.py not found at {registry_path}", file=sys.stderr)
        return 2

    original = registry_path.read_text(encoding="utf-8")
    updated, changed = patch_registry_text(original)
    if check:
        if changed:
            print(f"MISSING: honest_mcp is not wired in {registry_path}")
            return 1
        print(f"OK: honest_mcp already registered in {registry_path}")
        return 0

    if not changed:
        print(f"Already up to date: {registry_path}")
        return 0

    if dry_run:
        print(f"DRY-RUN: would update {registry_path}")
        return 0

    registry_path.write_text(updated, encoding="utf-8")
    print(f"Updated {registry_path} with honest_mcp registration.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Register honest_mcp in halo/environments/registry.py for Ray actors.")
    parser.add_argument("--halo-root", type=Path, default=Path("/workspace"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    return install_into_halo(args.halo_root.resolve(), dry_run=args.dry_run, check=args.check)


if __name__ == "__main__":
    raise SystemExit(main())
