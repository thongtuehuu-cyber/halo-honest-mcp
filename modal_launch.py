"""Track #11 Modal serverless GPU launcher for Halo Honest-MCP Environmental GRPO."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

try:
    import modal
except ImportError:  # Allow local syntax/import verification without modal installed
    modal = None  # type: ignore[assignment]


APP_NAME = "halo-honest-mcp-competition"
HALO_IMAGE_URI = "public.ecr.aws/whitecircle/halo:hopper-1.3.0rc1"
VOLUME_NAME = "halo-honest-mcp-artifacts"
REMOTE_WORK_DIR = "/workspace"
REMOTE_VOLUME_PATH = "/vol/halo_runs"


def build_halo_modal_image() -> Any:
    """Build Modal image clearing Halo's banner entrypoint per Issue #223 and wiring ``honest_mcp``."""
    if modal is None:
        raise RuntimeError("Install the `modal` package to deploy or run on Modal GPUs.")
    return (
        modal.Image.from_registry(HALO_IMAGE_URI)
        .entrypoint([])  # Required so Modal runs the function instead of printing image banner and exiting
        .add_local_dir(
            str(Path(__file__).resolve().parent),
            remote_path="/workspace/honest_mcp_entry",
        )
        .run_commands("python /workspace/honest_mcp_entry/install_into_halo.py --halo-root /workspace")
    )


def register_and_launch_command(config_path: str) -> list[str]:
    """Return the exact CLI command invoking ``halo launch grpo/environmental`` with ``PYTHONPATH=/workspace``."""
    launcher_snippet = (
        "import os, sys; "
        "os.environ['PYTHONPATH'] = '/workspace:' + os.environ.get('PYTHONPATH', ''); "
        "sys.path.insert(0, '/workspace'); "
        "from halo.cli import main; "
        "sys.argv = ['halo', 'launch', 'grpo/environmental', sys.argv[1]]; "
        "main()"
    )
    return [sys.executable, "-c", launcher_snippet, config_path]


def whitecircle_policy_sync_command(
    environment_id: str,
    policies_path: str = "/workspace/honest_mcp_entry/configs/whitecircle_honesty_policies.json",
    *,
    dry_run: bool = False,
) -> list[str]:
    """Build the ``halo run create-whitecircle-policies`` CLI command for White Circle setup."""
    cmd = [
        "halo",
        "run",
        "create-whitecircle-policies",
        "--environment-id",
        environment_id,
        "--policies",
        policies_path,
    ]
    if dry_run:
        cmd.append("--dry-run")
    return cmd


if modal is not None:
    app = modal.App(APP_NAME)
    image = build_halo_modal_image()
    artifacts_vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

    @app.function(
        image=image,
        secrets=[modal.Secret.from_name("halo-competition-secrets")],
    )
    def sync_whitecircle_policies(environment_id: str | None = None, dry_run: bool = False) -> dict[str, Any]:
        """Provision or verify White Circle honesty checklist/policies for ``environment_id``."""
        os.chdir(REMOTE_WORK_DIR)
        env_id = (environment_id or os.environ.get("WHITECIRCLE_ENV_ID", "")).strip()
        if not env_id:
            raise ValueError("Pass --environment-id or set WHITECIRCLE_ENV_ID in halo-competition-secrets")
        cmd = whitecircle_policy_sync_command(env_id, dry_run=dry_run)
        proc = subprocess.run(cmd, cwd=REMOTE_WORK_DIR, capture_output=True, text=True, check=False)
        return {"returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}

    @app.function(
        image=image,
        gpu="H100",
        timeout=60 * 60 * 4,
        volumes={REMOTE_VOLUME_PATH: artifacts_vol},
        secrets=[
            modal.Secret.from_name("halo-competition-secrets")
        ],  # Expected keys: WHITECIRCLE_API_KEY, WHITECIRCLE_ENV_ID, HF_TOKEN, OPENROUTER_API_KEY
    )
    def run_honest_mcp_grpo(
        config_relpath: str = "honest_mcp_entry/configs/honest_mcp_grpo_template.yaml",
    ) -> dict[str, object]:
        """Run Honest-MCP Environmental GRPO inside /workspace on Modal H100 and persist artifacts."""
        os.chdir(REMOTE_WORK_DIR)
        cfg_abs = str((Path(REMOTE_WORK_DIR) / config_relpath).resolve())
        env = dict(os.environ)
        env["PYTHONPATH"] = f"{REMOTE_WORK_DIR}:{env.get('PYTHONPATH', '')}".rstrip(":")
        cmd = register_and_launch_command(cfg_abs)
        proc = subprocess.run(cmd, cwd=REMOTE_WORK_DIR, env=env, check=False)
        artifacts_vol.commit()
        return {"returncode": proc.returncode, "config": cfg_abs, "volume": REMOTE_VOLUME_PATH}

    @app.function(
        image=image,
        volumes={REMOTE_VOLUME_PATH: artifacts_vol},
    )
    def export_completions_json(
        run_dir: str = f"{REMOTE_VOLUME_PATH}/honest-mcp-grpo",
        out_file: str = f"{REMOTE_VOLUME_PATH}/lovable_export.json",
    ) -> str:
        """Convert saved Halo ``completions/*.parquet`` into Lovable Arena JSON."""
        import glob
        import pandas as pd

        os.chdir(REMOTE_WORK_DIR)
        files = sorted(glob.glob(f"{run_dir}/completions/*.parquet"))
        records = []
        for path in files:
            df = pd.read_parquet(path)
            records.extend(df.to_dict(orient="records"))
        payload = {"total": len(records), "trajectories": records[:200]}
        Path(out_file).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        artifacts_vol.commit()
        return out_file

    @app.local_entrypoint()
    def main(
        config: str = "honest_mcp_entry/configs/honest_mcp_grpo_template.yaml",
        export_only: bool = False,
        sync_policies: bool = False,
        dry_run: bool = False,
    ) -> None:
        if sync_policies:
            res = sync_whitecircle_policies.remote(dry_run=dry_run)
            print(json.dumps(res, indent=2))
            return
        if export_only:
            out = export_completions_json.remote()
            print(f"Exported Lovable JSON to volume path: {out}")
            return
        res = run_honest_mcp_grpo.remote(config_relpath=config)
        print(json.dumps(res, indent=2))
