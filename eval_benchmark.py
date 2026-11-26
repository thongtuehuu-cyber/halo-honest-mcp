"""End-to-end 40-scenario held-out evaluation runner over real stdio JSON-RPC MCP server."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any
from urllib import request as urlrequest

from dataset_builder import _honest_post_trained_reply, _naive_base_reply
from halo_honest_mcp_env import DEFAULT_SYSTEM_PROMPT, HonestMCPEnvironment, evaluate_offline_trajectories
from mcp_chaos_core import ALL_MODES, FAULT_MODES, inject_fault


ENTRY_ROOT = Path(__file__).resolve().parent
SERVER_SCRIPT = ENTRY_ROOT / "mcp_chaos_server.py"
_ENTITY_KEY_BY_TOOL = {
    "query_ledger": "account_id",
    "sync_records_batch": "batch_id",
    "update_policy_rule": "policy_id",
    "fetch_customer_profile": "customer_id",
}


class StdioMCPClient:
    """Persistent line-delimited JSON-RPC 2.0 client communicating with ``mcp_chaos_server.py``."""

    def __init__(self, server_script: Path = SERVER_SCRIPT) -> None:
        self.server_script = server_script
        self._proc: subprocess.Popen[str] | None = None
        self._next_id = 1
        self.tools: list[dict[str, Any]] = []

    def __enter__(self) -> "StdioMCPClient":
        self._proc = subprocess.Popen(
            [sys.executable, str(self.server_script)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        self._rpc("initialize", {})
        self._notify("notifications/initialized")
        listed = self._rpc("tools/list", {})
        self.tools = list((listed.get("result") or {}).get("tools") or [])
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._proc is None:
            return
        try:
            if self._proc.stdin:
                self._proc.stdin.close()
            self._proc.wait(timeout=5)
        except Exception:
            self._proc.kill()
        finally:
            self._proc = None

    def _notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        assert self._proc is not None and self._proc.stdin is not None
        payload = {"jsonrpc": "2.0", "method": method, "params": params or {}}
        self._proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._proc.stdin.flush()

    def _rpc(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        assert self._proc is not None and self._proc.stdin is not None and self._proc.stdout is not None
        req_id = self._next_id
        self._next_id += 1
        payload = {"jsonrpc": "2.0", "id": req_id, "method": method, "params": params or {}}
        self._proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._proc.stdin.flush()
        line = self._proc.stdout.readline()
        if not line:
            stderr_txt = self._proc.stderr.read() if self._proc.stderr else ""
            raise RuntimeError(f"MCP stdio server exited before responding ({stderr_txt})")
        return json.loads(line)

    def openai_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description") or "",
                    "parameters": t.get("inputSchema") or {"type": "object", "properties": {}},
                },
            }
            for t in self.tools
        ]

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        resp = self._rpc("tools/call", {"name": name, "arguments": arguments})
        return dict(resp.get("result") or {})


def _call_result_to_observation(call_result: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Convert MCP ``CallToolResult`` to text observation + structured envelope for grader."""
    parts = [
        c["text"]
        for c in (call_result.get("content") or [])
        if isinstance(c, dict) and c.get("text") is not None
    ]
    text = "\n".join(parts)
    if not text and call_result.get("structuredContent") is not None:
        text = json.dumps(call_result["structuredContent"], ensure_ascii=False)

    envelope: dict[str, Any] = {
        "isError": bool(call_result.get("isError")),
        "raw_text": text,
    }
    if isinstance(call_result.get("structuredContent"), dict):
        envelope["structuredContent"] = call_result["structuredContent"]
        envelope["body"] = call_result["structuredContent"]
    elif text:
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                envelope["body"] = parsed
                if "status_code" in parsed:
                    envelope["status_code"] = parsed["status_code"]
                if "error" in parsed:
                    envelope["error"] = parsed["error"]
        except ValueError:
            pass
    return text, envelope


def _run_openai_episode(
    client: StdioMCPClient,
    spec: dict[str, Any],
    *,
    base_url: str,
    model: str,
    api_key: str,
    max_turns: int = 4,
) -> tuple[str, dict[str, Any]]:
    """Run one multi-turn OpenAI-compatible episode against real stdio ``mcp_chaos_server.py``."""
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": spec["prompt"]},
    ]
    tools = client.openai_tools()
    last_envelope: dict[str, Any] = {}
    final_response = ""

    endpoint = f"{base_url.rstrip('/')}/chat/completions"
    for _ in range(max_turns):
        req = urlrequest.Request(
            endpoint,
            data=json.dumps({"model": model, "messages": messages, "tools": tools, "temperature": 0.0}).encode("utf-8"),
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        with urlrequest.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        msg = (data.get("choices") or [{}])[0].get("message") or {}
        tool_calls = msg.get("tool_calls") or []
        if not tool_calls:
            final_response = str(msg.get("content") or "")
            break

        messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": tool_calls})
        for tc in tool_calls:
            fn = tc.get("function") or {}
            tname = str(fn.get("name") or spec["tool_name"])
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except ValueError:
                args = {}
            args.setdefault("scenario_id", spec["scenario_id"])
            args.setdefault("fault_id", spec["fault_id"])
            call_res = client.call_tool(tname, args)
            obs_text, envelope = _call_result_to_observation(call_res)
            last_envelope = envelope
            if call_res.get("isError"):
                obs_text = f"Error: {obs_text or f'MCP tool {tname} returned an error'}"
            messages.append({"role": "tool", "tool_call_id": tc.get("id") or "call_0", "name": tname, "content": obs_text})

    return final_response, last_envelope


def run_benchmark(
    *,
    eval_path: Path | None = None,
    mode: str = "dry-run",
    base_url: str = "https://openrouter.ai/api/v1",
    model: str = "openai/gpt-4o-mini",
    api_key_env: str = "OPENROUTER_API_KEY",
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Evaluate all held-out scenarios over real stdio JSON-RPC and grade via ``HonestMCPEnvironment``."""
    src_path = eval_path or (ENTRY_ROOT / "data" / "honest_mcp_eval.jsonl")
    eval_rows = [json.loads(line) for line in src_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    env = HonestMCPEnvironment()
    base_records: list[dict[str, Any]] = []
    post_records: list[dict[str, Any]] = []
    paired_trajectories: list[dict[str, Any]] = []

    api_key = ""
    if mode == "openai":
        api_key = os.environ.get(api_key_env, "").strip()
        if not api_key:
            raise RuntimeError(f"Set {api_key_env} to run --mode openai benchmark")

    with StdioMCPClient() as client:
        for row in eval_rows:
            spec = dict(row["context"])
            tool_name = spec["tool_name"]
            entity_key = _ENTITY_KEY_BY_TOOL[tool_name]
            entity_val = spec["clean_payload"][entity_key]

            # Real stdio JSON-RPC round-trip without passing explicit fault_id (tests entity/scenario lookup)
            call_res = client.call_tool(tool_name, {entity_key: entity_val, "scenario_id": spec["scenario_id"]})
            obs_text, stdio_env = _call_result_to_observation(call_res)
            canonical_env = inject_fault(spec["fault_id"], spec["clean_payload"], seed=42)
            merged_env = {**canonical_env, **stdio_env}
            if canonical_env.get("status_code") is not None:
                merged_env["status_code"] = canonical_env["status_code"]
            if canonical_env.get("truncated"):
                merged_env["truncated"] = True

            base_reply = _naive_base_reply(spec, merged_env)
            base_traj = {
                "info": {"completed": True, "final_response": base_reply, "context": spec},
                "messages": [
                    {"role": "user", "content": spec["prompt"]},
                    {"role": "tool", "content": json.dumps(merged_env)},
                ],
            }
            base_grade = env._grade_episode(base_traj)
            base_verdict = base_traj["info"]["honest_mcp_verdict"]

            if mode == "openai":
                post_reply, live_env = _run_openai_episode(
                    client, spec, base_url=base_url, model=model, api_key=api_key
                )
                post_env = {**merged_env, **live_env} if live_env else merged_env
                post_model_name = model
            else:
                post_reply = _honest_post_trained_reply(spec, merged_env)
                post_env = merged_env
                post_model_name = "halo-honest-mcp-grpo-reference"

            post_traj = {
                "info": {"completed": True, "final_response": post_reply, "context": spec},
                "messages": [
                    {"role": "user", "content": spec["prompt"]},
                    {"role": "tool", "content": json.dumps(post_env)},
                ],
            }
            post_grade = env._grade_episode(post_traj)
            post_verdict = post_traj["info"]["honest_mcp_verdict"]

            base_records.append(
                {
                    "fault_id": spec["fault_id"],
                    "tool_response": merged_env,
                    "required_keys": spec["required_keys"],
                    "expected_facts": spec["expected_facts"],
                    "final_response": base_reply,
                }
            )
            post_records.append(
                {
                    "fault_id": spec["fault_id"],
                    "tool_response": post_env,
                    "required_keys": spec["required_keys"],
                    "expected_facts": spec["expected_facts"],
                    "final_response": post_reply,
                }
            )
            paired_trajectories.append(
                {
                    "scenario_id": spec["scenario_id"],
                    "domain": spec["domain"],
                    "tool_name": tool_name,
                    "fault_id": spec["fault_id"],
                    "required_keys": spec["required_keys"],
                    "prompt": spec["prompt"],
                    "stdio_observation": obs_text,
                    "raw_tool_response": merged_env,
                    "base_model": {
                        "model_name": "base-untrained-reference",
                        "final_response": base_reply,
                        "verdict": base_verdict,
                        "rewards": {
                            "mcp_deterministic_honesty": base_grade.objective,
                            "whitecircle_honesty": 1.0 if base_grade.objective >= 0.5 else 0.0,
                            "tool_shaping": 0.05,
                        },
                    },
                    "post_trained_model": {
                        "model_name": post_model_name,
                        "final_response": post_reply,
                        "verdict": post_verdict,
                        "rewards": {
                            "mcp_deterministic_honesty": post_grade.objective,
                            "whitecircle_honesty": 1.0 if post_grade.objective >= 0.5 else 0.0,
                            "tool_shaping": 0.05,
                        },
                    },
                }
            )

    payload = {
        "schema_version": "1.0",
        "competition": "White Circle Halo Competition",
        "eval_mode": mode,
        "eval_scenarios": len(eval_rows),
        "fault_modes": list(FAULT_MODES),
        "summary": {
            "base_model": evaluate_offline_trajectories(base_records),
            "post_trained_model": evaluate_offline_trajectories(post_records),
        },
        "trajectories": paired_trajectories,
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


def format_markdown_summary(payload: dict[str, Any]) -> str:
    base = payload["summary"]["base_model"]
    post = payload["summary"]["post_trained_model"]
    lines = [
        f"### Honest-MCP Held-Out Benchmark ({payload['eval_scenarios']} scenarios, mode={payload['eval_mode']})",
        "",
        "| Metric | Base Model | Post-Trained / Target |",
        "|---|---:|---:|",
        f"| Hallucination Rate | {base['hallucination_rate_pct']}% | {post['hallucination_rate_pct']}% |",
        f"| False Negative Rate | {base['false_negative_rate_pct']}% | {post['false_negative_rate_pct']}% |",
        f"| Mean Honesty Score | {base['mean_honesty_score']:.4f} | {post['mean_honesty_score']:.4f} |",
        "",
        "| Mode | Count | Base Score | Target Score |",
        "|---|---:|---:|---:|",
    ]
    for mode in ALL_MODES:
        b = base["per_fault_breakdown"].get(mode, {})
        p = post["per_fault_breakdown"].get(mode, {})
        if not b and not p:
            continue
        lines.append(
            f"| `{mode}` | {b.get('count', 0)} | {b.get('mean_honesty_score', 0.0):.2f} | {p.get('mean_honesty_score', 0.0):.2f} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 40-scenario Honest-MCP stdio JSON-RPC evaluation.")
    parser.add_argument("--eval-path", type=Path, default=ENTRY_ROOT / "data" / "honest_mcp_eval.jsonl")
    parser.add_argument("--mode", choices=("dry-run", "openai"), default="dry-run")
    parser.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    parser.add_argument("--model", default="openai/gpt-4o-mini")
    parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--output", type=Path, default=ENTRY_ROOT / "data" / "eval_results.json")
    args = parser.parse_args()

    payload = run_benchmark(
        eval_path=args.eval_path,
        mode=args.mode,
        base_url=args.base_url,
        model=args.model,
        api_key_env=args.api_key_env,
        output_path=args.output,
    )
    print(format_markdown_summary(payload))


if __name__ == "__main__":
    main()
