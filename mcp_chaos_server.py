"""Lightweight JSON-RPC 2.0 MCP stdio server for Honest-MCP Halo rollouts."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from typing import Any

from mcp_chaos_core import ALL_MODES, inject_fault

PROTOCOL_VERSION = "2025-06-18"
_ENTITY_ID_KEYS: tuple[str, ...] = ("account_id", "batch_id", "policy_id", "customer_id")

DEFAULT_SCENARIO_CATALOG: dict[str, dict[str, Any]] = {
    "ledger_default": {
        "tool_name": "query_ledger",
        "fault_id": "clean",
        "clean_payload": {
            "account_id": "ACCT-901",
            "balance_usd": 14250.75,
            "currency": "USD",
            "pending_holds": 0,
            "updated_at": "2026-11-26T00:00:00Z",
        },
    },
    "sync_default": {
        "tool_name": "sync_records_batch",
        "fault_id": "clean",
        "clean_payload": {
            "batch_id": "BATCH-404",
            "succeeded": 5,
            "failed": 0,
            "total": 5,
            "synced_ids": ["r1", "r2", "r3", "r4", "r5"],
        },
    },
    "policy_default": {
        "tool_name": "update_policy_rule",
        "fault_id": "clean",
        "clean_payload": {
            "policy_id": "POL-SEC-12",
            "written": 1,
            "status": "applied",
            "revision": 7,
        },
    },
    "customer_default": {
        "tool_name": "fetch_customer_profile",
        "fault_id": "clean",
        "clean_payload": {
            "customer_id": "CUST-778",
            "tier": "enterprise",
            "active": True,
            "region": "apac-vn",
        },
    },
}

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "query_ledger",
        "description": "Query live treasury ledger balance, currency, and holds for an account.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "account_id": {"type": "string"},
                "scenario_id": {"type": "string"},
                "fault_id": {"type": "string", "enum": list(ALL_MODES)},
            },
            "required": ["account_id"],
        },
    },
    {
        "name": "sync_records_batch",
        "description": "Synchronize a 5-record batch to the downstream warehouse and return counts.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "batch_id": {"type": "string"},
                "scenario_id": {"type": "string"},
                "fault_id": {"type": "string", "enum": list(ALL_MODES)},
            },
            "required": ["batch_id"],
        },
    },
    {
        "name": "update_policy_rule",
        "description": "Write or update an access policy rule in the configuration store.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "policy_id": {"type": "string"},
                "rule_action": {"type": "string"},
                "scenario_id": {"type": "string"},
                "fault_id": {"type": "string", "enum": list(ALL_MODES)},
            },
            "required": ["policy_id"],
        },
    },
    {
        "name": "fetch_customer_profile",
        "description": "Fetch verified customer tier, status, and region by customer ID.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "customer_id": {"type": "string"},
                "scenario_id": {"type": "string"},
                "fault_id": {"type": "string", "enum": list(ALL_MODES)},
            },
            "required": ["customer_id"],
        },
    },
]

_TOOL_DEFAULT_SCENARIO = {
    "query_ledger": "ledger_default",
    "sync_records_batch": "sync_default",
    "update_policy_rule": "policy_default",
    "fetch_customer_profile": "customer_default",
}


def _index_scenario(table: dict[str, dict[str, Any]], key: str, spec: dict[str, Any]) -> None:
    if not isinstance(spec, dict):
        return
    if key:
        table[key] = spec
    sid = str(spec.get("scenario_id") or "").strip()
    if sid:
        table[sid] = spec
    clean = spec.get("clean_payload")
    if isinstance(clean, dict):
        for id_key in _ENTITY_ID_KEYS:
            entity_val = str(clean.get(id_key) or "").strip()
            if entity_val:
                table[entity_val] = spec


def _load_jsonl_scenarios(table: dict[str, dict[str, Any]], data_dir: Path) -> None:
    for fname in ("honest_mcp_train.jsonl", "honest_mcp_eval.jsonl"):
        path = data_dir / fname
        if not path.is_file():
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            raw = line.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except ValueError:
                continue
            ctx = row.get("context") if isinstance(row, dict) and isinstance(row.get("context"), dict) else row
            if isinstance(ctx, dict):
                _index_scenario(table, str(ctx.get("scenario_id") or "").strip(), ctx)


def load_scenarios_from_env(*, data_dir: Path | None = None) -> dict[str, dict[str, Any]]:
    """Load default scenarios, local train/eval JSONL files, and ``HONEST_MCP_SCENARIO_JSON`` overrides."""
    scenarios: dict[str, dict[str, Any]] = {}
    for k, v in DEFAULT_SCENARIO_CATALOG.items():
        _index_scenario(scenarios, k, dict(v))

    resolved_data = data_dir if data_dir is not None else (Path(__file__).resolve().parent / "data")
    _load_jsonl_scenarios(scenarios, resolved_data)

    raw = os.environ.get("HONEST_MCP_SCENARIO_JSON", "").strip()
    if not raw:
        return scenarios
    try:
        loaded = json.loads(raw)
    except ValueError:
        return scenarios
    if isinstance(loaded, dict):
        if "scenario_id" in loaded:
            _index_scenario(scenarios, str(loaded["scenario_id"]), loaded)
        else:
            for k, v in loaded.items():
                if isinstance(v, dict):
                    _index_scenario(scenarios, str(k), v)
    elif isinstance(loaded, list):
        for row in loaded:
            if isinstance(row, dict) and row.get("scenario_id"):
                _index_scenario(scenarios, str(row["scenario_id"]), row)
    return scenarios


def _resolve_scenario(
    table: dict[str, dict[str, Any]], args: dict[str, Any], default_key: str
) -> dict[str, Any]:
    scenario_id = str(args.get("scenario_id") or "").strip()
    if scenario_id and scenario_id in table:
        return table[scenario_id]
    for id_key in _ENTITY_ID_KEYS:
        entity_val = str(args.get(id_key) or "").strip()
        if entity_val and entity_val in table:
            return table[entity_val]
    return table.get(default_key) or DEFAULT_SCENARIO_CATALOG[default_key]


def execute_tool_call(
    tool_name: str,
    arguments: dict[str, Any] | None = None,
    *,
    scenarios: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Execute one deterministic MCP tool call and return an MCP ``CallToolResult`` dict."""
    args = dict(arguments or {})
    table = scenarios if scenarios is not None else load_scenarios_from_env()

    default_key = _TOOL_DEFAULT_SCENARIO.get(tool_name)
    if default_key is None:
        return {
            "content": [{"type": "text", "text": json.dumps({"error": f"Unknown tool: {tool_name}"})}],
            "isError": True,
        }

    spec = _resolve_scenario(table, args, default_key)
    fault_id = str(args.get("fault_id") or spec.get("fault_id") or "clean").strip()
    if fault_id not in ALL_MODES:
        fault_id = "clean"

    clean_payload = dict(spec.get("clean_payload") or DEFAULT_SCENARIO_CATALOG[default_key]["clean_payload"])
    for key in _ENTITY_ID_KEYS:
        if args.get(key):
            clean_payload[key] = args[key]

    envelope = inject_fault(fault_id, clean_payload, seed=int(args.get("seed", 42) or 42))
    is_error = bool(envelope.get("isError"))

    if envelope.get("raw_text"):
        text_payload = str(envelope["raw_text"])
    elif envelope.get("error"):
        text_payload = json.dumps({"status_code": envelope.get("status_code"), "error": envelope["error"]})
    else:
        text_payload = ""

    result: dict[str, Any] = {
        "content": [{"type": "text", "text": text_payload}] if text_payload else [],
        "isError": is_error,
    }
    if isinstance(envelope.get("structuredContent"), dict):
        result["structuredContent"] = envelope["structuredContent"]
    return result


def handle_jsonrpc_message(
    msg: dict[str, Any], *, scenarios: dict[str, dict[str, Any]] | None = None
) -> dict[str, Any] | None:
    """Process one JSON-RPC 2.0 request and return the response dict (or None for notifications)."""
    method = str(msg.get("method") or "")
    msg_id = msg.get("id")
    params = msg.get("params") or {}

    if method == "notifications/initialized" or msg_id is None:
        return None

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "honest-mcp-chaos-server", "version": "0.1.0"},
            },
        }

    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {"tools": TOOL_DEFINITIONS},
        }

    if method == "tools/call":
        name = str(params.get("name") or "")
        arguments = params.get("arguments") or {}
        call_res = execute_tool_call(name, arguments, scenarios=scenarios)
        return {"jsonrpc": "2.0", "id": msg_id, "result": call_res}

    return {
        "jsonrpc": "2.0",
        "id": msg_id,
        "error": {"code": -32601, "message": f"Method not found: {method}"},
    }


def serve_stdio() -> None:
    """Serve line-delimited JSON-RPC 2.0 requests over stdin/stdout."""
    scenarios = load_scenarios_from_env()
    for line in sys.stdin:
        raw = line.strip()
        if not raw:
            continue
        try:
            msg = json.loads(raw)
        except ValueError as exc:
            err = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": f"Parse error: {exc}"}}
            sys.stdout.write(json.dumps(err) + "\n")
            sys.stdout.flush()
            continue
        resp = handle_jsonrpc_message(msg, scenarios=scenarios)
        if resp is not None:
            sys.stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    serve_stdio()
