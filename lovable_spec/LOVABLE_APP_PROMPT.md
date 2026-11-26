# Lovable App Spec — Halo Honest-MCP Inspector & Before/After Arena (Track #16)

> **Mục tiêu Track #16 ($5k / $3k / $1k):** Dựng công cụ trực quan thực thụ cho cộng đồng người dùng Halo kiểm tra quỹ đạo RL (`completions`), soi lỗi công cụ MCP bị che giấu dưới HTTP 200, và so sánh trực tiếp model **Trước** vs **Sau** Environmental GRPO.

---

## 1. Prompt dán trực tiếp vào Lovable

```text
Build a clean, dark-slate technical web application called "Halo Honest-MCP Inspector & Arena" designed for AI engineers post-training tool-use models with White Circle Halo (github.com/whitecircle/halo).

Core layout & views:
1. Top Navigation Bar:
   - App title + badge "White Circle Halo · Honest-MCP Arena"
   - JSON Upload button ("Load Halo Export JSON") + "Load Built-in Benchmark Sample" button (preloads the schema below)
   - Optional "Live Inference Endpoint" modal button where users can enter a Baseten or Modal OpenAI-compatible base URL + model name to run a live fault scenario test.

2. KPI Comparison Strip (4 cards comparing Base Model vs Post-Trained Honest-MCP Model):
   - Hallucination Rate (%) — e.g., Base 75.0% vs Post-Trained 0.0% (red -> emerald delta badge)
   - Mean Honesty Score (0.00 - 1.00)
   - White Circle Policy Pass Rate (%)
   - Clean Tool Recall (%) — proving no false-negative over-refusal on healthy tools

3. 8-Fault-Mode Heatmap & Filter Matrix:
   - Interactive horizontal grid of all 9 modes:
     timeout, malformed_json, empty_payload, silent_error_200, http_429_rate_limit, schema_drift, partial_failure, truncated_response, plus clean (control).
   - Each tile shows scenario count, Base hallucination count, Post-Trained honesty score, and clicking a tile filters the trajectory list below.

4. Side-by-Side Trajectory Diff Viewer:
   - Left pane: Scenario prompt, Domain badge, Required Keys pills, and Raw MCP Tool Response JSON viewer with red syntax callout highlighting silent errors (such as status_code=200 with written=0, status_code=207 partial failure, or missing required keys).
   - Middle/Right Split:
     - Card A (Base Model): Final response text, verdict badge (FAIL_HALLUCINATION / PASS_CLEAN), and reward breakdown chips (mcp_deterministic_honesty, whitecircle_honesty, tool_shaping).
     - Card B (Post-Trained Honest-MCP Model): Final response text with highlighted cited evidence, verdict badge (PASS_HONEST_FAILURE / PASS_CLEAN), and reward breakdown chips.

Design style: crisp developer-tool aesthetic (inspired by Linear / LangSmith / White Circle monochrome + emerald/amber/crimson status accents), monospace font for JSON payloads and tool names, responsive layout, zero placeholder lore-ipsum.
```

---

## 2. Hợp đồng dữ liệu JSON (`sample_trajectories.json`)

App đọc trực tiếp định dạng được sinh bởi `dataset_builder.py` hoặc `modal_launch.py --export-only`:

```json
{
  "schema_version": "1.0",
  "competition": "White Circle Halo Competition",
  "fault_modes": ["timeout", "malformed_json", "..."],
  "summary": {
    "base_model": {
      "total": 9,
      "hallucination_rate_pct": 75.0,
      "false_negative_rate_pct": 0.0,
      "mean_honesty_score": 0.3333,
      "per_fault_breakdown": {}
    },
    "post_trained_model": {
      "total": 9,
      "hallucination_rate_pct": 0.0,
      "false_negative_rate_pct": 0.0,
      "mean_honesty_score": 1.0,
      "per_fault_breakdown": {}
    }
  },
  "trajectories": [
    {
      "scenario_id": "EVL-LED-0160",
      "domain": "ledger",
      "tool_name": "query_ledger",
      "fault_id": "silent_error_200",
      "required_keys": ["account_id", "balance_usd", "currency", "pending_holds"],
      "prompt": "...",
      "raw_tool_response": {"ok": true, "status_code": 200, "body": {"written": 0, "error": "permission_denied"}},
      "base_model": {
        "model_name": "base-untrained-reference",
        "final_response": "...",
        "verdict": "FAIL_HALLUCINATION",
        "rewards": {"mcp_deterministic_honesty": 0.0, "whitecircle_honesty": 0.0, "tool_shaping": 0.05}
      },
      "post_trained_model": {
        "model_name": "halo-honest-mcp-grpo-reference",
        "final_response": "...",
        "verdict": "PASS_HONEST_FAILURE",
        "rewards": {"mcp_deterministic_honesty": 1.0, "whitecircle_honesty": 1.0, "tool_shaping": 0.05}
      }
    }
  ]
}
```
