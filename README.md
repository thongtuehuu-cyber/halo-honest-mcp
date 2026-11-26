# Honest-MCP Agent — White Circle Halo Competition Entry

Post-training open-weight tool-use models with **White Circle Halo** (`github.com/whitecircle/halo`) Async Environmental GRPO so agents stop hallucinating `"Task completed"` when Model Context Protocol (MCP) tools fail silently (`HTTP 200` policy rejections, `HTTP 207` partial batch failures, schema drift, empty bodies, or truncated JSON).

## Multi-Track Alignment

| Competition Track | How This Entry Qualifies | Entry Artifact |
|---|---|---|
| **Best use of White Circle** | Combines deterministic env grading (`0.6`) with `WhiteCircleTerm` (`0.4`, `view: full`) using a 5-item honesty checklist + 2 quality policies | `configs/whitecircle_honesty_policies.json`, `configs/honest_mcp_grpo_template.yaml` |
| **Post-Training (`Inkling` / `Nemotron`)** | Multi-turn `HonestMCPEnvironment` extending `NativeMCPClientEnvironment` over a deterministic JSON-RPC 2.0 stdio chaos server | `halo_honest_mcp_env.py`, `mcp_chaos_server.py`, `data/` |
| **Best use of Modal** | Serverless H100 launcher clearing the Halo container entrypoint (`Issue #223`), wiring `honest_mcp` into `halo/environments/registry.py`, and exporting Parquet completions | `modal_launch.py`, `install_into_halo.py` |
| **Best use of Lovable** | **Halo Honest-MCP Inspector & Before/After Arena** visualizing 8-fault-mode heatmaps, raw MCP fault callouts, and side-by-side diffs | `lovable_spec/arena_preview.html`, `lovable_spec/LOVABLE_APP_PROMPT.md` |
| **Best use of OpenRouter** | Dataset paraphrase enrichment (`--openrouter-enrich`) and live OpenAI-compatible multi-turn benchmark runner (`--mode openai`) | `dataset_builder.py`, `eval_benchmark.py` |

## Quickstart (Local CPU Verification)

```bash
# 1. Generate 160 train / 40 held-out eval scenarios + Lovable sample JSON
python dataset_builder.py

# 2. Run all 40 held-out scenarios end-to-end against real stdio mcp_chaos_server.py
python eval_benchmark.py --mode dry-run --output data/eval_results.json

# 3. Run unit & integration test suite
python -m pytest tests/test_honest_mcp_entry.py -v
```

Open `lovable_spec/arena_preview.html` in a browser and drop `data/eval_results.json` to inspect every trajectory visually.

## White Circle & Modal GPU Launch (Dec 1–10)

```bash
# 1. Sync honesty checklist & quality policies to your White Circle environment
WHITECIRCLE_API_KEY=wc-... halo run create-whitecircle-policies \
  --environment-id $WHITECIRCLE_ENV_ID \
  --policies configs/whitecircle_honesty_policies.json

# 2. Wire honest_mcp into Halo's registry (done automatically inside modal_launch.py)
python install_into_halo.py --halo-root /workspace

# 3. Launch Environmental GRPO on Modal H100
modal run modal_launch.py --sync-policies --dry-run
modal run modal_launch.py --config honest_mcp_entry/configs/honest_mcp_grpo_template.yaml
```

## Live Evaluation Against OpenRouter / Served Model

```bash
OPENROUTER_API_KEY=sk-or-... python eval_benchmark.py \
  --mode openai \
  --base-url https://openrouter.ai/api/v1 \
  --model <model-id> \
  --output data/eval_live.json
```

## Reproducibility & Disclosures

- Deterministic dataset generation & fault injection (`seed=42`, zero train/eval overlap in `scenario_id` or `prompt`).
- Based on the MIT-licensed 8-fault MCP verification taxonomy from `thongtuehuu-cyber/mcp-guard-kit`.
- AI assistance disclosure: developed and verified with Antigravity; every test, YAML key, and stdio JSON-RPC path was verified locally.
- License: **Apache-2.0** (`LICENSE`).
