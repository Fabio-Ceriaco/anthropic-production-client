# anthropic-production-client

A production-grade Python wrapper for the [Anthropic Messages API](https://docs.anthropic.com/en/api/messages) with tiered model selection, automatic retries, cost tracking, and structured logging.

Built as part of an AI automation portfolio — designed for real-world use in support systems, internal tooling, and agentic workflows.

## Features

- **Tiered model routing** — map task complexity (`fast` / `balanced` / `power`) to the optimal Claude model, with automatic fallback if a model is unavailable
- **Exponential backoff with jitter** — retries transient errors (429, 500, network failures) while respecting `Retry-After` headers
- **Fail-fast on non-retryable errors** — authentication and bad request errors raise immediately
- **Per-call cost estimation** — logs estimated USD cost based on input/output token counts
- **Stop-reason inspection** — flags truncated responses, tool-use requests, and unexpected stop reasons
- **Pass-through kwargs** — supports extended thinking, tool use, and any future API parameters

## Quick start

```bash
# Clone and install
git clone https://github.com/Fabio-Ceriaco/anthropic-production-client.git
cd anthropic-production-client
pip install -r requirements.txt

# Set your API key
export ANTHROPIC_API_KEY="sk-ant-..."

# Run the demo
python examples/demo.py
```

## Usage

```python
from claude_client import claude_call

# Simple call — routes to Haiku (fast tier)
response = claude_call(
    prompt="Classify this ticket: 'My order hasn't arrived'",
    task_tier="fast",
    max_tokens=200,
)
print(response.content[0].text)

# With system prompt and extended thinking — routes to Opus (power tier)
response = claude_call(
    prompt="Analyze this architecture and suggest refactoring.",
    task_tier="power",
    max_tokens=8000,
    system="You are a senior Python developer.",
    thinking={"type": "adaptive"},
)

for block in response.content:
    if block.type == "thinking":
        print(f"[THINKING] {block.thinking[:200]}")
    elif block.type == "text":
        print(f"[RESPONSE] {block.text}")
```

## Model tiers

| Tier       | Model             | Best for                                           |
| ---------- | ----------------- | -------------------------------------------------- |
| `fast`     | claude-haiku-4-5  | Classification, routing, simple extraction         |
| `balanced` | claude-sonnet-4-6 | Code generation, analysis, multi-step tasks        |
| `power`    | claude-opus-4-6   | Complex reasoning, architecture, extended thinking |

If the preferred model is unavailable, the wrapper automatically falls back through the tier list.

## Configuration

Key constants in `claude_client.py`:

| Constant              | Default        | Description                                               |
| --------------------- | -------------- | --------------------------------------------------------- |
| `MAX_RETRIES`         | 5              | Maximum retry attempts for transient errors               |
| `MAX_BACKOFF_SECONDS` | 60             | Backoff cap in seconds                                    |
| `PRICING`             | Per-model dict | USD per 1M tokens — update when Anthropic changes pricing |

## Project structure

```
anthropic-production-client/
├── claude_client.py        # Main wrapper module
├── examples/
│   └── demo.py             # Usage examples (tiered calls, thinking, auth test)
├── requirements.txt
├── .gitignore
└── README.md
```

## Requirements

- Python 3.12+
- `anthropic` SDK ≥ 0.45.0
- `ANTHROPIC_API_KEY` environment variable

## License

No license specified.

## Author

Fábio Ceriaço — Technical Support Engineer & AI Automation Specialist
