"""
Example usage of the anthropic-production-client wrapper.

Run:
    python examples/demo.py

Requires:
    ANTHROPIC_API_KEY set as environment variable.
"""

import anthropic
from client.production_client import claude_call


# ---------------------------------------------------------------------------
# Example 1 — Tiered model routing
# ---------------------------------------------------------------------------
def example_simple_calls():
    """Each task routes to the optimal model for its complexity."""

    tasks = [
        ("fast", "Classify this support ticket: 'My order hasn't arrived'"),
        (
            "balanced",
            "Write a Python function to parse nested JSON with error handling.",
        ),
        (
            "power",
            "Analyze this codebase architecture and suggest refactoring strategy.",
        ),
    ]

    print("=" * 70)
    print("  Example 1: Simple calls — tiered model routing")
    print("=" * 70)

    for tier, prompt in tasks:
        print(f"\n--- Task tier: {tier} ---")
        try:
            result = claude_call(prompt=prompt, task_tier=tier, max_tokens=200)
            print(f"Answer: {result.content[0].text[:300]}...\n")
        except Exception as exc:
            print(f"Failed: {exc}\n")


# ---------------------------------------------------------------------------
# Example 2 — System prompt + extended thinking
# ---------------------------------------------------------------------------
def example_with_thinking():
    """Demonstrates system prompts and adaptive thinking."""

    tasks = [
        ("fast", "Classify this support ticket: 'My order hasn't arrived'"),
        (
            "balanced",
            "Write a Python function to parse nested JSON with error handling.",
        ),
        (
            "power",
            "Analyze this codebase architecture and suggest refactoring strategy.",
        ),
    ]

    print("\n" + "=" * 70)
    print("  Example 2: System prompt + extended thinking")
    print("=" * 70)

    for tier, prompt in tasks:
        print(f"\n--- Task tier: {tier} ---")
        try:
            result = claude_call(
                prompt=prompt,
                task_tier=tier,
                max_tokens=8000,
                system="You are a senior Python developer. Write production-grade code.",
                thinking={"type": "adaptive"},
            )

            for block in result.content:
                if block.type == "thinking":
                    print(f"    [THINKING]: {block.thinking[:200]}...")
                elif block.type == "text":
                    print(f"    [CODE]: {block.text[:300]}...")
        except Exception as exc:
            print(f"Failed: {exc}\n")


# ---------------------------------------------------------------------------
# Example 3 — Bad auth (should fail immediately, no retries)
# ---------------------------------------------------------------------------
def example_bad_auth():
    """Non-retryable errors are raised instantly."""

    print("\n" + "=" * 70)
    print("  Example 3: Bad auth — immediate failure, no retry")
    print("=" * 70 + "\n")

    bad_client = anthropic.Anthropic(api_key="sk-ant-fake-key", max_retries=0)
    try:
        bad_client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=100,
            messages=[{"role": "user", "content": "test"}],
        )
    except anthropic.AuthenticationError as exc:
        print(f"Caught AuthenticationError (no retry): {exc.message}")
    except Exception as exc:
        print(f"Caught: {type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# Run all examples
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    example_simple_calls()
    example_with_thinking()
    example_bad_auth()
