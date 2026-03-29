import time
import random
import logging
import anthropic

# --- Client ---

client = anthropic.Anthropic()


# --- Configure logging ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)

logger = logging.getLogger("boss_challenge")

# --- Defensine caps ---
MAX_RETRIES = 5
MAX_BACKOFF_SECONDS = 60

# --- Pricing table ---
PRICING = {
    "claude-haiku-4-5-20251001": {"input": 1.00, "output": 5.00},
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
    "claude-opus-4-6": {"input": 5.00, "output": 25.00},
}

# --- Errors that should be retried ---
RETRYABLE_ERRORS = (
    anthropic.RateLimitError,  # 429
    anthropic.InternalServerError,  # 500
    anthropic.APIConnectionError,  # network failures (include timeout)
)

# -- Model Tires: map task tiers to model aliases

MODEL_TRIES = {
    "fast": "claude-haiku-4-5-20251001",
    "balanced": "claude-sonnet-4-6",
    "power": "claude-opus-4-6",
}

# --- Fallback chain: if preferred model in unavailable, try next ---
FALLBACK_ORDER = ["power", "balanced", "fast"]


def _verify_model_availabel(model_id: str) -> bool:
    """
    Check if a model is available for our API key.
    Returns True if available, False ohterwise.
    """
    try:
        client.models.retrieve(model_id)
        return True
    except anthropic.NotFoundError:
        # Model doesn't exist or we don't have access
        return False
    except anthropic.APIError as e:
        # Any other API-level error (rate limit, serve issue)
        error_type = type(e).__name__
        status = getattr(e, "status_code", "N/A")
        logger.warning(
            f"API error checking model '{model_id}' | {error_type} (status={status})"
        )
        return False


# --- Calculate Backoff ---


def _calculate_backoff(attempt: int) -> float:
    """
    Exponential backoff with jitter.
    attempt 0 → ~1s, attempt 1 → ~2s, attempt 2 → ~4s, etc.
    Jitter prevents thundering herd when multiple clients retry simultaneously.
    """

    base = min(2**attempt, MAX_BACKOFF_SECONDS)
    jitter = random.uniform(0, base * 0.5)  # up to 50%
    return base * jitter


# --- Estimate Cost  Model ---


def _estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate cost in USD. Returns -1.0 if model not in pricing table."""
    pricing = PRICING.get(model)

    if not pricing:
        return -1.0
    return round(
        (input_tokens / 1_000_000) * pricing["input"]
        + (output_tokens / 1_000_000) * pricing["output"],
        6,
    )


def _select_model(task_tier: str = "balanced") -> str:
    """
        Select the best available model for the given task tier.
        Falls back through the tier list if the proferred model is unavailable.

        Args:
            task_tier: One of 'fast', 'balanced', 'power'

        Returns:
        A verified model ID string ready for messages.create()

    Raises:
        RuntimeError if no models are available at all
    """

    try:
        preferred = MODEL_TRIES.get(task_tier)
        if preferred and _verify_model_availabel(preferred):
            logger.info(f"Success : Using preferred model: {preferred}")
            return preferred
        logger.info(
            f"Info: Preferred tier '{task_tier}' unavailable, trying fallbacks ..."
        )
        for tier_name in FALLBACK_ORDER:
            fallback_model = MODEL_TRIES[tier_name]
            if fallback_model != preferred and _verify_model_availabel(fallback_model):
                logger.info(f"Success: Falling back to {fallback_model}")
                return fallback_model
    except RuntimeError as e:
        error_type = type(e).__name__
        status = getattr(e, "status_code", "N/A")
        logger.error(
            f"Error getting model {preferred} | {error_type} | (status={status})"
        )


# --- Claude Model Call ---


def claude_call(
    prompt: str,
    task_tier: str = "balanced",
    # model: str = 'claude-sonnet-4-6', # no need the model because it will be pass when _select_model() be call
    max_tokens: int = 1024,
    system: str | None = None,
    max_retries: int = MAX_RETRIES,
    timeout: float = 120.0,
    **kwargs,  # pass-through for thinking, tool, etc.
) -> anthropic.types.Message:
    """
    Production-safe wrapper for client.messages.create().

    Args:
        messages:    List of message dicts [{"role": ..., "content": ...}]
        model:       Model ID string
        max_tokens:  Max output tokens
        system:      Optional system prompt (top-level param)
        max_retries: Max retry attempts for transient errors
        timeout:     Request timeout in seconds
        task_tier:   One of 'fast', 'balanced', 'power'
        **kwargs:    Additional params (thinking, tools, tool_choice, etc.)

    Returns:
        anthropic.types.Message on success

    Raises:
        anthropic.AuthenticationError, anthropic.BadRequestError:
            Non-retryable errors are raised immediately
        RuntimeError: if all retries exhausted
    """

    # --- Create client with no build-in retries (we handle it ourselves) ---
    client = anthropic.Anthropic(max_retries=0, timeout=timeout)

    model = _select_model(task_tier)

    params = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
        **kwargs,  # spread any extra params (thinking, tools, etc.)
    }

    if system:
        params["system"] = system

    last_error = None

    # --- Retry loop with defensive cap ---

    for attempt in range(max_retries):
        try:
            logger.info(
                f"Attempt {attempt + 1} / {max_retries} | "
                f"model={model} | max_tokens={max_tokens}"
            )

            response = client.messages.create(**params)
            stop_reason = None
            match response.stop_reason:
                case "end_turn":
                    # Normal completion
                    stop_reason = "Complete response received"
                case "max_tokens":
                    # Response was truncated — you need to either:
                    # 1. Increase max_tokens
                    # 2. Continue the conversation to get the rest
                    stop_reason = (
                        "⚠️  Response TRUNCATED — increase max_tokens or continue."
                    )
                case "tool_use":
                    # Claude wants to use a tool — enter your tool dispatch loop
                    stop_reason = "Claude requested a tool call."
                case "stop_sequence":
                    stop_reason = "Hit a stop squence."
                case _:
                    stop_reason = f"Unexpected stop_reason : {response.stop_reason}"

            # --- Success: log usage and cost ---
            usage = response.usage
            cost = _estimate_cost(model, usage.input_tokens, usage.output_tokens)
            logger.info(
                f"Success | request_id={response._request_id} | "
                f"in={usage.input_tokens} out={usage.output_tokens} | "
                f"cost=${cost:.6f} | stop_reason={stop_reason}"
            )
            return response
        except RETRYABLE_ERRORS as e:
            # --- transient error: log, backoff, retry ---
            last_error = e
            wait = _calculate_backoff(attempt)

            # Check for rate limit headers id available
            retry_after = None
            if hasattr(e, "response") and e.response and e.response.headers:
                retry_after = e.response.headers.get("retry-after")

            # if API says "retry afetr X seconds", respect it

            if retry_after:
                try:
                    wait = max(wait, float(retry_after))
                except ValueError:
                    pass  # header wasn't a number, user our calculated backoff

            error_type = type(e).__name__
            status = getattr(e, "status_code", "N/A")
            logger.warning(
                f"Retryable error | {error_type} (status={status}) | "
                f"attempt {attempt + 1} / {max_retries} | "
                f"waiting {wait:.1f}s"
            )

            if attempt < max_retries - 1:
                time.sleep(wait)
            # else: fall through to raise after loop

        except anthropic.APIStatusError as e:
            # --- Non-retryable API error: raise immediately ---
            # This catches 400, 401, 403, 404, and any other status error
            # that isn't 429/500/529
            logger.error(
                f"Non-retryable error | {type(e).__name__} "
                f"(status={e.status_code}) | {e.message}"
            )
            raise  # let the caller handle it

    # --- All retries exhausted ---
    logger.error(f"All {max_retries} attempts failed. Last error: {last_error}")
    raise RuntimeError(
        f"Claude API call failed after {max_retries} attempts. "
        f"Last error: {last_error}"
    )


# =================================================================================
#   DEMO: Using claude_call() in production
# =================================================================================

if __name__ == "__main__":

    tasks = [
        ("fast", " Classify this support ticket: 'My order hasn't arrived'"),
        (
            "balanced",
            "Write a Python function to parse nested JSON with error handling.",
        ),
        (
            "power",
            "Analyze this codebase architecture and suggest refactoring strategy.",
        ),
    ]

    print("=== Example 1: Simple Call ===\n")

    for tier, prompt in tasks:
        print(f"\n--- Task tier: {tier} ---")
        try:
            result = claude_call(prompt=prompt, task_tier=tier, max_tokens=200)
            print(f"Answer: {result.content[0].text[:200]}...\n")
        except Exception as e:
            print(f"Failed: {e}\n")

    print("=== Example 2: With system prompt and extended thinking ===\n")

    for tier, prompt in tasks:
        print(f"\n--- Task tier: {tier}")
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
        except Exception as e:
            print(f"Failed: {e}")

    print("\n=== Example 3: Bad Auth (should fail immediately) ===\n")
    bad_client = anthropic.Anthropic(api_key="sk-ant-fake-key", max_retries=0)
    try:
        bad_client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=100,
            messages=[
                {
                    "role": "user",
                    "content": "test",
                }
            ],
        )
    except anthropic.AuthenticationError as e:
        print(f"Caught AuthenticationError (no retry): {e.message}")
    except Exception as e:
        print(f"Caught: {type(e).__name__}: {e}")
