import random
import time

import requests

RETRYABLE_HTTP_STATUSES = {429, 500, 502, 503, 504}


def log_http_message(logger, level: str, message: str) -> None:
    if logger is None:
        print(message)
        return
    getattr(logger, level)(message)


def parse_retry_after(raw_value, *, minimum: float = 0.0) -> float | None:
    if raw_value is None:
        return None
    try:
        parsed = float(raw_value)
    except (TypeError, ValueError):
        return None
    return max(float(minimum), parsed)


def is_retryable_http_status(status_code: int | None) -> bool:
    return status_code in RETRYABLE_HTTP_STATUSES


def compute_backoff_delay(
    attempt: int,
    *,
    retry_after: float | None = None,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter_ratio: float = 0.2,
) -> float:
    if retry_after is not None:
        return max(0.0, float(retry_after))

    capped_attempt = max(1, int(attempt))
    delay = min(float(max_delay), float(base_delay) * (2 ** (capped_attempt - 1)))
    jitter = 0.0
    if jitter_ratio > 0:
        jitter = random.uniform(0.0, delay * float(jitter_ratio))
    return max(0.0, delay + jitter)


def request_with_retry(
    send_request,
    *,
    logger=None,
    context: str = "HTTP request",
    max_retries: int = 4,
    sleep_fn=time.sleep,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter_ratio: float = 0.2,
    quota_cooldown_threshold: int = 3,
    quota_cooldown_seconds: float = 30.0,
):
    response = None
    consecutive_quota_errors = 0

    for attempt in range(1, max(1, int(max_retries)) + 1):
        try:
            response = send_request()
        except requests.exceptions.Timeout as exc:
            if attempt >= max_retries:
                log_http_message(
                    logger,
                    "error",
                    f"{context} timed out after {attempt} attempts: {exc}",
                )
                return None

            wait_seconds = compute_backoff_delay(
                attempt,
                base_delay=base_delay,
                max_delay=max_delay,
                jitter_ratio=jitter_ratio,
            )
            log_http_message(
                logger,
                "warning",
                f"{context} timed out on attempt {attempt}/{max_retries}: {exc}. "
                f"Retrying in {wait_seconds:.1f}s.",
            )
            sleep_fn(wait_seconds)
            continue
        except requests.exceptions.RequestException as exc:
            if attempt >= max_retries:
                log_http_message(
                    logger,
                    "error",
                    f"{context} failed after {attempt} attempts due to a network error: {exc}",
                )
                return None

            wait_seconds = compute_backoff_delay(
                attempt,
                base_delay=base_delay,
                max_delay=max_delay,
                jitter_ratio=jitter_ratio,
            )
            log_http_message(
                logger,
                "warning",
                f"{context} network error on attempt {attempt}/{max_retries}: {exc}. "
                f"Retrying in {wait_seconds:.1f}s.",
            )
            sleep_fn(wait_seconds)
            continue

        status_code = getattr(response, "status_code", None)
        if not is_retryable_http_status(status_code):
            return response

        retry_after = parse_retry_after(
            getattr(response, "headers", {}).get("Retry-After"),
            minimum=1.0 if status_code == 429 else 0.0,
        )
        wait_seconds = compute_backoff_delay(
            attempt,
            retry_after=retry_after,
            base_delay=base_delay,
            max_delay=max_delay,
            jitter_ratio=jitter_ratio,
        )

        if status_code == 429:
            consecutive_quota_errors += 1
            if consecutive_quota_errors >= quota_cooldown_threshold:
                wait_seconds = max(wait_seconds, float(quota_cooldown_seconds))
                message = (
                    f"{context} hit quota repeatedly on attempt {attempt}/{max_retries}. "
                    f"Cooling down for {wait_seconds:.1f}s before retrying."
                )
            else:
                message = (
                    f"{context} rate-limited on attempt {attempt}/{max_retries}. "
                    f"Waiting {wait_seconds:.1f}s before retrying."
                )
        else:
            consecutive_quota_errors = 0
            message = (
                f"{context} server error {status_code} on attempt {attempt}/{max_retries}. "
                f"Retrying in {wait_seconds:.1f}s."
            )

        if attempt >= max_retries:
            log_http_message(logger, "error", f"{context} failed after {attempt} attempts (status={status_code}).")
            return response

        log_http_message(logger, "warning", message)
        sleep_fn(wait_seconds)

    return response
