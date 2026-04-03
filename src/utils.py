import time
from functools import partial

import requests
from sympy import Integer as Int

from src.http_utils import request_with_retry

AUTH_URL = "https://api.worldquantbrain.com/authentication"
DEFAULT_AUTH_TIMEOUT = 30
DEFAULT_AUTH_RETRIES = 6


def authenticate_session(
    session: requests.Session,
    logger=None,
    *,
    context: str = "Authentication",
    max_retries: int = DEFAULT_AUTH_RETRIES,
    timeout: int = DEFAULT_AUTH_TIMEOUT,
):
    """Authenticate a WorldQuant session with rate-limit aware retries."""
    return request_with_retry(
        lambda: session.post(AUTH_URL, timeout=timeout),
        logger=logger,
        context=context,
        max_retries=max_retries,
        sleep_fn=time.sleep,
        base_delay=2.0,
        max_delay=60.0,
        jitter_ratio=0.15,
        quota_cooldown_threshold=2,
        quota_cooldown_seconds=60.0,
    )


def create_authenticated_session(username: str, password: str, logger=None, *, context: str = "Authentication"):
    """Create and authenticate a requests session."""
    session = requests.Session()
    session.auth = (username, password)
    response = authenticate_session(session, logger=logger, context=context)
    if response is not None and response.status_code == 201:
        return session, response
    return None, response

def evaluate_fitness(s: requests.Session, logger=None):
    from src.brain import simulate

    metric = partial(simulate, s, logger=logger)
    metric.requires_session = True
    return metric


def help_check_same(x: Int, y: Int) -> bool:
    if x == y:
        return x
    else:
        raise ValueError('This binary operator requires the same unit for both inputs')
