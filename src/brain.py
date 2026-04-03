import requests
from time import monotonic, sleep
import pandas as pd
from datetime import datetime
import os
import re
import threading
from functools import wraps

from src.http_utils import compute_backoff_delay, is_retryable_http_status, parse_retry_after

RESULTS_CSV_PATH = "simulation_results.csv"
LEGACY_RESULTS_CSV_PATH = "simulations.csv"
ALPHA_HISTORY_URL = "https://api.worldquantbrain.com/users/self/alphas"
LEGACY_RESULT_COLUMNS = [
    "alpha_id",
    "regular_code",
    "turnover",
    "returns",
    "drawdown",
    "margin",
    "fitness",
    "sharpe",
    "LOW_SHARPE",
    "LOW_FITNESS",
    "LOW_TURNOVER",
    "HIGH_TURNOVER",
    "CONCENTRATED_WEIGHT",
    "LOW_SUB_UNIVERSE_SHARPE",
    "SELF_CORRELATION",
    "MATCHES_COMPETITION",
    "date",
]
CONTEXT_RESULT_COLUMNS = [
    "region",
    "universe",
    "delay",
    "decay",
    "neutralization",
    "truncation",
]
RESULT_COLUMNS = LEGACY_RESULT_COLUMNS + CONTEXT_RESULT_COLUMNS
CHECK_RESULT_COLUMNS = [
    "LOW_SHARPE",
    "LOW_FITNESS",
    "LOW_TURNOVER",
    "HIGH_TURNOVER",
    "CONCENTRATED_WEIGHT",
    "LOW_SUB_UNIVERSE_SHARPE",
    "SELF_CORRELATION",
    "MATCHES_COMPETITION",
]
DEFAULT_SIMULATION_SETTINGS_CONTEXT = {
    "region": "USA",
    "universe": "TOP3000",
    "delay": 1,
    "decay": 5,
    "neutralization": "MARKET",
    "truncation": 0.05,
}
WORLDQUANT_NEUTRALIZATION_LABELS = {
    "none": "NONE",
    "market": "MARKET",
    "sector": "SECTOR",
    "industry": "INDUSTRY",
}
WORLDQUANT_INVALID_NEUTRALIZATION_FALLBACK = "INDUSTRY"
WORLDQUANT_QUOTA_COOLDOWN_THRESHOLD = 3
WORLDQUANT_QUOTA_COOLDOWN_SECONDS = 60.0

# Global lock for thread-safe CSV operations
_csv_lock = threading.Lock()


def _log_message(logger, level: str, message: str):
    if logger:
        getattr(logger, level)(message)
    else:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}")


def _safe_json(response):
    try:
        return response.json()
    except Exception:
        return {}


def _normalize_expression(expr: str) -> str:
    return re.sub(r"\s+", "", expr or "")


def _coerce_setting(settings: dict, key: str, default=None):
    if not isinstance(settings, dict):
        return default
    value = settings.get(key, default)
    return default if value is None else value


def _merge_settings_context(settings: dict | None = None) -> dict:
    merged = dict(DEFAULT_SIMULATION_SETTINGS_CONTEXT)
    if isinstance(settings, dict):
        for key in CONTEXT_RESULT_COLUMNS:
            value = settings.get(key)
            if value is not None:
                merged[key] = value
    return merged


def _normalize_worldquant_neutralization(value) -> tuple[str, bool]:
    text = str(value or "").strip()
    key = re.sub(r"[\s_-]+", "", text).lower()
    if key in WORLDQUANT_NEUTRALIZATION_LABELS:
        return WORLDQUANT_NEUTRALIZATION_LABELS[key], False
    if key == "subindustry":
        return WORLDQUANT_INVALID_NEUTRALIZATION_FALLBACK, True
    if not key:
        return WORLDQUANT_NEUTRALIZATION_LABELS["market"], False
    return WORLDQUANT_INVALID_NEUTRALIZATION_FALLBACK, True


def _sanitize_worldquant_settings_context(settings: dict | None = None, *, logger=None) -> dict:
    merged = _merge_settings_context(settings)
    normalized_neutralization, changed = _normalize_worldquant_neutralization(merged.get("neutralization"))
    if changed:
        _log_message(
            logger,
            "warning",
            "WorldQuant rejected unsupported neutralization "
            f"{merged.get('neutralization')!r}; submitting with {normalized_neutralization!r} instead.",
        )
    merged["neutralization"] = normalized_neutralization
    return merged


def _session_rate_limit_state(session: requests.Session) -> dict:
    state = getattr(session, "_brain_rate_limit_state", None)
    if isinstance(state, dict):
        return state
    state = {"consecutive_quota_errors": 0, "cooldown_until": 0.0}
    setattr(session, "_brain_rate_limit_state", state)
    return state


def _wait_for_session_cooldown(session: requests.Session, logger=None) -> None:
    state = _session_rate_limit_state(session)
    cooldown_until = float(state.get("cooldown_until", 0.0) or 0.0)
    remaining = cooldown_until - monotonic()
    if remaining > 0:
        _log_message(logger, "warning", f"WorldQuant cooldown active. Waiting {remaining:.1f}s before the next request.")
        sleep(remaining)


def _reset_session_quota_state(session: requests.Session) -> None:
    state = _session_rate_limit_state(session)
    state["consecutive_quota_errors"] = 0
    state["cooldown_until"] = 0.0


def _schedule_session_quota_backoff(
    session: requests.Session,
    *,
    attempt: int,
    retry_after: float | None = None,
) -> tuple[float, int]:
    state = _session_rate_limit_state(session)
    state["consecutive_quota_errors"] = int(state.get("consecutive_quota_errors", 0) or 0) + 1
    wait_time = compute_backoff_delay(
        attempt,
        retry_after=retry_after,
        base_delay=2.0,
        max_delay=60.0,
        jitter_ratio=0.15,
    )
    if state["consecutive_quota_errors"] >= WORLDQUANT_QUOTA_COOLDOWN_THRESHOLD:
        wait_time = max(wait_time, WORLDQUANT_QUOTA_COOLDOWN_SECONDS)
    state["cooldown_until"] = max(float(state.get("cooldown_until", 0.0) or 0.0), monotonic() + wait_time)
    return wait_time, state["consecutive_quota_errors"]


def _extract_settings_context(payload: dict | None) -> dict:
    settings = payload.get("settings", {}) if isinstance(payload, dict) else {}
    return {
        "region": _coerce_setting(settings, "region"),
        "universe": _coerce_setting(settings, "universe"),
        "delay": _coerce_setting(settings, "delay"),
        "decay": _coerce_setting(settings, "decay"),
        "neutralization": _coerce_setting(settings, "neutralization"),
        "truncation": _coerce_setting(settings, "truncation"),
    }


def _build_placeholder_alpha_performance(
    fast_expr: str,
    *,
    settings_context: dict | None = None,
    check_status: str = "PENDING",
    alpha_id: str | None = None,
) -> dict:
    merged_context = _merge_settings_context(settings_context)
    return {
        "alpha_id": alpha_id or "",
        "regular_code": fast_expr,
        "turnover": None,
        "returns": None,
        "drawdown": None,
        "margin": None,
        "fitness": None,
        "sharpe": None,
        **{name: check_status for name in CHECK_RESULT_COLUMNS},
        **merged_context,
    }


def _context_value_missing(value) -> bool:
    if pd.isna(value):
        return True
    text = str(value).strip()
    return not text or text.lower() in {"none", "nan", "null", "<na>"}


def _extract_alpha_id(payload):
    if isinstance(payload, dict):
        alpha = payload.get("alpha")
        if isinstance(alpha, str) and alpha:
            return alpha
        if isinstance(alpha, dict):
            for key in ("id", "alpha_id", "alphaId"):
                value = alpha.get(key)
                if isinstance(value, str) and value:
                    return value

        for key in ("alpha_id", "alphaId"):
            value = payload.get(key)
            if isinstance(value, str) and value:
                return value

        for value in payload.values():
            alpha_id = _extract_alpha_id(value)
            if alpha_id:
                return alpha_id
    elif isinstance(payload, list):
        for item in payload:
            alpha_id = _extract_alpha_id(item)
            if alpha_id:
                return alpha_id
    return None


def _recover_alpha_id_from_history(s: requests.Session, fast_expr: str, logger=None, max_attempts: int = 3):
    normalized_expr = _normalize_expression(fast_expr)
    for attempt in range(max_attempts):
        try:
            history_response = s.get(ALPHA_HISTORY_URL)
        except requests.exceptions.RequestException as exc:
            _log_message(logger, "warning", f"Alpha history recovery request failed: {exc}")
            return None

        if history_response.status_code != 200:
            _log_message(
                logger,
                "warning",
                f"Alpha history recovery returned status {history_response.status_code}.",
            )
            return None

        payload = _safe_json(history_response)
        results = payload.get("results", payload if isinstance(payload, list) else [])
        for alpha in results:
            regular = alpha.get("regular", {}) if isinstance(alpha, dict) else {}
            code = regular.get("code") or alpha.get("regular_code")
            if _normalize_expression(code) == normalized_expr:
                alpha_id = alpha.get("id") or alpha.get("alpha_id")
                if alpha_id:
                    return alpha_id

        if attempt < max_attempts - 1:
            sleep(2)
    return None


def _dedupe_results_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    work = df.copy()
    work["__alpha_id"] = work["alpha_id"].fillna("").astype(str).str.strip()
    work["__expr_key"] = work["regular_code"].fillna("").astype(str).map(_normalize_expression)

    exprs_with_final_ids = set(
        work.loc[(work["__alpha_id"] != "") & (work["__expr_key"] != ""), "__expr_key"].tolist()
    )
    if exprs_with_final_ids:
        work = work.loc[
            ~((work["__alpha_id"] == "") & work["__expr_key"].isin(exprs_with_final_ids))
        ].copy()

    work["__dedupe_key"] = work["__alpha_id"].where(
        work["__alpha_id"] != "",
        "expr:" + work["__expr_key"],
    )
    work = work.drop_duplicates(subset=["__dedupe_key"], keep="last")
    return work[RESULT_COLUMNS]

def thread_safe_csv(func):
    """Decorator to make any CSV operation thread-safe."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        with _csv_lock:
            return func(*args, **kwargs)
    return wrapper


def _load_results_dataframe_unlocked(resolved_csv_path):
    empty_df = pd.DataFrame(columns=RESULT_COLUMNS)
    if not os.path.exists(resolved_csv_path):
        return empty_df

    df = pd.read_csv(resolved_csv_path)

    if "regular_code" not in df.columns:
        df = pd.read_csv(resolved_csv_path, header=None)
        legacy_columns = LEGACY_RESULT_COLUMNS[:-1] + ["datetime"]
        usable_columns = legacy_columns[: len(df.columns)]
        df.columns = usable_columns

    if "datetime" in df.columns and "date" not in df.columns:
        df = df.rename(columns={"datetime": "date"})

    for column in RESULT_COLUMNS:
        if column not in df.columns:
            df[column] = pd.NA

    return df[RESULT_COLUMNS]

@thread_safe_csv
def save_alpha_to_csv(alpha_performance, logger=None):
    """
    Save alpha performance data to CSV in a thread-safe manner using pandas.
    Always saves to 'simulation_results.csv' in the root directory.
    
    Parameters
    ----------
    alpha_performance : dict
        Dictionary containing alpha performance metrics
    logger : Logger, optional
        Logger instance for logging messages
    """
    if not alpha_performance:
        return
    
    csv_path = RESULTS_CSV_PATH
    
    # Add timestamp to the performance data
    alpha_performance = alpha_performance.copy()  # Create a copy to avoid modifying the original
    alpha_performance['date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    try:
        # Convert single record to DataFrame
        df_new = pd.DataFrame([alpha_performance])
        for column in RESULT_COLUMNS:
            if column not in df_new.columns:
                df_new[column] = None
        df_new = df_new[RESULT_COLUMNS]
        
        # Check if file exists and append or create new
        if os.path.exists(csv_path):
            existing_df = _load_results_dataframe_unlocked(csv_path)
            if existing_df.empty:
                df_combined = df_new.copy()
            else:
                combined_records = existing_df.to_dict("records")
                combined_records.extend(df_new.to_dict("records"))
                df_combined = pd.DataFrame.from_records(combined_records, columns=RESULT_COLUMNS)
            df_combined = _dedupe_results_dataframe(df_combined)
            df_combined.to_csv(csv_path, index=False)
        else:
            # Create new file with headers
            df_new.to_csv(csv_path, index=False)
            
    except Exception as e:
        error_msg = f"Failed to write to CSV file: {e}"
        if logger:
            logger.error(error_msg)
        else:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {error_msg}")


@thread_safe_csv
def migrate_results_csv_context(csv_path=None, *, logger=None, defaults: dict | None = None) -> dict:
    """Backfill missing context columns in legacy result CSVs using known default settings."""
    resolved_csv_path = _resolve_results_csv_path(csv_path)
    if not os.path.exists(resolved_csv_path):
        return {
            "status": "file_missing",
            "path": resolved_csv_path,
            "rows": 0,
            "filled_columns": [],
            "message": f"CSV file not found: {resolved_csv_path}",
        }

    df = _load_results_dataframe_unlocked(resolved_csv_path)
    if df.empty:
        df.to_csv(resolved_csv_path, index=False)
        return {
            "status": "empty",
            "path": resolved_csv_path,
            "rows": 0,
            "filled_columns": [],
            "message": "CSV exists but has no rows. Wrote standardized header only.",
        }

    defaults = {**DEFAULT_SIMULATION_SETTINGS_CONTEXT, **(defaults or {})}
    filled_columns = []
    filled_counts = {}
    for column in CONTEXT_RESULT_COLUMNS:
        mask = df[column].apply(_context_value_missing)
        missing_count = int(mask.sum())
        if missing_count:
            df.loc[mask, column] = defaults.get(column)
            filled_columns.append(column)
            filled_counts[column] = missing_count

    if filled_columns:
        df = df[RESULT_COLUMNS]
        df.to_csv(resolved_csv_path, index=False)
        message = (
            "Backfilled legacy context columns using default simulation settings: "
            + ", ".join(f"{column}({filled_counts[column]})" for column in filled_columns)
        )
        _log_message(logger, "log", message)
        return {
            "status": "migrated",
            "path": resolved_csv_path,
            "rows": int(len(df)),
            "filled_columns": filled_columns,
            "filled_counts": filled_counts,
            "message": message,
        }

    return {
        "status": "already_contextualized",
        "path": resolved_csv_path,
        "rows": int(len(df)),
        "filled_columns": [],
        "filled_counts": {},
        "message": "Context columns already present and populated.",
    }


def _resolve_results_csv_path(csv_path=None):
    if csv_path:
        return csv_path

    for candidate in (RESULTS_CSV_PATH, LEGACY_RESULTS_CSV_PATH):
        if os.path.exists(candidate):
            return candidate

    return RESULTS_CSV_PATH


@thread_safe_csv
def read_simulations_csv(csv_path=None):
    """
    Read simulation data from CSV in a thread-safe manner.
    Supports both the current headered `simulation_results.csv` format and the
    older headerless `simulations.csv` format.

    Parameters
    ----------
    csv_path : str, optional
        Path to the CSV file. When omitted, the function looks for
        `simulation_results.csv` first, then `simulations.csv`.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing simulation results with a standard column set.
        Returns an empty DataFrame if the file doesn't exist or an error occurs.
    """
    resolved_csv_path = _resolve_results_csv_path(csv_path)
    empty_df = pd.DataFrame(columns=RESULT_COLUMNS)

    try:
        if not os.path.exists(resolved_csv_path):
            print(f"Warning: CSV file not found: {resolved_csv_path}")
            return empty_df

        df = _load_results_dataframe_unlocked(resolved_csv_path)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

        return df

    except pd.errors.EmptyDataError:
        print(f"Warning: CSV file is empty: {resolved_csv_path}")
        return empty_df
    except Exception as e:
        error_msg = f"Error reading or processing CSV file '{resolved_csv_path}': {e}"
        print(f" Error: {error_msg}")
        # Return empty DataFrame with standard columns on error
        return empty_df

def get_alpha_performance(
    s: requests.Session,
    alpha_id: str,
    *,
    fallback_expression: str | None = None,
    fallback_settings: dict | None = None,
    logger=None,
):
    alpha = s.get("https://api.worldquantbrain.com/alphas/" + alpha_id)
    if alpha.status_code != 200:
        _log_message(logger, "warning", f"Alpha detail fetch failed for {alpha_id} with status {alpha.status_code}.")
        return None
    payload = _safe_json(alpha)
    if not isinstance(payload, dict) or not payload:
        _log_message(logger, "warning", f"Alpha detail payload for {alpha_id} was empty or invalid.")
        return None
    regular = payload.get('regular', {})
    investment_summary = payload.get('is', {})
    checks = payload.get('is', {}).get('checks', [])
    check_results = {check['name']: check['result'] for check in checks}
    settings_context = _merge_settings_context(fallback_settings)
    settings_context.update(
        {
            key: value
            for key, value in _extract_settings_context(payload).items()
            if value is not None
        }
    )

    # 创建一个包含所需信息的字典
    alpha_performance = {
        'alpha_id' : alpha_id,
        'regular_code': regular.get('code') or fallback_expression,
        'turnover': investment_summary.get('turnover'),
        'returns': investment_summary.get('returns'),
        'drawdown': investment_summary.get('drawdown'),
        'margin': investment_summary.get('margin'),
        'fitness': investment_summary.get('fitness'),
        'sharpe': investment_summary.get('sharpe'),
        'LOW_SHARPE': check_results.get('LOW_SHARPE', 'Not Found'),
        'LOW_FITNESS': check_results.get('LOW_FITNESS', 'Not Found'),
        'LOW_TURNOVER': check_results.get('LOW_TURNOVER', 'Not Found'),
        'HIGH_TURNOVER': check_results.get('HIGH_TURNOVER', 'Not Found'),
        'CONCENTRATED_WEIGHT': check_results.get('CONCENTRATED_WEIGHT', 'Not Found'),
        'LOW_SUB_UNIVERSE_SHARPE': check_results.get('LOW_SUB_UNIVERSE_SHARPE', 'Not Found'),
        'SELF_CORRELATION': check_results.get('SELF_CORRELATION', 'Not Found'),
        'MATCHES_COMPETITION': check_results.get('MATCHES_COMPETITION', 'Not Found'),
        **settings_context,
    }
    return alpha_performance


def simulate(
    s: requests.Session,
    fast_expr: str,
    timeout=300,
    logger=None,
    settings: dict | None = None,
) -> dict | None:
    simulation_settings_context = _sanitize_worldquant_settings_context(settings, logger=logger)
    simulation_data = {
    'type': 'REGULAR',
    'settings': {
        'instrumentType': 'EQUITY',
        **simulation_settings_context,
        'pasteurization': 'ON',
        'unitHandling': 'VERIFY',
        'nanHandling': 'OFF',
        'language': 'FASTEXPR',
        'visualization': False,
    },
    'regular': fast_expr }
    
    max_submit_retries = 5
    progress_retry_attempt = 0
    simulation_response = None
    
    for retry in range(1, max_submit_retries + 1):
        _wait_for_session_cooldown(s, logger=logger)
        try:
            simulation_response = s.post('https://api.worldquantbrain.com/simulations', json=simulation_data)
        except requests.exceptions.Timeout as exc:
            if retry >= max_submit_retries:
                _log_message(logger, "error", f"Simulation submit timed out after {retry} attempts: {exc}")
                return None
            wait_time = compute_backoff_delay(retry, base_delay=2.0, max_delay=45.0, jitter_ratio=0.15)
            _log_message(
                logger,
                "warning",
                f"Simulation submit timed out on attempt {retry}/{max_submit_retries}. Retrying in {wait_time:.1f}s.",
            )
            sleep(wait_time)
            continue
        except requests.exceptions.RequestException as exc:
            if retry >= max_submit_retries:
                _log_message(logger, "error", f"Simulation submit failed after {retry} attempts: {exc}")
                return None
            wait_time = compute_backoff_delay(retry, base_delay=2.0, max_delay=45.0, jitter_ratio=0.15)
            _log_message(
                logger,
                "warning",
                f"Simulation submit network error on attempt {retry}/{max_submit_retries}: {exc}. "
                f"Retrying in {wait_time:.1f}s.",
            )
            sleep(wait_time)
            continue

        if simulation_response.status_code == 201:
            _reset_session_quota_state(s)
            break

        if simulation_response.status_code == 401:
            log_msg1 = "Authentication error: Incorrect credentials."
            log_msg2 = f"Response: {simulation_response.text}"
            _log_message(logger, "error", log_msg1)
            _log_message(logger, "error", log_msg2)
            return None

        if simulation_response.status_code == 429:
            retry_after = parse_retry_after(simulation_response.headers.get("Retry-After"), minimum=1.0)
            wait_time, quota_errors = _schedule_session_quota_backoff(
                s,
                attempt=retry,
                retry_after=retry_after,
            )
            if retry >= max_submit_retries:
                break
            if quota_errors >= WORLDQUANT_QUOTA_COOLDOWN_THRESHOLD:
                _log_message(
                    logger,
                    "warning",
                    f"Simulation submit hit quota repeatedly. Cooling down for {wait_time:.1f}s before retry "
                    f"{retry + 1}/{max_submit_retries}.",
                )
            else:
                _log_message(
                    logger,
                    "warning",
                    f"Simulation submit rate-limited. Waiting {wait_time:.1f}s before retry "
                    f"{retry + 1}/{max_submit_retries}.",
                )
            continue

        _reset_session_quota_state(s)
        if is_retryable_http_status(simulation_response.status_code) and retry < max_submit_retries:
            wait_time = compute_backoff_delay(retry, base_delay=2.0, max_delay=45.0, jitter_ratio=0.15)
            _log_message(
                logger,
                "warning",
                f"Simulation submit server error {simulation_response.status_code} on attempt {retry}/{max_submit_retries}. "
                f"Retrying in {wait_time:.1f}s.",
            )
            sleep(wait_time)
            continue
        break
    
    if simulation_response.status_code != 201:
        log_msg1 = f"Failed to send simulation after {max_submit_retries} retries. Status code: {simulation_response.status_code}"
        log_msg2 = f"Response: {simulation_response.text}"
        _log_message(logger, "error", log_msg1)
        _log_message(logger, "error", log_msg2)
        return None
    
    _log_message(logger, "log", f"Simulation sent successfully: {fast_expr}")
    
    simulation_progress_url = simulation_response.headers['Location']
    finished = False
    total_wait_time = 0
    
    while not finished and total_wait_time < timeout:
        try:
            simulation_progress = s.get(simulation_progress_url)
        except requests.exceptions.Timeout as exc:
            progress_retry_attempt += 1
            wait_time = compute_backoff_delay(progress_retry_attempt, base_delay=2.0, max_delay=30.0, jitter_ratio=0.15)
            total_wait_time += wait_time
            if total_wait_time >= timeout:
                _log_message(logger, "warning", f"Simulation progress polling timed out: {exc}")
                pending = _build_placeholder_alpha_performance(
                    fast_expr,
                    settings_context=simulation_settings_context,
                    check_status="PENDING",
                )
                save_alpha_to_csv(pending, logger=logger)
                return pending
            _log_message(
                logger,
                "warning",
                f"Simulation progress timed out on retry {progress_retry_attempt}. Waiting {wait_time:.1f}s before polling again.",
            )
            sleep(wait_time)
            continue
        except requests.exceptions.RequestException as exc:
            progress_retry_attempt += 1
            wait_time = compute_backoff_delay(progress_retry_attempt, base_delay=2.0, max_delay=30.0, jitter_ratio=0.15)
            total_wait_time += wait_time
            if total_wait_time >= timeout:
                _log_message(logger, "warning", f"Simulation progress request failed repeatedly: {exc}")
                pending = _build_placeholder_alpha_performance(
                    fast_expr,
                    settings_context=simulation_settings_context,
                    check_status="PENDING",
                )
                save_alpha_to_csv(pending, logger=logger)
                return pending
            _log_message(
                logger,
                "warning",
                f"Simulation progress request failed: {exc}. Waiting {wait_time:.1f}s before polling again.",
            )
            sleep(wait_time)
            continue

        progress_payload = _safe_json(simulation_progress)
        
        if simulation_progress.status_code == 401:
            log_msg1 = "Authentication error during simulation progress monitoring."
            log_msg2 = f"Response: {simulation_progress.text}"
            _log_message(logger, "error", log_msg1)
            _log_message(logger, "error", log_msg2)
            return None

        if simulation_progress.status_code == 429:
            progress_retry_attempt += 1
            retry_after = parse_retry_after(simulation_progress.headers.get("Retry-After"), minimum=1.0)
            wait_time, quota_errors = _schedule_session_quota_backoff(
                s,
                attempt=progress_retry_attempt,
                retry_after=retry_after,
            )
            total_wait_time += wait_time
            if total_wait_time >= timeout:
                pending = _build_placeholder_alpha_performance(
                    fast_expr,
                    settings_context=simulation_settings_context,
                    check_status="PENDING",
                    alpha_id=_extract_alpha_id(progress_payload),
                )
                save_alpha_to_csv(pending, logger=logger)
                return pending
            if quota_errors >= WORLDQUANT_QUOTA_COOLDOWN_THRESHOLD:
                _log_message(
                    logger,
                    "warning",
                    f"Simulation progress hit quota repeatedly. Cooling down for {wait_time:.1f}s.",
                )
            sleep(wait_time)
            continue

        if is_retryable_http_status(simulation_progress.status_code):
            progress_retry_attempt += 1
            wait_time = compute_backoff_delay(progress_retry_attempt, base_delay=2.0, max_delay=30.0, jitter_ratio=0.15)
            total_wait_time += wait_time
            if total_wait_time >= timeout:
                pending = _build_placeholder_alpha_performance(
                    fast_expr,
                    settings_context=simulation_settings_context,
                    check_status="PENDING",
                    alpha_id=_extract_alpha_id(progress_payload),
                )
                save_alpha_to_csv(pending, logger=logger)
                return pending
            _log_message(
                logger,
                "warning",
                f"Simulation progress returned {simulation_progress.status_code}. Waiting {wait_time:.1f}s before polling again.",
            )
            sleep(wait_time)
            continue

        if simulation_progress.status_code != 200:
            log_msg1 = f"Unexpected simulation progress status: {simulation_progress.status_code}"
            log_msg2 = f"Response: {simulation_progress.text}"
            _log_message(logger, "error", log_msg1)
            _log_message(logger, "error", log_msg2)
            return None

        progress_retry_attempt = 0
        _reset_session_quota_state(s)
        wait_time = parse_retry_after(simulation_progress.headers.get("Retry-After"))
        if wait_time is None or wait_time <= 0:
            finished = True
            break
        
        total_wait_time += wait_time
        
        if total_wait_time >= timeout:
            log_msg = f"Timeout of {timeout} seconds will be exceeded. Aborting."
            _log_message(logger, "warning", log_msg)
            pending = _build_placeholder_alpha_performance(
                fast_expr,
                settings_context=simulation_settings_context,
                check_status="PENDING",
                alpha_id=_extract_alpha_id(progress_payload),
            )
            save_alpha_to_csv(pending, logger=logger)
            return pending
            
        sleep(wait_time)
        
    if finished:
        try:
            progress_payload = _safe_json(simulation_progress)
            alpha_id = _extract_alpha_id(progress_payload)
            if not alpha_id:
                _log_message(
                    logger,
                    "warning",
                    "Completed simulation payload did not include alpha_id. Trying recovery from recent alpha history.",
                )
                alpha_id = _recover_alpha_id_from_history(s, fast_expr, logger=logger)
            if not alpha_id:
                _log_message(
                    logger,
                    "error",
                    "Could not recover alpha_id for a completed simulation. Skipping this result without resubmitting.",
                )
                not_found = _build_placeholder_alpha_performance(
                    fast_expr,
                    settings_context=simulation_settings_context,
                    check_status="NOT FOUND",
                )
                save_alpha_to_csv(not_found, logger=logger)
                return not_found

            alpha_performance = get_alpha_performance(
                s,
                alpha_id,
                fallback_expression=fast_expr,
                fallback_settings=simulation_settings_context,
                logger=logger,
            )
            if alpha_performance:
                # Save the performance data to CSV
                save_alpha_to_csv(alpha_performance, logger=logger)
                # Add the expression to the returned alpha_performance object for reference
                alpha_performance['expression'] = fast_expr
                return alpha_performance
            not_found = _build_placeholder_alpha_performance(
                fast_expr,
                settings_context=simulation_settings_context,
                check_status="NOT FOUND",
                alpha_id=alpha_id,
            )
            save_alpha_to_csv(not_found, logger=logger)
            return not_found
        except Exception as e:
            log_msg = f"Error processing completed simulation: {e}"
            _log_message(logger, "error", log_msg)
            return None
    else:
        log_msg = f"Simulation timed out after {total_wait_time} seconds"
        _log_message(logger, "warning", log_msg)
        pending = _build_placeholder_alpha_performance(
            fast_expr,
            settings_context=simulation_settings_context,
            check_status="PENDING",
        )
        save_alpha_to_csv(pending, logger=logger)
        return pending
    
def get_alpha_history(s : requests.Session, pandas = True):
    all_alphas = s.get("https://api.worldquantbrain.com/users/self/alphas").json()['results']
    alpha_list = []
    for alpha in all_alphas:
        regular = alpha.get('regular', {})
        investment_summary = alpha.get('is', {})
        checks = alpha.get('is', {}).get('checks', [])
        settings_context = _extract_settings_context(alpha)

        check_results = {check['name']: check['result'] for check in checks}
        data = {
            'id': alpha.get('id'),
            'regular_code': regular.get('code'),
            'turnover': investment_summary.get('turnover'),
            'returns': investment_summary.get('returns'),
            'drawdown': investment_summary.get('drawdown'),
            'margin': investment_summary.get('margin'),
            'fitness': investment_summary.get('fitness'),
            'sharpe': investment_summary.get('sharpe'),
            'LOW_SHARPE': check_results.get('LOW_SHARPE', 'Not Found'),
            'LOW_FITNESS': check_results.get('LOW_FITNESS', 'Not Found'),
            'LOW_TURNOVER': check_results.get('LOW_TURNOVER', 'Not Found'),
            'HIGH_TURNOVER': check_results.get('HIGH_TURNOVER', 'Not Found'),
            'CONCENTRATED_WEIGHT': check_results.get('CONCENTRATED_WEIGHT', 'Not Found'),
            'LOW_SUB_UNIVERSE_SHARPE': check_results.get('LOW_SUB_UNIVERSE_SHARPE', 'Not Found'),
            'SELF_CORRELATION': check_results.get('SELF_CORRELATION', 'Not Found'),
            'MATCHES_COMPETITION': check_results.get('MATCHES_COMPETITION', 'Not Found'),
            **settings_context,
        }
        alpha_list.append(data)
        
    return pd.DataFrame(alpha_list) if pandas else alpha_list
