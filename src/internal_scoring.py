"""Internal proxy scoring for local-only alpha evaluation."""

from __future__ import annotations

import csv
import hashlib
import math
import os
import re
from concurrent.futures import ProcessPoolExecutor
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter

from src.brain import LEGACY_RESULTS_CSV_PATH, RESULT_COLUMNS, RESULTS_CSV_PATH, save_alpha_to_csv

CHECK_COLUMNS = [
    "LOW_SHARPE",
    "LOW_FITNESS",
    "LOW_TURNOVER",
    "HIGH_TURNOVER",
    "CONCENTRATED_WEIGHT",
    "LOW_SUB_UNIVERSE_SHARPE",
    "SELF_CORRELATION",
    "MATCHES_COMPETITION",
]

LOOKAHEAD_PATTERN = re.compile(r"(lead\(|future|ts_delay\([^)]*,\s*-\d+)", re.IGNORECASE)
WINDOW_PATTERN = re.compile(r",\s*(\d+)\)")
LOCAL_SCORE_VERSION = "internal_proxy_v1"
DEFAULT_SETTINGS_TEXT = "USA, TOP3000, Decay 5, Delay 1, Truncation 0.05, Neutralization Market"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SURROGATE_MODEL_VERSION = "brain_surrogate_shadow_v1"
SURROGATE_MIN_TRAINING_ROWS = 40
SURROGATE_ARTIFACT_PATH = PROJECT_ROOT / "artifacts" / "_trinh_sat" / "mo_hinh_surrogate_shadow_v1.joblib"
SURROGATE_TARGET_COLUMNS = ("fitness", "sharpe", "returns")
_SURROGATE_CACHE: dict[tuple[str, int, int], dict] = {}
_BATCH_HISTORY_INDEX: "HistoryIndex | None" = None
_BATCH_SURROGATE_BUNDLE: dict | None = None
_BATCH_SURROGATE_CSV_PATH: str | Path | None = None


@dataclass(frozen=True)
class ScoringSettings:
    region: str = "USA"
    universe: str = "TOP3000"
    delay: int = 1
    decay: int = 5
    truncation: float = 0.05
    neutralization: str = "Market"


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _safe_worker_count(max_workers: int | None, task_count: int) -> int:
    if task_count <= 0:
        return 1
    cpu_total = os.cpu_count() or 1
    auto_workers = max(1, min(task_count, min(8, cpu_total if cpu_total <= 2 else cpu_total - 1)))
    if max_workers is None or max_workers <= 0:
        return auto_workers
    return max(1, min(task_count, max_workers))


def _normalize_expression(expr: str) -> str:
    return re.sub(r"\s+", "", expr or "")


def _skeletonize(expr: str) -> str:
    return re.sub(r"\d+(?:\.\d+)?", "N", _normalize_expression(expr))


def _count_hits(text: str, terms: tuple[str, ...]) -> int:
    return sum(text.count(term) for term in terms)


def _hash_noise(expr: str, *, salt: str) -> float:
    digest = hashlib.sha1(f"{salt}:{expr}".encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    return (bucket * 2.0) - 1.0


def _balanced_parentheses(expr: str) -> bool:
    balance = 0
    for char in expr:
        if char == "(":
            balance += 1
        elif char == ")":
            balance -= 1
            if balance < 0:
                return False
    return balance == 0


def _classify_family(expr: str) -> str:
    normalized = _normalize_expression(expr).lower()
    if not normalized:
        return "unknown"
    if (
        "inverse(close)" in normalized
        or "close/ts_delay(close,3)-1" in normalized
        or "ts_corr(close,ts_delay(close,1),10)" in normalized
        or "multiply((close/ts_delay(close,3)-1),rank(volume))" in normalized
    ):
        return "simple_price_patterns"
    if "ts_regression" in normalized or "beta_last_" in normalized or "systematic_risk" in normalized or "unsystematic_risk" in normalized:
        return "residual_beta"
    if (
        "close/ts_delay(close,180)-1" in normalized
        or "ts_mean(high,10)-close" in normalized
        or "-ts_zscore(close,21)" in normalized
        or "ts_corr(open,volume,10)" in normalized
        or "ts_sum(close,10)" in normalized
        or "ts_sum(close,21)" in normalized
    ):
        return "technical_indicator"
    if "vwap" in normalized and ("close-vwap" in normalized or "abs(close-vwap)" in normalized or "(high+low)/2-close" in normalized):
        return "vwap_dislocation"
    if "ts_corr(ts_rank(volume" in normalized or "ts_delta(volume" in normalized:
        return "pv_divergence"
    if "(1-close/ts_delay(close,5))" in normalized or "(1-close/ts_delay(close,N))" in normalized:
        return "reversal_conditioned"
    if "ts_std_dev" in normalized or "abs(close-vwap)" in normalized or "volume,ts_mean(volume,63)" in normalized:
        return "shock_response"
    return "unknown"


def _extract_style_tags(expr: str) -> list[str]:
    normalized = _normalize_expression(expr).lower()
    tags = set()
    if "ts_corr" in normalized or "correlation_last_" in normalized:
        tags.add("correlation")
    if "volume" in normalized:
        tags.update({"volume", "liquidity"})
    if "vwap" in normalized:
        tags.update({"vwap", "liquidity"})
    if "rank(" in normalized or "ts_rank" in normalized:
        tags.update({"rank", "cross_sectional"})
    if "zscore" in normalized or "ts_zscore" in normalized or "winsorize" in normalized:
        tags.add("normalization")
    if "winsorize" in normalized:
        tags.add("winsorize")
    if "divide(" in normalized or "inverse(close)" in normalized or "/ts_delay(" in normalized or "/ts_mean(" in normalized:
        tags.add("ratio_like")
    if "inverse(close)" in normalized or "/ts_delay(" in normalized or "/ts_mean(" in normalized:
        tags.add("simple")
    if "beta_last_" in normalized or "ts_regression" in normalized or "systematic_risk" in normalized or "unsystematic_risk" in normalized:
        tags.update({"residual", "beta"})
    if "close/ts_delay(close,180)-1" in normalized or "ts_sum(close,10)" in normalized or "ts_sum(close,21)" in normalized:
        tags.update({"momentum", "trend"})
    if "1-close/ts_delay" in normalized or "ts_delay(close" in normalized:
        tags.add("reversal")
    if "ts_std_dev" in normalized or "systematic_risk" in normalized or "unsystematic_risk" in normalized:
        tags.add("volatility")
    if "ts_mean(high,10)-close" in normalized or "-ts_zscore(close,21)" in normalized:
        tags.add("technical")
    if (
        "inverse(close)" in normalized
        or "close/ts_delay(close,3)-1" in normalized
        or "ts_corr(close,ts_delay(close,1),10)" in normalized
        or "multiply((close/ts_delay(close,3)-1),rank(volume))" in normalized
    ):
        tags.add("book_alpha_design")
    return sorted(tags)


def _build_optimization_hints(
    *,
    syntax_ok: bool,
    no_lookahead_bias: bool,
    sharpe: float,
    turnover: float,
    uniqueness: float,
    stability: float,
    capacity: float,
    ensemble_proxy: float,
    concentration_risk: bool,
    complexity_penalty: float,
    ratio_like_terms: int,
    data_category_count: int,
) -> list[str]:
    hints = []

    if not syntax_ok:
        hints.append("Fix syntax first. Internal scoring assumes the expression is structurally valid before optimization.")
    if not no_lookahead_bias:
        hints.append("Remove look-ahead bias. Do not use future-looking fields or negative delays.")
    if sharpe < 0.8:
        hints.append("Sharpe is weak. Start simpler, add stronger normalization, and compare variants one change at a time.")
    elif sharpe < 1.2:
        hints.append("Sharpe is only moderate. Validate each tweak on held-out periods instead of relying on one in-sample score.")
    if turnover > 0.65:
        hints.append("Turnover is high. Try smoothing with ts_mean/ts_rank, longer windows like 21/63, or threshold-style trading proxies.")
    elif turnover > 0.45:
        hints.append("Turnover is elevated. Prefer slower horizons and penalize fast delta-style components before adding complexity.")
    if turnover < 0.08:
        hints.append("Turnover is very low. Add a faster trigger or shorter lookback so the signal is not too inert.")
    if uniqueness < 0.40:
        hints.append("Uniqueness is weak. Change the skeleton or switch family instead of only tuning parameters.")
    if stability < 0.45:
        hints.append("Stability is weak. Run walk-forward style checks across multiple windows and keep only changes that hold out-of-sample.")
    if capacity < 0.38:
        hints.append("Capacity looks limited. Add liquidity conditioning and avoid concentrated inverse-style structures.")
    if ensemble_proxy < 0.35:
        hints.append("The signal looks one-dimensional. Blend less-correlated motifs such as momentum plus reversal, or residual plus volume.")
    if concentration_risk:
        hints.append("Weight concentration risk is high. Wrap the signal with rank, zscore, or winsorize before deeper tuning.")
    if complexity_penalty > 0.16:
        hints.append("The expression may be overfit. Simplify it and sweep 5/10/21/63 windows before stacking more operators.")
    if ratio_like_terms == 0:
        hints.append("Make the signal more ratio-like. Compare current values to their own history or to a close peer variable before ranking.")
    if data_category_count > 3:
        hints.append("The signal mixes too many data categories. Keep the hypothesis tighter instead of blending unrelated motifs.")
    hints.append("Use an optimization loop: baseline -> one controlled change -> rescore -> keep only improvements that raise Sharpe without blowing up turnover.")

    deduped = []
    seen = set()
    for hint in hints:
        if hint not in seen:
            deduped.append(hint)
            seen.add(hint)
    return deduped[:7]


def _rescale(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return _clip((value - low) / (high - low), 0.0, 1.0)


def parse_scoring_settings(settings: str | dict | ScoringSettings | None) -> ScoringSettings:
    if settings is None:
        return ScoringSettings()
    if isinstance(settings, ScoringSettings):
        return settings

    if isinstance(settings, dict):
        region = str(settings.get("region") or "USA").strip().upper() or "USA"
        universe = str(settings.get("universe") or "TOP3000").strip().upper().replace(" ", "") or "TOP3000"
        delay = _safe_int(settings.get("delay"), 1)
        decay = _safe_int(settings.get("decay"), 5)
        truncation = _safe_float(settings.get("truncation"), 0.05)
        neutralization = str(settings.get("neutralization") or "Market").strip() or "Market"
        return ScoringSettings(
            region=region,
            universe=universe,
            delay=max(0, delay),
            decay=max(0, decay),
            truncation=_clip(truncation if truncation is not None else 0.05, 0.0, 0.25),
            neutralization=neutralization,
        )

    text = str(settings).strip()
    if not text:
        return ScoringSettings()

    upper = text.upper()
    region_match = re.search(r"\b(USA|CHN|EUR|APAC|JPN|KOR|HKG|GLB|GLOBAL)\b", upper)
    universe_match = re.search(r"\bTOP\s*(200|500|1000|3000)\b", upper)
    delay_match = re.search(r"DELAY\s*([01])\b", upper)
    decay_match = re.search(r"DECAY\s*(\d+)\b", upper)
    truncation_match = re.search(r"TRUNCATION\s*([0-9.]+)\b", upper)
    neutralization_match = re.search(
        r"NEUTRALIZATION\s*(NONE|MARKET|SECTOR|INDUSTRY|SUBINDUSTRY)\b",
        upper,
    )

    return ScoringSettings(
        region=(region_match.group(1) if region_match else "USA"),
        universe=(f"TOP{universe_match.group(1)}" if universe_match else "TOP3000"),
        delay=max(0, int(delay_match.group(1))) if delay_match else 1,
        decay=max(0, int(decay_match.group(1))) if decay_match else 5,
        truncation=_clip(float(truncation_match.group(1)), 0.0, 0.25) if truncation_match else 0.05,
        neutralization=(neutralization_match.group(1).title() if neutralization_match else "Market"),
    )


def format_scoring_settings(settings: ScoringSettings) -> str:
    return (
        f"{settings.region}, {settings.universe}, Decay {settings.decay}, "
        f"Delay {settings.delay}, Truncation {settings.truncation:.2f}, "
        f"Neutralization {settings.neutralization}"
    )


def _resolve_scoring_csv_path(csv_path: str | Path | None = None) -> Path | None:
    if csv_path:
        candidate = Path(csv_path)
        if candidate.exists():
            return candidate
        project_candidate = PROJECT_ROOT / candidate
        return project_candidate if project_candidate.exists() else candidate

    candidates = (
        Path(RESULTS_CSV_PATH),
        PROJECT_ROOT / RESULTS_CSV_PATH,
        Path(LEGACY_RESULTS_CSV_PATH),
        PROJECT_ROOT / LEGACY_RESULTS_CSV_PATH,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return PROJECT_ROOT / RESULTS_CSV_PATH


def _safe_float(value, default: float | None = None) -> float | None:
    if value is None:
        return default
    text = str(value).strip()
    if not text or text.lower() in {"none", "nan", "null"}:
        return default
    try:
        return float(text)
    except ValueError:
        return default


def _safe_int(value, default: int) -> int:
    coerced = _safe_float(value)
    return int(coerced) if coerced is not None else default


def _raw_context_from_settings(settings: str | dict | ScoringSettings | None) -> dict:
    if isinstance(settings, ScoringSettings):
        return {
            "region": settings.region,
            "universe": settings.universe,
            "delay": settings.delay,
            "decay": settings.decay,
            "truncation": settings.truncation,
            "neutralization": settings.neutralization,
        }
    if isinstance(settings, dict):
        return {key: settings.get(key) for key in ("region", "universe", "delay", "decay", "truncation", "neutralization")}
    if isinstance(settings, str) and settings.strip():
        parsed = parse_scoring_settings(settings)
        return {
            "region": parsed.region,
            "universe": parsed.universe,
            "delay": parsed.delay,
            "decay": parsed.decay,
            "truncation": parsed.truncation,
            "neutralization": parsed.neutralization,
        }
    return {}


def _extract_structural_features(
    expression: str,
    *,
    settings: str | dict | ScoringSettings | None = None,
    keep_missing_context: bool = False,
) -> dict:
    normalized = _normalize_expression(expression)
    lower = normalized.lower()
    style_tags = _extract_style_tags(expression)
    tag_set = set(style_tags)
    family = _classify_family(expression)
    alpha_type = _classify_alpha_type(expression, style_tags)
    raw_context = _raw_context_from_settings(settings)
    parsed_settings = parse_scoring_settings(settings)

    operator_count = expression.count("(")
    multiply_count = lower.count("multiply(")
    divide_count = lower.count("divide(")
    correlation_terms = _count_hits(lower, ("ts_corr(",))
    regression_terms = _count_hits(lower, ("ts_regression(",))
    normalization_terms = _count_hits(lower, ("rank(", "zscore(", "ts_zscore(", "winsorize("))
    ratio_like_terms = _count_hits(lower, ("inverse(close)", "/ts_delay(", "/ts_mean("))
    volume_terms = _count_hits(lower, ("volume", "ts_mean(volume,63)", "ts_mean(volume, 63)", "adv"))
    liquidity_terms = _count_hits(lower, ("volume", "ts_mean(volume,63)", "ts_mean(volume, 63)", "vwap"))
    residual_terms = _count_hits(lower, ("beta_last_", "systematic_risk", "unsystematic_risk", "ts_regression("))
    reactivity_terms = _count_hits(lower, ("ts_delta(", "surprise10", "surprise21", "surprise63", "reverse(", "(1-close/ts_delay("))
    cross_sectional_terms = _count_hits(lower, ("rank(",))
    temporal_smoothing_terms = _count_hits(lower, ("ts_mean(", "ts_rank("))
    data_category_count = sum(
        [
            1 if _count_hits(lower, ("close", "open", "high", "low")) else 0,
            1 if liquidity_terms else 0,
            1 if residual_terms else 0,
            1 if _count_hits(lower, ("ts_std_dev", "volatility", "systematic_risk", "unsystematic_risk")) else 0,
            1 if _count_hits(lower, ("ts_corr(", "correlation_last_", "ts_regression(")) else 0,
        ]
    )

    windows = []
    short_horizon_terms = 0
    medium_horizon_terms = 0
    long_horizon_terms = 0
    for raw_window in WINDOW_PATTERN.findall(lower):
        try:
            window = int(raw_window)
        except ValueError:
            continue
        windows.append(window)
        if window <= 10:
            short_horizon_terms += 1
        elif window <= 21:
            medium_horizon_terms += 1
        else:
            long_horizon_terms += 1

    raw_region = str(raw_context.get("region", "") or "").strip().upper()
    raw_universe = str(raw_context.get("universe", "") or "").strip().upper().replace(" ", "")
    raw_neutralization = str(raw_context.get("neutralization", "") or "").strip().title()
    context_missing_fields = 0

    if keep_missing_context:
        region = raw_region or "__missing__"
        universe = raw_universe or "__missing__"
        neutralization = raw_neutralization or "__missing__"
        if not raw_region:
            context_missing_fields += 1
        if not raw_universe:
            context_missing_fields += 1
        if not raw_neutralization:
            context_missing_fields += 1
    else:
        region = parsed_settings.region
        universe = parsed_settings.universe
        neutralization = parsed_settings.neutralization

    raw_delay = raw_context.get("delay")
    raw_decay = raw_context.get("decay")
    raw_truncation = raw_context.get("truncation")
    if keep_missing_context:
        if _safe_float(raw_delay) is None:
            context_missing_fields += 1
        if _safe_float(raw_decay) is None:
            context_missing_fields += 1
        if _safe_float(raw_truncation) is None:
            context_missing_fields += 1

    delay = _safe_int(raw_delay, parsed_settings.delay)
    decay = _safe_int(raw_decay, parsed_settings.decay)
    truncation = _safe_float(raw_truncation, parsed_settings.truncation)
    syntax_ok = bool(normalized) and _balanced_parentheses(expression)
    no_lookahead_bias = LOOKAHEAD_PATTERN.search(expression) is None
    concentration_risk = normalization_terms == 0 and (multiply_count + divide_count >= 3 or lower.count("min(") + lower.count("max(") >= 2)

    features = {
        "family": family,
        "alpha_type": alpha_type,
        "settings_region": region,
        "settings_universe": universe,
        "settings_neutralization": neutralization,
        "settings_delay": float(max(0, delay)),
        "settings_decay": float(max(0, decay)),
        "settings_truncation": float(_clip(truncation if truncation is not None else parsed_settings.truncation, 0.0, 0.25)),
        "settings_missing_fields": float(context_missing_fields),
        "operator_count": float(operator_count),
        "expression_length": float(len(normalized)),
        "multiply_count": float(multiply_count),
        "divide_count": float(divide_count),
        "correlation_terms": float(correlation_terms),
        "regression_terms": float(regression_terms),
        "normalization_terms": float(normalization_terms),
        "ratio_like_terms": float(ratio_like_terms),
        "volume_terms": float(volume_terms),
        "liquidity_terms": float(liquidity_terms),
        "residual_terms": float(residual_terms),
        "reactivity_terms": float(reactivity_terms),
        "cross_sectional_terms": float(cross_sectional_terms),
        "temporal_smoothing_terms": float(temporal_smoothing_terms),
        "data_category_count": float(data_category_count),
        "style_tag_count": float(len(style_tags)),
        "short_horizon_terms": float(short_horizon_terms),
        "medium_horizon_terms": float(medium_horizon_terms),
        "long_horizon_terms": float(long_horizon_terms),
        "max_window": float(max(windows) if windows else 0),
        "avg_window": float(sum(windows) / len(windows)) if windows else 0.0,
        "syntax_ok": 1.0 if syntax_ok else 0.0,
        "no_lookahead_bias": 1.0 if no_lookahead_bias else 0.0,
        "concentration_risk": 1.0 if concentration_risk else 0.0,
        "uses_vwap": 1.0 if "vwap" in tag_set else 0.0,
        "uses_volume": 1.0 if "volume" in tag_set else 0.0,
        "uses_reversal": 1.0 if "reversal" in tag_set else 0.0,
        "uses_momentum": 1.0 if "momentum" in tag_set else 0.0,
        "uses_residual": 1.0 if "residual" in tag_set else 0.0,
        "uses_volatility": 1.0 if "volatility" in tag_set else 0.0,
        "uses_normalization": 1.0 if "normalization" in tag_set else 0.0,
    }
    for tag in style_tags:
        features[f"style_tag__{tag}"] = 1.0
    return features


def _surrogate_csv_signature(path: Path) -> tuple[str, int, int]:
    resolved = path.resolve()
    stat = resolved.stat()
    return (str(resolved), stat.st_mtime_ns, stat.st_size)


def _load_cached_surrogate_bundle(artifact_path: Path, signature: tuple[str, int, int]) -> dict | None:
    try:
        from joblib import load
    except Exception:
        return None
    if not artifact_path.exists():
        return None
    try:
        bundle = load(artifact_path)
    except Exception:
        return None
    if not isinstance(bundle, dict):
        return None
    if bundle.get("version") != SURROGATE_MODEL_VERSION:
        return None
    if tuple(bundle.get("csv_signature", ())) != signature:
        return None
    return bundle


def _should_persist_surrogate_bundle(path: Path) -> bool:
    resolved = path.resolve()
    return resolved in {
        (PROJECT_ROOT / RESULTS_CSV_PATH).resolve(),
        (PROJECT_ROOT / LEGACY_RESULTS_CSV_PATH).resolve(),
    }


def _persist_surrogate_bundle(artifact_path: Path, bundle: dict) -> None:
    try:
        from joblib import dump
    except Exception:
        return
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    dump(bundle, artifact_path)


def load_or_train_surrogate_shadow(csv_path: str | Path | None = None) -> dict:
    path = _resolve_scoring_csv_path(csv_path)
    if path is None or not path.exists():
        return {
            "status": "missing_history",
            "version": SURROGATE_MODEL_VERSION,
            "training_rows": 0,
            "distinct_expressions": 0,
            "csv_path": str(path) if path else "",
        }

    signature = _surrogate_csv_signature(path)
    cached = _SURROGATE_CACHE.get(signature)
    if cached is not None:
        return cached

    persisted = _load_cached_surrogate_bundle(SURROGATE_ARTIFACT_PATH, signature)
    if persisted is not None:
        _SURROGATE_CACHE[signature] = persisted
        return persisted

    try:
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.feature_extraction import DictVectorizer
    except Exception:
        bundle = {
            "status": "dependency_missing",
            "version": SURROGATE_MODEL_VERSION,
            "training_rows": 0,
            "distinct_expressions": 0,
            "csv_path": str(path.resolve()),
        }
        _SURROGATE_CACHE[signature] = bundle
        return bundle

    rows = HistoryIndex._read_rows(path)
    features = []
    labels = {name: [] for name in SURROGATE_TARGET_COLUMNS}
    distinct_expressions = set()
    skipped_local_rows = 0
    for row in rows:
        alpha_id = str(row.get("alpha_id", "") or "").strip().upper()
        if alpha_id.startswith("LOCAL-"):
            skipped_local_rows += 1
            continue

        expression = row.get("regular_code", "")
        fitness = _safe_float(row.get("fitness"))
        sharpe = _safe_float(row.get("sharpe"))
        returns = _safe_float(row.get("returns"))
        if not expression or fitness is None or sharpe is None or returns is None:
            continue

        training_settings = {
            "region": row.get("region"),
            "universe": row.get("universe"),
            "delay": row.get("delay"),
            "decay": row.get("decay"),
            "truncation": row.get("truncation"),
            "neutralization": row.get("neutralization"),
        }
        features.append(_extract_structural_features(expression, settings=training_settings, keep_missing_context=True))
        labels["fitness"].append(fitness)
        labels["sharpe"].append(sharpe)
        labels["returns"].append(returns)
        distinct_expressions.add(_normalize_expression(expression))

    if len(features) < SURROGATE_MIN_TRAINING_ROWS:
        bundle = {
            "status": "insufficient_rows",
            "version": SURROGATE_MODEL_VERSION,
            "training_rows": len(features),
            "distinct_expressions": len(distinct_expressions),
            "csv_path": str(path.resolve()),
            "csv_signature": list(signature),
            "skipped_local_rows": skipped_local_rows,
            "minimum_rows": SURROGATE_MIN_TRAINING_ROWS,
        }
        _SURROGATE_CACHE[signature] = bundle
        return bundle

    vectorizer = DictVectorizer(sparse=False)
    matrix = vectorizer.fit_transform(features)
    models = {}
    label_ranges = {}
    for index, target_name in enumerate(SURROGATE_TARGET_COLUMNS, start=1):
        model = GradientBoostingRegressor(
            random_state=100 + index,
            n_estimators=180,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.9,
        )
        model.fit(matrix, labels[target_name])
        models[target_name] = model
        label_ranges[target_name] = {
            "min": min(labels[target_name]),
            "max": max(labels[target_name]),
        }

    bundle = {
        "status": "ready",
        "version": SURROGATE_MODEL_VERSION,
        "training_rows": len(features),
        "distinct_expressions": len(distinct_expressions),
        "csv_path": str(path.resolve()),
        "csv_signature": list(signature),
        "vectorizer": vectorizer,
        "models": models,
        "label_ranges": label_ranges,
        "skipped_local_rows": skipped_local_rows,
    }
    _SURROGATE_CACHE[signature] = bundle
    if _should_persist_surrogate_bundle(path):
        _persist_surrogate_bundle(SURROGATE_ARTIFACT_PATH, bundle)
    return bundle


def _predict_surrogate_shadow(
    expression: str,
    *,
    scoring_settings: ScoringSettings,
    heuristic_fitness: float,
    heuristic_sharpe: float,
    surrogate_bundle: dict | None,
) -> dict:
    bundle = surrogate_bundle or {}
    status = bundle.get("status", "missing_history")
    shadow = {
        "status": status,
        "version": bundle.get("version", SURROGATE_MODEL_VERSION),
        "training_rows": int(bundle.get("training_rows", 0) or 0),
        "distinct_expressions": int(bundle.get("distinct_expressions", 0) or 0),
        "csv_path": bundle.get("csv_path", ""),
    }
    if status != "ready":
        if bundle.get("minimum_rows"):
            shadow["minimum_rows"] = int(bundle["minimum_rows"])
        if bundle.get("skipped_local_rows"):
            shadow["skipped_local_rows"] = int(bundle["skipped_local_rows"])
        return shadow

    vectorizer = bundle["vectorizer"]
    models = bundle["models"]
    label_ranges = bundle.get("label_ranges", {})
    feature_row = _extract_structural_features(expression, settings=scoring_settings)
    matrix = vectorizer.transform([feature_row])
    predicted = {}
    for target_name, model in models.items():
        value = float(model.predict(matrix)[0])
        bounds = label_ranges.get(target_name)
        if bounds:
            value = _clip(value, float(bounds["min"]), float(bounds["max"]))
        predicted[target_name] = value

    predicted_fitness = predicted.get("fitness")
    predicted_sharpe = predicted.get("sharpe")
    alignment = "unknown"
    if predicted_fitness is not None and predicted_sharpe is not None:
        fitness_gap = predicted_fitness - heuristic_fitness
        sharpe_gap = predicted_sharpe - heuristic_sharpe
        if abs(fitness_gap) <= 0.20 and abs(sharpe_gap) <= 0.20:
            alignment = "aligned"
        elif fitness_gap <= -0.35 or sharpe_gap <= -0.35:
            alignment = "more_cautious"
        elif fitness_gap >= 0.35 or sharpe_gap >= 0.35:
            alignment = "more_optimistic"
        else:
            alignment = "mixed"
        shadow["fitness_gap_vs_heuristic"] = round(fitness_gap, 2)
        shadow["sharpe_gap_vs_heuristic"] = round(sharpe_gap, 2)

    preview_verdict = "UNAVAILABLE"
    if predicted_fitness is not None and predicted_sharpe is not None:
        required_sharpe = _required_sharpe(scoring_settings)
        if predicted_sharpe >= required_sharpe and predicted_fitness >= 1.0:
            preview_verdict = "PASS"
        elif predicted_sharpe >= (required_sharpe - 0.15) and predicted_fitness >= 0.85:
            preview_verdict = "LIKELY_PASS"
        elif predicted_sharpe >= (required_sharpe - 0.45) and predicted_fitness >= 0.45:
            preview_verdict = "BORDERLINE"
        else:
            preview_verdict = "FAIL"

    shadow.update(
        {
            "predicted_fitness": round(predicted_fitness, 2) if predicted_fitness is not None else None,
            "predicted_sharpe": round(predicted_sharpe, 2) if predicted_sharpe is not None else None,
            "predicted_returns": round(predicted.get("returns"), 4) if predicted.get("returns") is not None else None,
            "preview_verdict": preview_verdict,
            "alignment": alignment,
        }
    )
    return shadow


def _required_sharpe(settings: ScoringSettings) -> float:
    base = {
        "TOP200": 1.00,
        "TOP500": 1.05,
        "TOP1000": 1.10,
        "TOP3000": 1.20,
    }.get(settings.universe, 1.15)
    if settings.delay == 0:
        base += 0.08
    return base


def _max_turnover(settings: ScoringSettings) -> float:
    baseline = 0.80 if settings.delay >= 1 else 0.72
    baseline += min(0.08, settings.decay * 0.006)
    return _clip(baseline, 0.60, 0.95)


def _classify_alpha_type(expression: str, style_tags: list[str]) -> str:
    normalized = _normalize_expression(expression).lower()
    tag_set = set(style_tags)
    fundamental_terms = (
        "mdf_",
        "fundamental",
        "revenue",
        "income",
        "assets",
        "roe",
        "roa",
        "cap",
        "shares",
        "book",
    )
    alternative_terms = (
        "snt_",
        "news",
        "buzz",
        "implied_volatility",
        "sentiment",
        "option",
    )

    if any(term in normalized for term in alternative_terms):
        return "alternative"
    if any(term in normalized for term in fundamental_terms):
        return "fundamental"
    if {"momentum", "reversal"} & tag_set and len(tag_set & {"momentum", "reversal", "volume", "vwap", "residual", "technical"}) >= 2:
        return "hybrid"
    if "reversal" in tag_set or "vwap" in tag_set:
        return "mean_reversion"
    if {"momentum", "trend", "technical"} & tag_set:
        return "momentum"
    if len(tag_set & {"volume", "correlation", "residual"}) >= 2:
        return "hybrid"
    return "hybrid" if len(tag_set) >= 3 else "momentum"


def _build_strengths_and_weaknesses(
    *,
    sharpe: float,
    fitness: float,
    turnover: float,
    capacity: float,
    stability: float,
    uniqueness: float,
    concentration_risk: bool,
    required_sharpe: float,
    max_turnover: float,
) -> tuple[list[str], list[str]]:
    strengths = []
    weaknesses = []

    if sharpe >= required_sharpe + 0.25:
        strengths.append("Sharpe proxy vượt ngưỡng mục tiêu khá rõ.")
    elif sharpe >= required_sharpe:
        strengths.append("Sharpe proxy đang ở mức có thể chấp nhận cho settings này.")
    else:
        weaknesses.append("Sharpe proxy còn dưới ngưỡng mong muốn của universe hiện tại.")

    if fitness >= 1.3:
        strengths.append("Fitness proxy theo công thức WorldQuant đang khá khỏe.")
    elif fitness < 1.0:
        weaknesses.append("Fitness proxy chưa qua ngưỡng tối thiểu 1.0.")

    if turnover <= max_turnover * 0.55:
        strengths.append("Turnover vẫn được kiểm soát tương đối tốt.")
    elif turnover > max_turnover:
        weaknesses.append("Turnover đang quá cao so với cấu hình hiện tại.")

    if stability >= 0.55:
        strengths.append("Tín hiệu có độ ổn định tương đối tốt.")
    else:
        weaknesses.append("Độ ổn định còn yếu, dễ overfit khi đổi tham số.")

    if capacity >= 0.50:
        strengths.append("Capacity proxy ổn, ít dấu hiệu quá khó scale.")
    else:
        weaknesses.append("Capacity proxy còn thấp, nên ưu tiên liquidity conditioning.")

    if uniqueness >= 0.55:
        strengths.append("Alpha có độ độc nhất tạm ổn.")
    else:
        weaknesses.append("Alpha còn khá giống các mẫu đã có hoặc cùng skeleton.")

    if concentration_risk:
        weaknesses.append("Có dấu hiệu concentrated weight nếu không bọc signal tốt hơn.")

    return strengths[:5], weaknesses[:5]


def _build_structured_suggestions(
    *,
    alpha_type: str,
    sharpe: float,
    turnover: float,
    uniqueness: float,
    stability: float,
    concentration_risk: bool,
    settings: ScoringSettings,
) -> list[dict]:
    suggestions = []

    if sharpe < _required_sharpe(settings):
        suggestions.append(
            {
                "type": "concept",
                "description": "Tăng chất lượng signal trước khi tune thêm tham số.",
                "example": "So sanh raw signal voi rank(ts_zscore(...)) va winsorize(...) tren cung horizon 5/10/21/63.",
            }
        )
    if turnover > _max_turnover(settings):
        suggestions.append(
            {
                "type": "params",
                "description": "Giam turnover bang decay cao hon, horizon dai hon, hoac trigger it nhay hon.",
                "example": "Thu Decay 8-12, Delay 1, hoac boc signal bang ts_mean/ts_rank truoc khi rank ngoai cung.",
            }
        )
    if uniqueness < 0.45:
        suggestions.append(
            {
                "type": "code",
                "description": "Doi skeleton hoac blend them mot motif it tuong quan hon.",
                "example": "Neu alpha dang nghieng ve vwap reversion, thu them residual/volume thay vi chi doi window.",
            }
        )
    if stability < 0.45:
        suggestions.append(
            {
                "type": "concept",
                "description": "Chay walk-forward noi bo va giu rule moi chi khi out-of-sample cung tot len.",
                "example": "Baseline -> sua 1 thay doi -> score lai tren 5/10/21/63 -> rollback neu holdout xau di.",
            }
        )
    if concentration_risk:
        suggestions.append(
            {
                "type": "code",
                "description": "Giam nguy co weight tap trung bang normalization va truncation chat hon.",
                "example": "Boc them rank/zscore/winsorize va can nhac Truncation 0.01-0.03 neu signal co cuc tri manh.",
            }
        )
    if alpha_type in {"momentum", "mean_reversion"} and len(suggestions) < 4:
        suggestions.append(
            {
                "type": "concept",
                "description": "Thu ensemble voi mot nhanh khac gia dinh de giam do mong manh cua mot motif don.",
                "example": "0.6 alpha hien tai + 0.4 mot nhanh residual hoac volume-conditioned signal.",
            }
        )

    return suggestions[:4]


def _build_analysis_text(
    *,
    verdict: str,
    alpha_type: str,
    settings: ScoringSettings,
    sharpe: float,
    fitness: float,
    turnover: float,
    stability: float,
    uniqueness: float,
) -> str:
    return (
        f"Alpha duoc xep loai {alpha_type} va duoc cham voi settings {format_scoring_settings(settings)}. "
        f"Verdict hien tai la {verdict} vi Sharpe proxy={sharpe:.2f}, Fitness proxy={fitness:.2f}, "
        f"Turnover={turnover:.2%}, Stability={stability:.2f}, Uniqueness={uniqueness:.2f}. "
        "Danh gia nay la proxy noi bo de loc y tuong, khong phai ket qua backtest proprietary cua WorldQuant."
    )


def _build_score_breakdown(
    *,
    ic_proxy: float,
    sharpe_score: float,
    fitness_score: float,
    turnover_score: float,
    capacity: float,
    stability: float,
    uniqueness: float,
    ensemble_proxy: float,
    validation_score: float,
    turnover_gap_penalty: float,
    complexity_penalty: float,
    duplicate_penalty: float,
    surrogate_penalty: float,
    concentration_risk: bool,
    required_sharpe: float,
    max_turnover: float,
    alpha_score: float,
) -> dict:
    contributors = {
        "ic_proxy": round(0.17 * ic_proxy, 4),
        "sharpe": round(0.17 * sharpe_score, 4),
        "fitness": round(0.14 * fitness_score, 4),
        "turnover_fit": round(0.12 * turnover_score, 4),
        "capacity": round(0.14 * capacity, 4),
        "stability": round(0.13 * stability, 4),
        "uniqueness": round(0.13 * uniqueness, 4),
        "ensemble": round(0.08 * ensemble_proxy, 4),
        "validation": round(0.05 * validation_score, 4),
    }
    penalties = {
        "turnover_gap": round(0.07 * turnover_gap_penalty, 4),
        "complexity": round(0.05 * min(1.0, complexity_penalty / 0.35), 4),
        "duplicate": round(0.08 * min(1.0, duplicate_penalty / 0.55), 4),
        "surrogate_shadow": round(surrogate_penalty / 100.0, 4),
        "concentration": round(0.03 if concentration_risk else 0.0, 4),
    }
    gross_score = round(sum(contributors.values()), 4)
    penalty_score = round(sum(penalties.values()), 4)
    return {
        "contributors": contributors,
        "penalties": penalties,
        "gross_score": gross_score,
        "net_score": round(max(0.0, gross_score - penalty_score), 4),
        "alpha_score": round(alpha_score, 1),
        "gates": {
            "required_sharpe": round(required_sharpe, 3),
            "max_turnover": round(max_turnover, 4),
        },
    }


def _summarize_surrogate_shadow_risk(
    shadow: dict,
    *,
    heuristic_fitness: float,
    heuristic_sharpe: float,
) -> dict:
    summary = {
        "alpha_penalty": 0.0,
        "reasons": [],
        "hard_signal": "none",
        "fitness_gap": 0.0,
        "sharpe_gap": 0.0,
    }
    if str(shadow.get("status") or "").lower() != "ready":
        return summary

    preview_verdict = str(shadow.get("preview_verdict") or "UNAVAILABLE").upper()
    alignment = str(shadow.get("alignment") or "unknown")
    predicted_fitness = shadow.get("predicted_fitness")
    predicted_sharpe = shadow.get("predicted_sharpe")
    fitness_gap = max(0.0, heuristic_fitness - float(predicted_fitness or 0.0))
    sharpe_gap = max(0.0, heuristic_sharpe - float(predicted_sharpe or 0.0))
    penalty = 0.0
    reasons = []

    if preview_verdict == "FAIL":
        penalty += 10.0
        reasons.append("surrogate_preview_fail")
    elif preview_verdict == "BORDERLINE":
        penalty += 4.0
        reasons.append("surrogate_preview_borderline")
    elif preview_verdict == "LIKELY_PASS":
        penalty += 1.5
        reasons.append("surrogate_less_confident_than_heuristic")

    if alignment == "more_cautious":
        penalty += 6.0
        reasons.append("surrogate_more_cautious")
    elif alignment == "mixed":
        penalty += 2.0
        reasons.append("surrogate_mixed_signal")

    if fitness_gap >= 0.25:
        penalty += min(6.0, fitness_gap * 3.0)
        reasons.append(f"fitness_gap={round(fitness_gap, 2)}")
    if sharpe_gap >= 0.25:
        penalty += min(6.0, sharpe_gap * 2.5)
        reasons.append(f"sharpe_gap={round(sharpe_gap, 2)}")

    hard_signal = "none"
    if preview_verdict == "FAIL" and alignment == "more_cautious" and (fitness_gap + sharpe_gap) >= 1.10:
        hard_signal = "severe_mismatch"
        penalty = max(penalty, 18.0)
    elif preview_verdict == "FAIL" and alignment in {"more_cautious", "mixed"}:
        hard_signal = "soft_mismatch"

    summary.update(
        {
            "alpha_penalty": round(_clip(penalty, 0.0, 20.0), 2),
            "reasons": list(dict.fromkeys(reasons)),
            "hard_signal": hard_signal,
            "fitness_gap": round(fitness_gap, 2),
            "sharpe_gap": round(sharpe_gap, 2),
        }
    )
    return summary


@dataclass
class HistoryIndex:
    total_rows: int = 0
    exact_counts: Counter[str] = field(default_factory=Counter)
    skeleton_counts: Counter[str] = field(default_factory=Counter)
    family_counts: Counter[str] = field(default_factory=Counter)
    family_score_totals: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    family_observations: Counter[str] = field(default_factory=Counter)
    family_pass_counts: Counter[str] = field(default_factory=Counter)

    @classmethod
    def from_csv(cls, csv_path: str | Path | None = None) -> "HistoryIndex":
        instance = cls()
        path = cls._resolve_csv_path(csv_path)
        if path is None or not path.exists():
            return instance

        for row in cls._read_rows(path):
            expression = row.get("regular_code", "")
            if not expression:
                continue
            instance.observe_expression(expression, row)
        return instance

    @staticmethod
    def _resolve_csv_path(csv_path: str | Path | None) -> Path | None:
        if csv_path:
            return Path(csv_path)
        for candidate in (RESULTS_CSV_PATH, LEGACY_RESULTS_CSV_PATH):
            path = Path(candidate)
            if path.exists():
                return path
        return Path(RESULTS_CSV_PATH)

    @staticmethod
    def _read_rows(path: Path) -> list[dict]:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.reader(handle))

        if not rows:
            return []

        first = [cell.strip() for cell in rows[0]]
        has_header = "regular_code" in first or "alpha_id" in first
        if has_header:
            headers = first
            data_rows = rows[1:]
        else:
            headers = RESULT_COLUMNS[: len(first)]
            data_rows = rows

        parsed_rows = []
        for raw_row in data_rows:
            if not any(cell.strip() for cell in raw_row):
                continue
            row = {}
            for index, header in enumerate(headers):
                row[header] = raw_row[index].strip() if index < len(raw_row) else ""
            parsed_rows.append(row)
        return parsed_rows

    @staticmethod
    def _coerce_float(value) -> float | None:
        if value is None:
            return None
        text = str(value).strip()
        if not text or text.lower() in {"none", "nan", "null"}:
            return None
        try:
            return float(text)
        except ValueError:
            return None

    @staticmethod
    def _passes_all(row: dict) -> bool:
        available = [name for name in CHECK_COLUMNS if name in row]
        if not available:
            return False
        return all(str(row.get(name, "")).upper() == "PASS" for name in available)

    def observe_expression(self, expression: str, row: dict | None = None) -> None:
        normalized = _normalize_expression(expression)
        if not normalized:
            return

        skeleton = _skeletonize(expression)
        family = _classify_family(expression)
        self.total_rows += 1
        self.exact_counts[normalized] += 1
        self.skeleton_counts[skeleton] += 1
        self.family_counts[family] += 1

        if row:
            sharpe = self._coerce_float(row.get("sharpe"))
            fitness = self._coerce_float(row.get("fitness"))
            observed_score = 0.0
            if sharpe is not None:
                observed_score += sharpe * 0.6
            if fitness is not None:
                observed_score += fitness * 0.4
            self.family_score_totals[family] += observed_score
            self.family_observations[family] += 1
            if self._passes_all(row):
                self.family_pass_counts[family] += 1

    def exact_match_count(self, expression: str) -> int:
        return self.exact_counts[_normalize_expression(expression)]

    def skeleton_match_count(self, expression: str) -> int:
        return self.skeleton_counts[_skeletonize(expression)]

    def family_count(self, expression: str) -> int:
        return self.family_counts[_classify_family(expression)]

    def family_history_bonus(self, expression: str) -> float:
        family = _classify_family(expression)
        observations = self.family_observations[family]
        if observations == 0:
            return 0.0
        average_score = self.family_score_totals[family] / observations
        pass_rate = self.family_pass_counts[family] / observations
        return _clip((average_score * 0.025) + (pass_rate * 0.06) - 0.03, -0.05, 0.05)


def _init_batch_scoring_worker(
    history_index: HistoryIndex,
    surrogate_bundle: dict | None,
    surrogate_csv_path: str | Path | None,
) -> None:
    global _BATCH_HISTORY_INDEX, _BATCH_SURROGATE_BUNDLE, _BATCH_SURROGATE_CSV_PATH
    _BATCH_HISTORY_INDEX = history_index
    _BATCH_SURROGATE_BUNDLE = surrogate_bundle
    _BATCH_SURROGATE_CSV_PATH = surrogate_csv_path


def _score_expression_batch_task(task: tuple[str, str | dict | ScoringSettings | None]) -> dict:
    expression, settings = task
    return score_expression(
        expression,
        history_index=_BATCH_HISTORY_INDEX or HistoryIndex(),
        settings=settings,
        surrogate_bundle=_BATCH_SURROGATE_BUNDLE,
        surrogate_csv_path=_BATCH_SURROGATE_CSV_PATH,
    )


def score_expressions_batch(
    expressions: list[str],
    *,
    history_index: HistoryIndex | None = None,
    settings_list: list[str | dict | ScoringSettings | None] | None = None,
    max_workers: int | None = None,
    min_parallel_tasks: int = 4,
    surrogate_bundle: dict | None = None,
    surrogate_csv_path: str | Path | None = None,
    return_profile: bool = False,
) -> list[dict] | tuple[list[dict], dict]:
    expressions = list(expressions or [])
    if settings_list is None:
        settings_list = [None] * len(expressions)
    else:
        settings_list = list(settings_list)

    if len(settings_list) != len(expressions):
        raise ValueError("settings_list must match expressions length")

    history_index = history_index or HistoryIndex()
    tasks = [(expression, settings_list[index]) for index, expression in enumerate(expressions)]
    if not tasks:
        profile = {
            "task_count": 0,
            "worker_count": 0,
            "mode": "sequential",
            "prepare_seconds": 0.0,
            "scoring_seconds": 0.0,
            "total_seconds": 0.0,
            "bottleneck_hint": "none",
        }
        return ([], profile) if return_profile else []

    overall_started_at = perf_counter()
    prepare_started_at = perf_counter()
    resolved_surrogate_bundle = surrogate_bundle or load_or_train_surrogate_shadow(surrogate_csv_path)
    prepare_seconds = perf_counter() - prepare_started_at

    worker_count = _safe_worker_count(max_workers, len(tasks))
    use_parallel = worker_count > 1 and len(tasks) >= max(2, int(min_parallel_tasks))

    scoring_started_at = perf_counter()
    if use_parallel:
        with ProcessPoolExecutor(
            max_workers=worker_count,
            initializer=_init_batch_scoring_worker,
            initargs=(history_index, resolved_surrogate_bundle, surrogate_csv_path),
        ) as executor:
            results = list(executor.map(_score_expression_batch_task, tasks))
    else:
        results = [
            score_expression(
                expression,
                history_index=history_index,
                settings=settings,
                surrogate_bundle=resolved_surrogate_bundle,
                surrogate_csv_path=surrogate_csv_path,
            )
            for expression, settings in tasks
        ]
    scoring_seconds = perf_counter() - scoring_started_at
    total_seconds = perf_counter() - overall_started_at

    if prepare_seconds >= scoring_seconds * 0.8:
        bottleneck_hint = "io_and_model_load"
    elif use_parallel and scoring_seconds >= prepare_seconds:
        bottleneck_hint = "parse_and_python_cpu"
    else:
        bottleneck_hint = "single_process_cpu"

    profile = {
        "task_count": len(tasks),
        "worker_count": worker_count if use_parallel else 1,
        "mode": "parallel" if use_parallel else "sequential",
        "prepare_seconds": round(prepare_seconds, 4),
        "scoring_seconds": round(scoring_seconds, 4),
        "total_seconds": round(total_seconds, 4),
        "bottleneck_hint": bottleneck_hint,
    }
    return (results, profile) if return_profile else results


def score_expression(
    expression: str,
    *,
    history_index: HistoryIndex | None = None,
    settings: str | dict | ScoringSettings | None = None,
    surrogate_bundle: dict | None = None,
    surrogate_csv_path: str | Path | None = None,
) -> dict:
    history_index = history_index or HistoryIndex()
    scoring_settings = parse_scoring_settings(settings)
    normalized = _normalize_expression(expression)
    lower = normalized.lower()
    style_tags = _extract_style_tags(expression)
    style_diversification = _clip(len(style_tags) / 8.0, 0.0, 1.0)
    signal_buckets = {
        tag
        for tag in style_tags
        if tag in {"momentum", "reversal", "volume", "vwap", "residual", "technical", "correlation", "volatility"}
    }
    ensemble_proxy = 0.12 + 0.28 * style_diversification + 0.18 * min(1.0, len(signal_buckets) / 3.0)
    if "normalization" in style_tags:
        ensemble_proxy += 0.08
    if {"momentum", "reversal"} <= set(style_tags):
        ensemble_proxy += 0.07
    if {"residual", "volume"} <= set(style_tags) or {"residual", "vwap"} <= set(style_tags):
        ensemble_proxy += 0.07
    ensemble_proxy = _clip(ensemble_proxy, 0.0, 1.0)

    operator_count = expression.count("(")
    length = len(normalized)
    multiply_count = lower.count("multiply(")
    divide_count = lower.count("divide(")
    correlation_terms = _count_hits(lower, ("ts_corr(",))
    regression_terms = _count_hits(lower, ("ts_regression(",))
    normalization_terms = _count_hits(lower, ("rank(", "zscore(", "ts_zscore(", "winsorize("))
    robustness_wrappers = _count_hits(lower, ("rank(", "zscore(", "ts_zscore(", "winsorize("))
    ratio_like_terms = _count_hits(lower, ("inverse(close)", "/ts_delay(", "/ts_mean("))
    volume_terms = _count_hits(lower, ("volume", "ts_mean(volume,63)", "ts_mean(volume, 63)", "adv"))
    liquidity_terms = _count_hits(lower, ("volume", "ts_mean(volume,63)", "ts_mean(volume, 63)", "vwap"))
    residual_terms = _count_hits(lower, ("beta_last_", "systematic_risk", "unsystematic_risk", "ts_regression("))
    reactivity_terms = _count_hits(lower, ("ts_delta(", "surprise10", "surprise21", "surprise63", "reverse(", "(1-close/ts_delay("))
    cross_sectional_terms = _count_hits(lower, ("rank(",))
    temporal_smoothing_terms = _count_hits(lower, ("ts_mean(", "ts_rank("))
    data_category_count = sum(
        [
            1 if _count_hits(lower, ("close", "open", "high", "low")) else 0,
            1 if liquidity_terms else 0,
            1 if residual_terms else 0,
            1 if _count_hits(lower, ("ts_std_dev", "volatility", "systematic_risk", "unsystematic_risk")) else 0,
            1 if _count_hits(lower, ("ts_corr(", "correlation_last_", "ts_regression(")) else 0,
        ]
    )
    long_horizon_terms = 0
    medium_horizon_terms = 0
    short_horizon_terms = 0
    for raw_window in WINDOW_PATTERN.findall(lower):
        try:
            window = int(raw_window)
        except ValueError:
            continue
        if window <= 10:
            short_horizon_terms += 1
        elif window <= 21:
            medium_horizon_terms += 1
        else:
            long_horizon_terms += 1

    family = _classify_family(expression)
    exact_match_count = history_index.exact_match_count(expression)
    skeleton_match_count = history_index.skeleton_match_count(expression)
    family_count = history_index.family_count(expression)
    family_density = (family_count / history_index.total_rows) if history_index.total_rows else 0.0

    syntax_ok = bool(normalized) and _balanced_parentheses(expression)
    no_lookahead_bias = LOOKAHEAD_PATTERN.search(expression) is None
    validation_score = 1.0 if syntax_ok and no_lookahead_bias else 0.0
    alpha_type = _classify_alpha_type(expression, style_tags)
    tag_set = set(style_tags)

    reactive_price_volume_risk = 0.0
    if {"reversal", "volume"} <= tag_set:
        reactive_price_volume_risk += 0.12
    if {"reversal", "volume", "volatility"} <= tag_set:
        reactive_price_volume_risk += 0.08
    if alpha_type in {"mean_reversion", "hybrid"} and short_horizon_terms >= 1:
        reactive_price_volume_risk += 0.08
    if medium_horizon_terms >= 1:
        reactive_price_volume_risk += 0.04
    if cross_sectional_terms >= 3 and normalization_terms >= 4:
        reactive_price_volume_risk += 0.08
    if temporal_smoothing_terms == 0:
        reactive_price_volume_risk += 0.06
    if residual_terms == 0 and "vwap" not in tag_set and family in {"reversal_conditioned", "shock_response", "pv_divergence", "simple_price_patterns", "unknown"}:
        reactive_price_volume_risk += 0.06
    reactive_price_volume_risk = _clip(reactive_price_volume_risk, 0.0, 0.42)

    complexity_penalty = 0.0
    if operator_count < 2:
        complexity_penalty += 0.20
    if length > 220:
        complexity_penalty += 0.18
    if length > 320:
        complexity_penalty += 0.10
    complexity_penalty += max(0.0, multiply_count - 2) * 0.04
    complexity_penalty += max(0.0, divide_count - 2) * 0.035
    complexity_penalty += max(0.0, data_category_count - 3) * 0.05

    duplicate_penalty = 0.0
    if exact_match_count:
        duplicate_penalty += 0.55
    duplicate_penalty += min(0.24, skeleton_match_count * 0.08)
    duplicate_penalty += min(0.22, family_density * 0.38)

    concentration_risk = normalization_terms == 0 and (multiply_count + divide_count >= 3 or lower.count("min(") + lower.count("max(") >= 2)

    ic_proxy = 0.18
    ic_proxy += min(0.18, normalization_terms * 0.05)
    ic_proxy += min(0.16, correlation_terms * 0.06)
    ic_proxy += min(0.12, regression_terms * 0.05)
    ic_proxy += min(0.10, volume_terms * 0.035)
    ic_proxy += min(0.08, residual_terms * 0.03)
    ic_proxy += min(0.08, ratio_like_terms * 0.025)
    ic_proxy += 0.08 * ensemble_proxy
    ic_proxy += history_index.family_history_bonus(expression)
    ic_proxy -= complexity_penalty * 0.65
    ic_proxy -= duplicate_penalty * 0.35
    ic_proxy += _hash_noise(normalized, salt="ic") * 0.03
    ic_proxy = _clip(ic_proxy, 0.02, 0.96)

    turnover = 0.16
    turnover += min(0.22, short_horizon_terms * 0.07)
    turnover += min(0.10, medium_horizon_terms * 0.03)
    turnover += min(0.18, reactivity_terms * 0.05)
    turnover -= min(0.12, long_horizon_terms * 0.035)
    turnover -= min(0.08, temporal_smoothing_terms * 0.028)
    turnover -= 0.015 if liquidity_terms and long_horizon_terms >= 1 else 0.0
    turnover += min(0.08, cross_sectional_terms * 0.014) if short_horizon_terms >= 1 else 0.0
    turnover += 0.24 * reactive_price_volume_risk
    turnover -= min(0.05, scoring_settings.decay * 0.004)
    turnover -= 0.015 if scoring_settings.delay >= 1 else -0.03
    turnover += abs(_hash_noise(normalized, salt="turnover")) * 0.04
    turnover = _clip(turnover, 0.03, 1.20)

    stability = 0.30
    stability += min(0.13, normalization_terms * 0.03)
    stability += min(0.12, residual_terms * 0.04)
    stability += min(0.10, long_horizon_terms * 0.03)
    stability += min(0.08, temporal_smoothing_terms * 0.03)
    stability += min(0.05, robustness_wrappers * 0.01)
    stability += min(0.06, ratio_like_terms * 0.02)
    stability += 0.02 if liquidity_terms else 0.0
    stability += 0.10 * ensemble_proxy
    stability += 0.02 if scoring_settings.delay >= 1 else -0.02
    stability += min(0.03, scoring_settings.decay * 0.003)
    stability += 0.03 if scoring_settings.neutralization in {"Sector", "Industry", "Subindustry"} else 0.01
    stability -= max(0.0, turnover - 0.50) * 0.18
    stability -= 0.22 * reactive_price_volume_risk
    stability -= max(0.0, cross_sectional_terms - temporal_smoothing_terms - 2) * 0.018
    stability -= complexity_penalty * 0.45
    stability -= duplicate_penalty * 0.18
    stability += _hash_noise(normalized, salt="stability") * 0.03
    stability = _clip(stability, 0.02, 0.95)

    capacity = 0.34
    capacity += min(0.16, liquidity_terms * 0.05)
    capacity += min(0.08, normalization_terms * 0.025)
    capacity += 0.03 if family in {"residual_beta", "vwap_dislocation"} else 0.0
    capacity += 0.04 if family == "simple_price_patterns" else 0.0
    capacity += 0.05 * ensemble_proxy
    capacity += 0.035 if scoring_settings.truncation <= 0.03 else 0.015
    capacity += 0.02 if data_category_count <= 2 else -0.02 if data_category_count > 3 else 0.0
    capacity -= max(0.0, turnover - 0.35) * 0.34
    capacity -= 0.18 * reactive_price_volume_risk
    capacity -= 0.02 if cross_sectional_terms >= 3 and short_horizon_terms >= 1 else 0.0
    capacity -= 0.14 if concentration_risk else 0.0
    capacity -= complexity_penalty * 0.20
    capacity += _hash_noise(normalized, salt="capacity") * 0.025
    capacity = _clip(capacity, 0.02, 0.95)

    uniqueness = 0.88
    uniqueness -= 0.70 if exact_match_count else 0.0
    uniqueness -= min(0.24, skeleton_match_count * 0.08)
    uniqueness -= min(0.22, family_density * 0.36)
    uniqueness += 0.04 if family_count == 0 else 0.0
    uniqueness += 0.03 if len(signal_buckets) >= 2 else 0.0
    uniqueness += 0.02 if family == "simple_price_patterns" and ratio_like_terms else 0.0
    uniqueness += 0.015 if scoring_settings.neutralization in {"Industry", "Subindustry"} else 0.0
    uniqueness -= 0.18 * reactive_price_volume_risk
    uniqueness -= 0.05 if cross_sectional_terms >= 3 and {"reversal", "volume"} <= tag_set else 0.0
    uniqueness += _hash_noise(normalized, salt="uniqueness") * 0.02
    uniqueness = _clip(uniqueness, 0.0, 0.98)

    edge_signal = 0.28 * ic_proxy
    edge_signal += 0.20 * stability
    edge_signal += 0.18 * capacity
    edge_signal += 0.14 * uniqueness
    edge_signal += 0.10 * ensemble_proxy
    edge_signal += 0.06 * validation_score
    edge_signal -= 0.18 * abs(turnover - 0.25)
    edge_signal -= 0.12 * complexity_penalty
    edge_signal -= 0.14 * duplicate_penalty

    sigma_daily = 0.008 + 0.015 * turnover + 0.004 * (1.0 - stability) + min(0.004, operator_count * 0.00035)
    mu_daily = (edge_signal - 0.14) * 0.0038
    sharpe = 0.0
    if sigma_daily > 0:
        sharpe = (mu_daily / sigma_daily) * math.sqrt(252.0)
    sharpe = _clip(sharpe, -2.0, 3.5)
    sharpe += 0.015 if scoring_settings.delay >= 1 else -0.015
    sharpe += min(0.04, scoring_settings.decay * 0.003)
    sharpe += 0.02 if scoring_settings.neutralization in {"Sector", "Industry", "Subindustry"} else 0.0
    sharpe = _clip(sharpe, -2.0, 3.5)

    annual_return = _clip(mu_daily * 252.0, -0.45, 0.65)
    annual_return += 0.006 if scoring_settings.delay >= 1 else -0.006
    annual_return = _clip(annual_return, -0.45, 0.65)
    drawdown = _clip(
        0.04 + (1.0 - stability) * 0.18 + max(0.0, turnover - 0.35) * 0.22 - min(0.03, scoring_settings.truncation * 0.4),
        0.01,
        0.60,
    )
    margin = round(mu_daily * (0.60 + (0.40 * capacity)), 6)
    quality_proxy = (
        (0.55 * sharpe)
        + (0.85 * ic_proxy)
        + (0.45 * stability)
        + (0.30 * capacity)
        + (0.25 * uniqueness)
        - (0.80 * abs(turnover - 0.25))
    )
    quality_proxy = _clip(quality_proxy, -1.0, 2.5)
    effective_return = _clip(abs(annual_return), 0.02, 0.24)
    raw_fitness = sharpe * math.sqrt(effective_return / max(turnover, 0.125))
    raw_fitness *= 0.78 + (0.14 * validation_score)
    raw_fitness *= 0.82 + (0.10 * stability) + (0.08 * uniqueness)
    if raw_fitness >= 0:
        fitness = 3.15 * math.tanh(raw_fitness / 2.85)
    else:
        fitness = 1.15 * math.tanh(raw_fitness / 1.15)
    fitness = _clip(fitness, -1.0, 3.15)

    sharpe_score = _rescale(sharpe, -0.25, 2.80)
    fitness_score = _rescale(fitness, 0.25, 2.80)
    turnover_gap_penalty = _clip(abs(turnover - 0.22) / 0.58, 0.0, 1.0)
    turnover_score = 1.0 - turnover_gap_penalty
    out_of_sample_risk = _clip(
        0.45 * reactive_price_volume_risk
        + 0.20 * _clip((turnover - 0.28) / 0.32, 0.0, 1.0)
        + 0.15 * _clip((0.60 - stability) / 0.25, 0.0, 1.0)
        + 0.10 * _clip((cross_sectional_terms - temporal_smoothing_terms - 1) / 3.0, 0.0, 1.0)
        + 0.10 * (1.0 if alpha_type in {"mean_reversion", "hybrid"} and {"reversal", "volume"} <= tag_set else 0.0),
        0.0,
        1.0,
    )
    alpha_score = 100.0 * (
        (0.17 * ic_proxy)
        + (0.17 * sharpe_score)
        + (0.14 * fitness_score)
        + (0.12 * turnover_score)
        + (0.14 * capacity)
        + (0.13 * stability)
        + (0.13 * uniqueness)
        + (0.08 * ensemble_proxy)
        + (0.05 * validation_score)
        - (0.07 * turnover_gap_penalty)
        - (0.05 * min(1.0, complexity_penalty / 0.35))
        - (0.08 * min(1.0, duplicate_penalty / 0.55))
        - (0.06 * out_of_sample_risk)
        - (0.03 if concentration_risk else 0.0)
    )
    alpha_score = _clip(alpha_score, 0.0, 100.0)
    resolved_surrogate_bundle = surrogate_bundle or load_or_train_surrogate_shadow(surrogate_csv_path)
    surrogate_shadow = _predict_surrogate_shadow(
        expression,
        scoring_settings=scoring_settings,
        heuristic_fitness=fitness,
        heuristic_sharpe=sharpe,
        surrogate_bundle=resolved_surrogate_bundle,
    )
    surrogate_shadow_summary = _summarize_surrogate_shadow_risk(
        surrogate_shadow,
        heuristic_fitness=fitness,
        heuristic_sharpe=sharpe,
    )
    surrogate_penalty = surrogate_shadow_summary["alpha_penalty"]
    alpha_score = _clip(alpha_score - surrogate_penalty, 0.0, 100.0)
    optimization_hints = _build_optimization_hints(
        syntax_ok=syntax_ok,
        no_lookahead_bias=no_lookahead_bias,
        sharpe=sharpe,
        turnover=turnover,
        uniqueness=uniqueness,
        stability=stability,
        capacity=capacity,
        ensemble_proxy=ensemble_proxy,
        concentration_risk=concentration_risk,
        complexity_penalty=complexity_penalty,
        ratio_like_terms=ratio_like_terms,
        data_category_count=data_category_count,
    )

    required_sharpe = _required_sharpe(scoring_settings)
    max_turnover = _max_turnover(scoring_settings)
    low_sharpe = "PASS" if sharpe >= required_sharpe else "FAIL"
    low_fitness = "PASS" if fitness >= 1.0 else "FAIL"
    low_turnover = "FAIL" if turnover < 0.05 else "PASS"
    high_turnover = "FAIL" if turnover > max_turnover else "PASS"
    concentrated_weight = "FAIL" if concentration_risk and scoring_settings.truncation >= 0.05 else "PASS"
    low_sub_universe_sharpe = "FAIL" if stability < 0.45 or capacity < 0.32 or sharpe < (required_sharpe - 0.20) else "PASS"
    self_correlation = "FAIL" if uniqueness < 0.35 or exact_match_count > 0 else "PASS"
    matches_competition = "FAIL" if exact_match_count > 0 or skeleton_match_count >= 3 else "PASS"
    out_of_sample_alignment = "FAIL" if out_of_sample_risk >= 0.24 else "PASS"
    preview_verdict = str(surrogate_shadow.get("preview_verdict") or "UNAVAILABLE").upper()
    shadow_status = str(surrogate_shadow.get("status") or "")
    shadow_alignment = str(surrogate_shadow.get("alignment") or "unknown")
    if shadow_status == "ready" and preview_verdict == "FAIL":
        out_of_sample_alignment = "FAIL"
    if surrogate_shadow_summary["hard_signal"] == "severe_mismatch":
        low_sub_universe_sharpe = "FAIL"
    failure_set = {
        name
        for name, value in {
            "LOW_SHARPE": low_sharpe,
            "LOW_FITNESS": low_fitness,
            "LOW_TURNOVER": low_turnover,
            "HIGH_TURNOVER": high_turnover,
            "CONCENTRATED_WEIGHT": concentrated_weight,
            "LOW_SUB_UNIVERSE_SHARPE": low_sub_universe_sharpe,
            "SELF_CORRELATION": self_correlation,
            "MATCHES_COMPETITION": matches_competition,
            "OUT_OF_SAMPLE_ALIGNMENT": out_of_sample_alignment,
        }.items()
        if value == "FAIL"
    }
    core_failures = failure_set & {"LOW_SHARPE", "LOW_FITNESS", "LOW_SUB_UNIVERSE_SHARPE"}
    soft_dimensions = {
        "turnover": low_turnover == "PASS" and high_turnover == "PASS",
        "uniqueness": uniqueness >= 0.45,
        "stability": stability >= 0.48,
        "capacity": capacity >= 0.42,
        "concentration": concentrated_weight == "PASS",
        "out_of_sample_alignment": out_of_sample_alignment == "PASS",
    }
    soft_failure_count = sum(1 for ok in soft_dimensions.values() if not ok)
    if not syntax_ok or not no_lookahead_bias:
        verdict = "FAIL"
    elif (
        not core_failures
        and alpha_score >= 68.0
        and uniqueness >= 0.45
        and soft_failure_count == 0
    ):
        verdict = "PASS"
    elif (
        not core_failures
        and alpha_score >= 60.0
        and uniqueness >= 0.40
        and soft_failure_count <= 1
    ):
        verdict = "LIKELY_PASS"
    elif structural_valid := (syntax_ok and no_lookahead_bias):
        verdict = "BORDERLINE"
    else:
        verdict = "FAIL"

    if verdict == "BORDERLINE" and (core_failures or alpha_score < 40.0):
        verdict = "FAIL"

    if verdict == "LIKELY_PASS" and (failure_set & {"SELF_CORRELATION", "MATCHES_COMPETITION"}) and alpha_score < 66.0:
        verdict = "BORDERLINE"
    if verdict == "PASS" and out_of_sample_alignment == "FAIL":
        verdict = "LIKELY_PASS" if alpha_score >= 74.0 else "BORDERLINE"
    if verdict == "LIKELY_PASS" and out_of_sample_risk >= 0.34:
        verdict = "BORDERLINE"
    if shadow_status == "ready":
        if surrogate_shadow_summary["hard_signal"] == "severe_mismatch":
            verdict = "FAIL"
        elif preview_verdict == "FAIL" and verdict in {"PASS", "LIKELY_PASS"}:
            verdict = "BORDERLINE"
        elif preview_verdict == "BORDERLINE" and verdict == "PASS":
            verdict = "LIKELY_PASS"
        elif shadow_alignment == "more_cautious" and verdict == "PASS":
            verdict = "LIKELY_PASS"

    if verdict == "BORDERLINE" and preview_verdict == "FAIL" and shadow_alignment == "more_cautious" and alpha_score < 62.0:
        verdict = "FAIL"

    confidence = "MEDIUM"
    if not syntax_ok or len(style_tags) < 2:
        confidence = "LOW"
    elif verdict == "PASS" and alpha_score >= 76.0 and soft_failure_count == 0:
        confidence = "HIGH"
    elif verdict == "FAIL" and (len(core_failures) >= 1 or alpha_score < 42.0):
        confidence = "HIGH"
    elif abs(alpha_score - 60.0) <= 4.0 or abs(alpha_score - 68.0) <= 4.0:
        confidence = "MEDIUM"
    if shadow_status == "ready":
        if surrogate_shadow_summary["hard_signal"] == "severe_mismatch":
            confidence = "HIGH" if verdict == "FAIL" else "MEDIUM"
        elif preview_verdict == "FAIL" and confidence == "HIGH":
            confidence = "MEDIUM"

    strengths, weaknesses = _build_strengths_and_weaknesses(
        sharpe=sharpe,
        fitness=fitness,
        turnover=turnover,
        capacity=capacity,
        stability=stability,
        uniqueness=uniqueness,
        concentration_risk=concentration_risk,
        required_sharpe=required_sharpe,
        max_turnover=max_turnover,
    )
    if shadow_status == "ready" and preview_verdict == "FAIL":
        weaknesses.append("Surrogate shadow tu real Brain history dang canh bao FAIL cho expression nay.")
    if surrogate_shadow_summary["hard_signal"] == "severe_mismatch":
        weaknesses.append("Heuristic proxy dang optimistic hon rat nhieu so voi shadow hoc tu ket qua Brain that.")
    suggestions = _build_structured_suggestions(
        alpha_type=alpha_type,
        sharpe=sharpe,
        turnover=turnover,
        uniqueness=uniqueness,
        stability=stability,
        concentration_risk=concentration_risk,
        settings=scoring_settings,
    )
    if shadow_status == "ready" and preview_verdict == "FAIL":
        suggestions.insert(
            0,
            {
                "type": "concept",
                "description": "Dung day candidate nay o submit gate va doi family/thesis khac neu shadow hoc tu Brain du doan FAIL.",
                "example": "Shift sang family chua bi real-fail streak thay vi tiep tuc tune cung mot nhanh.",
            },
        )
        suggestions = suggestions[:4]
    analysis = _build_analysis_text(
        verdict=verdict,
        alpha_type=alpha_type,
        settings=scoring_settings,
        sharpe=sharpe,
        fitness=fitness,
        turnover=turnover,
        stability=stability,
        uniqueness=uniqueness,
    )
    score_breakdown = _build_score_breakdown(
        ic_proxy=ic_proxy,
        sharpe_score=sharpe_score,
        fitness_score=fitness_score,
        turnover_score=turnover_score,
        capacity=capacity,
        stability=stability,
        uniqueness=uniqueness,
        ensemble_proxy=ensemble_proxy,
        validation_score=validation_score,
        turnover_gap_penalty=turnover_gap_penalty,
        complexity_penalty=complexity_penalty,
        duplicate_penalty=duplicate_penalty,
        surrogate_penalty=surrogate_penalty,
        concentration_risk=concentration_risk,
        required_sharpe=required_sharpe,
        max_turnover=max_turnover,
        alpha_score=alpha_score,
    )

    local_alpha_id = f"LOCAL-{hashlib.sha1(normalized.encode('utf-8')).hexdigest()[:8].upper()}"
    result = {
        "alpha_id": local_alpha_id,
        "regular_code": expression,
        "turnover": round(turnover, 4),
        "returns": round(annual_return, 4),
        "drawdown": round(drawdown, 4),
        "margin": margin,
        "fitness": round(fitness, 2),
        "sharpe": round(sharpe, 2),
        "LOW_SHARPE": low_sharpe,
        "LOW_FITNESS": low_fitness,
        "LOW_TURNOVER": low_turnover,
        "HIGH_TURNOVER": high_turnover,
        "CONCENTRATED_WEIGHT": concentrated_weight,
        "LOW_SUB_UNIVERSE_SHARPE": low_sub_universe_sharpe,
        "SELF_CORRELATION": self_correlation,
        "MATCHES_COMPETITION": matches_competition,
        "OUT_OF_SAMPLE_ALIGNMENT": out_of_sample_alignment,
        "expression": expression,
        "score_source": LOCAL_SCORE_VERSION,
        "verdict": verdict,
        "confidence": confidence,
        "alpha_type": alpha_type,
        "alpha_score": round(alpha_score, 1),
        "ic_proxy": round(ic_proxy, 4),
        "capacity_proxy": round(capacity, 4),
        "stability_proxy": round(stability, 4),
        "uniqueness_proxy": round(uniqueness, 4),
        "out_of_sample_risk": round(out_of_sample_risk, 4),
        "quality_proxy": round(quality_proxy, 4),
        "style_tags": style_tags,
        "style_diversification": round(style_diversification, 4),
        "ensemble_proxy": round(ensemble_proxy, 4),
        "optimization_hints": optimization_hints,
        "estimated_metrics": {
            "sharpe": round(sharpe, 2),
            "turnover": round(turnover, 4),
            "returns": round(annual_return, 4),
            "fitness": round(fitness, 2),
            "drawdown": round(drawdown, 4),
        },
        "settings": {
            "region": scoring_settings.region,
            "universe": scoring_settings.universe,
            "delay": scoring_settings.delay,
            "decay": scoring_settings.decay,
            "truncation": round(scoring_settings.truncation, 4),
            "neutralization": scoring_settings.neutralization,
            "label": format_scoring_settings(scoring_settings),
        },
        "strengths": strengths,
        "weaknesses": weaknesses,
        "suggestions": suggestions,
        "analysis": analysis,
        "score_breakdown": score_breakdown,
        "surrogate_shadow": surrogate_shadow,
        "surrogate_shadow_status": surrogate_shadow.get("status"),
        "surrogate_shadow_preview_verdict": surrogate_shadow.get("preview_verdict"),
        "surrogate_shadow_alignment": surrogate_shadow.get("alignment"),
        "surrogate_shadow_penalty": round(surrogate_penalty, 2),
        "surrogate_shadow_reasons": surrogate_shadow_summary["reasons"],
        "surrogate_shadow_hard_signal": surrogate_shadow_summary["hard_signal"],
        "mu_daily_proxy": round(mu_daily, 6),
        "sigma_daily_proxy": round(sigma_daily, 6),
        "validation": {
            "syntax_ok": syntax_ok,
            "no_lookahead_bias": no_lookahead_bias,
        },
    }
    return result


def build_internal_metric(
    logger=None,
    *,
    csv_path: str | Path | None = None,
    settings: str | dict | ScoringSettings | None = None,
):
    history_index = HistoryIndex.from_csv(csv_path)
    scoring_settings = parse_scoring_settings(settings)
    surrogate_bundle = load_or_train_surrogate_shadow(csv_path)

    def metric(expression: str) -> dict:
        result = score_expression(
            expression,
            history_index=history_index,
            settings=scoring_settings,
            surrogate_bundle=surrogate_bundle,
            surrogate_csv_path=csv_path,
        )
        save_alpha_to_csv(result, logger=logger)
        history_index.observe_expression(expression, result)
        if logger is not None:
            logger.log(
                f"Internal score for {expression[:60]}... "
                f"sharpe={result['sharpe']:.2f}, fitness={result['fitness']:.2f}, "
                f"alpha_score={result['alpha_score']:.1f}, verdict={result['verdict']}, "
                f"surrogate={result.get('surrogate_shadow_status')}"
            )
        return result

    metric.requires_session = False
    return metric
