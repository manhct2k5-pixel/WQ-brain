#!/usr/bin/env python3
"""Scout public factor ideas, convert them into local alpha candidates, and rank strict daily picks."""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable

import requests

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.seed_store import load_seed_store as load_shared_seed_store

try:
    from .plan_next_batch import THESIS_INDEX, THESIS_LIBRARY, load_memory, score_style_alignment
    from .results_digest import (
        classify_expression_family,
        coerce_float,
        discover_csv,
        failing_checks,
        passes_all_checks,
        read_rows,
        row_has_pending_checks,
    )
except ImportError:
    from plan_next_batch import THESIS_INDEX, THESIS_LIBRARY, load_memory, score_style_alignment
    from results_digest import (
        classify_expression_family,
        coerce_float,
        discover_csv,
        failing_checks,
        passes_all_checks,
        read_rows,
        row_has_pending_checks,
    )

from src.internal_scoring import HistoryIndex, score_expression, score_expressions_batch
from src.program_tokens import render_token_program, validate_token_program
from src.brain import migrate_results_csv_context
from src.http_utils import compute_backoff_delay, is_retryable_http_status, parse_retry_after
from scripts.flow_utils import atomic_write_json as shared_atomic_write_json
from scripts.flow_utils import atomic_write_text as shared_atomic_write_text
from scripts.flow_utils import load_json as shared_load_json
from scripts.lineage_utils import ensure_candidate_lineage

OPENALEX_URL = "https://api.openalex.org/works"
ARXIV_URL = "https://export.arxiv.org/api/query"
GITHUB_SEARCH_URL = "https://api.github.com/search/repositories"
GITHUB_README_URL = "https://api.github.com/repos/{repo}/readme"
DEFAULT_OUTPUT = Path("artifacts/trinh_sat_hang_ngay.md")
DEFAULT_PLAN = Path("artifacts/_trinh_sat/du_lieu.json")
DEFAULT_BATCH = Path("artifacts/_trinh_sat/bieu_thuc_da_chon.txt")
DEFAULT_HISTORY = Path("artifacts/_trinh_sat/lich_su.jsonl")
DEFAULT_MEMORY = Path("artifacts/_trinh_sat/bo_nho.json")
DEFAULT_SUBMITTED_ALPHAS = Path("artifacts/alpha_da_gui.json")
DEFAULT_BRAIN_FEEDBACK = Path("artifacts/_trinh_sat/phan_hoi_brain.json")
DEFAULT_SELF_LEARNING = Path("artifacts/_trinh_sat/kien_thuc_tu_hoc.json")
DEFAULT_FETCH_CACHE = Path("artifacts/_trinh_sat/cache_tim_kiem.json")
DEFAULT_REPORT_STATE = Path("artifacts/_trinh_sat/trang_thai_bao_cao.json")
DEFAULT_REPORT_ARCHIVE_ROOT = Path("artifacts/bao_cao_ngay")
DEFAULT_DIVERSITY_WEIGHT = 0.35
DEFAULT_ARCHIVE_FREQUENCY = "hour"
DEFAULT_REPORT_INTERVAL_MINUTES = 60
DEFAULT_PUBLIC_API_TIMEOUT = 20
DEFAULT_PUBLIC_API_MAX_RETRIES = 3
DEFAULT_PUBLIC_API_REQUEST_DELAY_SECONDS = 0.35
DEFAULT_PUBLIC_API_MAX_REQUESTS_PER_RUN = 48
DEFAULT_PUBLIC_API_QUOTA_COOLDOWN_SECONDS = 30.0
DEFAULT_PUBLIC_API_QUOTA_COOLDOWN_THRESHOLD = 2
DEFAULT_REPORT_MIN_ALPHA_SCORE = 76.0
DEFAULT_REPORT_MIN_CONFIDENCE_SCORE = 0.64
DEFAULT_REPORT_MIN_ROBUSTNESS_SCORE = 0.62
REPORT_OVERFLOW_SELECTION_BUFFER = 0.015
RELAXED_FILL_RATIO = 0.5
RELAXED_ALPHA_BUFFER = 4.0
RELAXED_CONFIDENCE_BUFFER = 0.05
RELAXED_BRAIN_FEEDBACK_CAP = 0.38
SURROGATE_SHADOW_MAX_PENALTY = 0.18
SURROGATE_SHADOW_MAX_CONFIDENCE_DRAG = 0.14
WATCHLIST_ALPHA_BUFFER = 8.0
WATCHLIST_CONFIDENCE_BUFFER = 0.14
WATCHLIST_BRAIN_FEEDBACK_CAP = 0.42
BRAIN_CONTEXT_FIELDS = ("region", "universe", "delay", "decay", "neutralization", "truncation")
DEFAULT_GITHUB_QUERY_LIMIT = 6
DEFAULT_GITHUB_README_LIMIT = 4
DEFAULT_ZIP_SEED_LIMIT = 8
DEFAULT_WORLDQUANT_MINER_ZIP = Path("/mnt/c/Users/OS/Downloads/worldquant-miner-master.zip")
FETCH_CACHE_TTLS = {
    "openalex": 6 * 3600,
    "arxiv": 6 * 3600,
    "github": 6 * 3600,
    "github_readme": 24 * 3600,
}
REACTIVE_FAMILIES = {
    "technical_indicator",
    "reversal_conditioned",
    "pv_divergence",
    "vwap_dislocation",
    "simple_price_patterns",
}
CONTRAST_FAMILIES = {"residual_beta", "shock_response"}

BASE_QUERY_PROFILES = [
    {
        "query": "equity alpha simple price hypothesis rank decay",
        "families": ["simple_price_patterns", "reversal_conditioned"],
        "style_tags": ["simple", "ratio_like", "rank", "book_alpha_design"],
        "horizon": "short",
    },
    {
        "query": "equity factor momentum anomaly",
        "families": ["technical_indicator"],
        "style_tags": ["momentum", "trend", "technical"],
        "horizon": "long",
    },
    {
        "query": "short term mean reversion equities",
        "families": ["reversal_conditioned", "vwap_dislocation"],
        "style_tags": ["reversal", "vwap", "liquidity"],
        "horizon": "short",
    },
    {
        "query": "price volume anomaly equities",
        "families": ["pv_divergence", "shock_response"],
        "style_tags": ["volume", "correlation", "liquidity"],
        "horizon": "medium",
    },
    {
        "query": "volatility shock equity anomaly",
        "families": ["shock_response", "technical_indicator"],
        "style_tags": ["volatility", "technical"],
        "horizon": "medium",
    },
    {
        "query": "residual beta equity anomaly",
        "families": ["residual_beta"],
        "style_tags": ["residual", "beta", "correlation"],
        "horizon": "long",
    },
    {
        "query": "intraday vwap dislocation stocks",
        "families": ["vwap_dislocation", "reversal_conditioned"],
        "style_tags": ["vwap", "liquidity", "reversal"],
        "horizon": "short",
    },
    {
        "query": "ratio like stock predictor cross sectional rank",
        "families": ["simple_price_patterns", "technical_indicator"],
        "style_tags": ["ratio_like", "normalization", "rank", "book_alpha_design"],
        "horizon": "medium",
    },
]

EXTENDED_QUERY_PROFILES = [
    {
        "query": "stock contrarian overreaction underreaction anomaly",
        "families": ["reversal_conditioned", "vwap_dislocation"],
        "style_tags": ["reversal", "technical"],
        "horizon": "short",
    },
    {
        "query": "lagged return reversal stock alpha cross sectional",
        "families": ["simple_price_patterns", "reversal_conditioned"],
        "style_tags": ["simple", "reversal", "ratio_like"],
        "horizon": "short",
    },
    {
        "query": "order flow liquidity imbalance stock returns",
        "families": ["pv_divergence", "shock_response"],
        "style_tags": ["volume", "liquidity", "correlation"],
        "horizon": "medium",
    },
    {
        "query": "trading volume shock cross sectional stock anomaly",
        "families": ["pv_divergence", "shock_response"],
        "style_tags": ["volume", "volatility", "liquidity"],
        "horizon": "medium",
    },
    {
        "query": "intraday price pressure liquidity provision equities",
        "families": ["vwap_dislocation", "pv_divergence"],
        "style_tags": ["vwap", "liquidity", "volume"],
        "horizon": "short",
    },
    {
        "query": "idiosyncratic volatility stock anomaly market neutral",
        "families": ["residual_beta", "shock_response"],
        "style_tags": ["residual", "beta", "volatility"],
        "horizon": "long",
    },
    {
        "query": "market neutral residual return stock selection anomaly",
        "families": ["residual_beta"],
        "style_tags": ["residual", "beta", "correlation"],
        "horizon": "long",
    },
    {
        "query": "price delay information diffusion stock returns",
        "families": ["simple_price_patterns", "technical_indicator"],
        "style_tags": ["simple", "ratio_like", "trend"],
        "horizon": "medium",
    },
    {
        "query": "cross sectional volatility dispersion stock returns",
        "families": ["shock_response", "technical_indicator"],
        "style_tags": ["volatility", "technical", "normalization"],
        "horizon": "medium",
    },
    {
        "query": "stock correlation structure anomaly market neutral alpha",
        "families": ["residual_beta", "pv_divergence"],
        "style_tags": ["correlation", "residual", "beta"],
        "horizon": "long",
    },
    {
        "query": "abnormal turnover stock reversal anomaly",
        "families": ["pv_divergence", "reversal_conditioned"],
        "style_tags": ["volume", "reversal", "liquidity"],
        "horizon": "short",
    },
    {
        "query": "high low close range intraday reversal stocks",
        "families": ["vwap_dislocation", "technical_indicator"],
        "style_tags": ["vwap", "reversal", "technical"],
        "horizon": "short",
    },
]

EXPLORATION_QUERY_PROFILES = [
    {
        "query": "closing location value stock reversal anomaly",
        "families": ["technical_indicator", "vwap_dislocation"],
        "style_tags": ["technical", "reversal", "vwap"],
        "horizon": "short",
    },
    {
        "query": "range position stock alpha high low close",
        "families": ["technical_indicator", "reversal_conditioned"],
        "style_tags": ["technical", "reversal", "ratio_like"],
        "horizon": "short",
    },
    {
        "query": "volume acceleration stock drift anomaly",
        "families": ["pv_divergence", "technical_indicator"],
        "style_tags": ["volume", "momentum", "technical"],
        "horizon": "medium",
    },
    {
        "query": "liquidity vacuum mean reversion stock anomaly",
        "families": ["vwap_dislocation", "shock_response"],
        "style_tags": ["liquidity", "reversal", "volatility"],
        "horizon": "short",
    },
    {
        "query": "cross sectional price range compression stocks",
        "families": ["technical_indicator", "shock_response"],
        "style_tags": ["technical", "volatility", "normalization"],
        "horizon": "medium",
    },
    {
        "query": "abnormal trading activity stock continuation anomaly",
        "families": ["pv_divergence", "shock_response"],
        "style_tags": ["volume", "liquidity", "momentum"],
        "horizon": "medium",
    },
    {
        "query": "market neutral beta dislocation stock selection",
        "families": ["residual_beta", "technical_indicator"],
        "style_tags": ["residual", "beta", "technical"],
        "horizon": "long",
    },
    {
        "query": "idiosyncratic reversal stock anomaly market neutral",
        "families": ["residual_beta", "reversal_conditioned"],
        "style_tags": ["residual", "reversal", "beta"],
        "horizon": "medium",
    },
    {
        "query": "intraday dislocation closing pressure equities",
        "families": ["vwap_dislocation", "pv_divergence"],
        "style_tags": ["vwap", "liquidity", "volume"],
        "horizon": "short",
    },
    {
        "query": "ranked historical ratio stock anomaly",
        "families": ["simple_price_patterns", "technical_indicator"],
        "style_tags": ["simple", "ratio_like", "rank"],
        "horizon": "medium",
    },
    {
        "query": "cross sectional standardized stock alpha expression",
        "families": ["simple_price_patterns", "technical_indicator"],
        "style_tags": ["normalization", "rank", "simple"],
        "horizon": "medium",
    },
    {
        "query": "short horizon liquidity reversal stock factor",
        "families": ["reversal_conditioned", "pv_divergence"],
        "style_tags": ["reversal", "liquidity", "volume"],
        "horizon": "short",
    },
]

STYLE_QUERY_EXPANSIONS = {
    "momentum": [
        {"query": "cross sectional trend continuation stock anomaly", "families": ["technical_indicator"], "style_tags": ["momentum", "trend", "technical"], "horizon": "long"},
        {"query": "relative strength equity factor stock returns", "families": ["technical_indicator"], "style_tags": ["momentum", "trend"], "horizon": "long"},
    ],
    "reversal": [
        {"query": "short horizon stock contrarian reversal factor", "families": ["reversal_conditioned", "vwap_dislocation"], "style_tags": ["reversal"], "horizon": "short"},
        {"query": "temporary mispricing stock mean reversion anomaly", "families": ["reversal_conditioned"], "style_tags": ["reversal", "technical"], "horizon": "short"},
    ],
    "volume": [
        {"query": "equity liquidity shock trading activity anomaly", "families": ["pv_divergence", "shock_response"], "style_tags": ["volume", "liquidity"], "horizon": "medium"},
        {"query": "turnover and volume conditioned stock returns", "families": ["pv_divergence"], "style_tags": ["volume", "liquidity", "correlation"], "horizon": "medium"},
    ],
    "residual": [
        {"query": "idiosyncratic risk equity alpha anomaly", "families": ["residual_beta"], "style_tags": ["residual", "beta"], "horizon": "long"},
        {"query": "market neutral residual stock selection factor", "families": ["residual_beta"], "style_tags": ["residual", "beta", "correlation"], "horizon": "long"},
    ],
    "vwap": [
        {"query": "intraday vwap deviation stock anomaly", "families": ["vwap_dislocation"], "style_tags": ["vwap", "liquidity"], "horizon": "short"},
        {"query": "microstructure dislocation price pressure stocks", "families": ["vwap_dislocation", "pv_divergence"], "style_tags": ["vwap", "liquidity", "volume"], "horizon": "short"},
    ],
    "volatility": [
        {"query": "volatility regime equity anomaly stock returns", "families": ["shock_response", "technical_indicator"], "style_tags": ["volatility"], "horizon": "medium"},
        {"query": "dispersion shock stock cross sectional anomaly", "families": ["shock_response"], "style_tags": ["volatility", "technical"], "horizon": "medium"},
    ],
    "ratio_like": [
        {"query": "ratio based stock predictor cross sectional alpha", "families": ["simple_price_patterns", "technical_indicator"], "style_tags": ["ratio_like", "normalization"], "horizon": "medium"},
        {"query": "historical value comparison stock alpha signal", "families": ["simple_price_patterns"], "style_tags": ["simple", "ratio_like"], "horizon": "medium"},
    ],
    "simple": [
        {"query": "simple stock alpha expression rank decay", "families": ["simple_price_patterns"], "style_tags": ["simple", "book_alpha_design"], "horizon": "short"},
        {"query": "price delay simple hypothesis equity anomaly", "families": ["simple_price_patterns", "reversal_conditioned"], "style_tags": ["simple", "ratio_like", "book_alpha_design"], "horizon": "short"},
    ],
    "correlation": [
        {"query": "stock co movement anomaly cross sectional alpha", "families": ["pv_divergence", "residual_beta"], "style_tags": ["correlation"], "horizon": "medium"},
        {"query": "equity correlation structure anomaly", "families": ["residual_beta", "pv_divergence"], "style_tags": ["correlation", "beta"], "horizon": "long"},
    ],
}

FAMILY_QUERY_EXPANSIONS = {
    "simple_price_patterns": [
        "price delay ratio stock anomaly",
        "simple ranked price history stock factor",
    ],
    "technical_indicator": [
        "trend persistence stock anomaly cross sectional",
        "technical stock indicator alpha anomaly",
    ],
    "reversal_conditioned": [
        "short term contrarian stock anomaly",
        "mean reversion conditioned stock returns",
    ],
    "vwap_dislocation": [
        "intraday dislocation stock mean reversion",
        "vwap gap stock reversion anomaly",
    ],
    "pv_divergence": [
        "price volume divergence stock anomaly",
        "liquidity conditioned stock alpha signal",
    ],
    "shock_response": [
        "volatility shock response stock anomaly",
        "risk regime stock response alpha",
    ],
    "residual_beta": [
        "residual beta stock selection anomaly",
        "idiosyncratic beta market neutral stock alpha",
    ],
}

SEARCH_BREADTH_LIMITS = {
    "focused": 10,
    "standard": 18,
    "wide": 30,
    "explore": 42,
}

BOOK_IDEA_BLUEPRINTS = [
    {
        "title": "Finding Alphas: short-term reversion ladder",
        "summary": "Start from a simple 5-day price-delay hypothesis, then neutralize, rank cross-sectionally, and use decay to reduce turnover.",
        "bias_families": ["simple_price_patterns", "reversal_conditioned"],
        "bias_style_tags": ["simple", "reversal", "rank", "ratio_like", "book_alpha_design"],
        "bias_horizon": "short",
    },
    {
        "title": "Finding Alphas: trend with volume confirmation",
        "summary": "Trending stocks with increasing volume can outperform; use a ratio-like price-delay expression confirmed by ranked volume.",
        "bias_families": ["simple_price_patterns", "pv_divergence"],
        "bias_style_tags": ["simple", "trend", "volume", "ratio_like", "book_alpha_design"],
        "bias_horizon": "medium",
    },
    {
        "title": "Finding Alphas: ratio-like predictors",
        "summary": "Turn raw price or volume into ratio-like variables by comparing current values to historical values before ranking.",
        "bias_families": ["simple_price_patterns", "technical_indicator"],
        "bias_style_tags": ["simple", "ratio_like", "normalization", "book_alpha_design"],
        "bias_horizon": "medium",
    },
    {
        "title": "Finding Alphas: robust ranking and winsorization",
        "summary": "Improve robustness with ranking, z-scoring, winsorization, and truncation instead of overfitting constants.",
        "bias_families": ["simple_price_patterns", "technical_indicator"],
        "bias_style_tags": ["simple", "rank", "normalization", "winsorize", "book_alpha_design"],
        "bias_horizon": "medium",
    },
]

WINDOW_GROUPS = {
    "CORR": {"short": "CORR_10", "medium": "CORR_21", "long": "CORR_63"},
    "RCORR": {"short": "RCORR_10", "medium": "RCORR_21", "long": "RCORR_63"},
    "STD": {"short": "STD_10", "medium": "STD_21", "long": "STD_63"},
    "TSR": {"short": "TSR_10", "medium": "TSR_21", "long": "TSR_63"},
    "TSS": {"short": "TSS_10", "medium": "TSS_21", "long": "TSS_63"},
    "SURPRISE": {"short": "SURPRISE_10", "medium": "SURPRISE_21", "long": "SURPRISE_63"},
    "TSZ": {"short": "TSZ_21", "medium": "TSZ_21", "long": "TSZ_63"},
}

FAMILY_SETTINGS_BASE = {
    "technical_indicator": ("USA", "TOP3000", 1, 6, 0.05, "Industry"),
    "reversal_conditioned": ("USA", "TOP1000", 1, 3, 0.03, "Subindustry"),
    "simple_price_patterns": ("USA", "TOP1000", 1, 4, 0.03, "Industry"),
    "vwap_dislocation": ("USA", "TOP200", 1, 3, 0.02, "Subindustry"),
    "pv_divergence": ("USA", "TOP1000", 1, 5, 0.04, "Industry"),
    "shock_response": ("USA", "TOP1000", 1, 6, 0.04, "Industry"),
    "residual_beta": ("USA", "TOP3000", 1, 7, 0.06, "Industry"),
}


def normalize_text(value: str) -> str:
    return " ".join((value or "").lower().split())


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _normalize_doi(value: str) -> str:
    text = normalize_text(value)
    text = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", text)
    return text.strip("/")


def _extract_openalex_id(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    match = re.search(r"(W\d+)", text)
    return match.group(1) if match else text.rsplit("/", 1)[-1]


def _extract_arxiv_id(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    match = re.search(r"arxiv\.org/(?:abs|pdf)/([^/?#]+)", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).removesuffix(".pdf")
    return text.rsplit("/", 1)[-1].removesuffix(".pdf")


def _extract_github_repo_name(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    match = re.search(r"github\.com/([^/?#]+/[^/?#]+)", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip("/")
    if "/" in text and " " not in text:
        return text.strip("/")
    return ""


def source_key_from_idea(idea: dict) -> str:
    doi = _normalize_doi(str(idea.get("doi") or ""))
    if doi:
        return f"doi:{doi}"

    openalex_source = idea.get("openalex_id") or (idea.get("url") if idea.get("source") == "openalex" else "")
    openalex_id = _extract_openalex_id(openalex_source)
    if openalex_id:
        return f"oax:{openalex_id}"

    arxiv_source = idea.get("arxiv_id") or (idea.get("url") if idea.get("source") == "arxiv" else "")
    arxiv_id = _extract_arxiv_id(arxiv_source)
    if arxiv_id:
        return f"arxiv:{arxiv_id}"

    github_source = idea.get("github_full_name") or (idea.get("url") if idea.get("source") == "github" else "")
    github_repo = _extract_github_repo_name(github_source)
    if github_repo:
        prefix = "ghreadme" if idea.get("source") == "github_readme" else "gh"
        return f"{prefix}:{normalize_text(github_repo)}"

    title = normalize_text(idea.get("title") or "")
    if title:
        return f"title:{title}"

    query = normalize_text(idea.get("query") or "")
    return f"query:{query}" if query else ""


def _normalize_brain_context_value(field: str, value) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "nan", "null", "not found"}:
        return None
    if field in {"region", "universe"}:
        return text.upper().replace(" ", "")
    if field in {"delay", "decay"}:
        try:
            return str(int(float(text)))
        except ValueError:
            return None
    if field == "truncation":
        try:
            return f"{float(text):.4f}"
        except ValueError:
            return None
    if field == "neutralization":
        lowered = text.lower().replace(" ", "")
        mapping = {
            "subindustry": "Subindustry",
            "industry": "Industry",
            "sector": "Sector",
            "market": "Market",
            "none": "None",
        }
        return mapping.get(lowered, text.title())
    return normalize_text(text)


def extract_brain_context(source: dict | None) -> dict:
    if not isinstance(source, dict):
        return {}
    context = {}
    settings_source = source.get("settings") if isinstance(source.get("settings"), dict) else source
    for field in BRAIN_CONTEXT_FIELDS:
        normalized = _normalize_brain_context_value(field, settings_source.get(field))
        if normalized is not None:
            context[field] = normalized
    return context


def format_brain_context(context: dict) -> str:
    if not context:
        return "global"
    ordered = [f"{field}={context[field]}" for field in BRAIN_CONTEXT_FIELDS if field in context]
    return ", ".join(ordered) if ordered else "global"


def _brain_context_matches(row_context: dict, candidate_context: dict) -> bool:
    if not row_context or not candidate_context:
        return False
    return all(candidate_context.get(field) == value for field, value in row_context.items())


def expression_skeleton(expression: str) -> str:
    return re.sub(r"\d+(?:\.\d+)?", "N", re.sub(r"\s+", "", expression or ""))


def _round(value, digits: int = 4):
    return round(float(value), digits)


def _empty_stat() -> dict:
    return {
        "attempts": 0,
        "selected_count": 0,
        "watchlist_count": 0,
        "strong_count": 0,
        "avg_alpha_score": 0.0,
        "avg_confidence_score": 0.0,
        "avg_source_quality": 0.0,
    }


def _update_running_average(current: float, count: int, value: float) -> float:
    return ((current * count) + value) / (count + 1)


def load_json(path: str | Path | None) -> dict:
    file_path = Path(path) if path else None
    label = str(file_path) if file_path else "scout_ideas payload"
    return shared_load_json(
        path,
        default={},
        context=label,
        warn=lambda message: print(f"[scout-ideas] Warning: {message}", file=sys.stderr),
    )


def _atomic_write_text(path: str | Path | None, content: str) -> None:
    if not path:
        return
    shared_atomic_write_text(path, content)


def _atomic_write_json(path: str | Path | None, payload: dict) -> None:
    if not path:
        return
    shared_atomic_write_json(path, payload)


def save_json(path: str | Path | None, payload: dict) -> None:
    if not path:
        return
    _atomic_write_json(path, payload)


def _parse_iso_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def assess_report_publish_status(
    run_timestamp: datetime,
    *,
    reportable_count: int,
    report_interval_minutes: int,
    report_state: dict | None = None,
) -> dict:
    if not isinstance(report_state, dict):
        report_state = {}
    interval_minutes = max(0, int(report_interval_minutes))
    last_published_at = _parse_iso_datetime(str(report_state.get("last_published_at") or ""))
    status = {
        "published": False,
        "reason": "",
        "interval_minutes": interval_minutes,
        "last_published_at": last_published_at.isoformat(timespec="seconds") if last_published_at else "",
        "next_publish_after": "",
        "minutes_until_next": 0,
    }
    if reportable_count <= 0:
        status["reason"] = "no_reportable_pick"
        return status
    if interval_minutes <= 0 or last_published_at is None:
        status["published"] = True
        status["reason"] = "reportable_pick_found"
        return status

    next_publish_at = last_published_at + timedelta(minutes=interval_minutes)
    if run_timestamp >= next_publish_at:
        status["published"] = True
        status["reason"] = "reportable_pick_found"
        status["next_publish_after"] = next_publish_at.isoformat(timespec="seconds")
        return status

    remaining_seconds = max(0, int((next_publish_at - run_timestamp).total_seconds()))
    remaining_minutes = (remaining_seconds + 59) // 60 if remaining_seconds else 0
    status["reason"] = "report_interval_not_elapsed"
    status["next_publish_after"] = next_publish_at.isoformat(timespec="seconds")
    status["minutes_until_next"] = remaining_minutes
    return status


def build_report_state(run_timestamp: datetime, *, report_status: dict) -> dict:
    return {
        "last_published_at": run_timestamp.isoformat(timespec="seconds"),
        "interval_minutes": int(report_status.get("interval_minutes", DEFAULT_REPORT_INTERVAL_MINUTES) or DEFAULT_REPORT_INTERVAL_MINUTES),
        "reason": report_status.get("reason", ""),
    }


def load_fetch_cache(path: str | Path | None) -> dict:
    payload = load_json(path)
    if not isinstance(payload, dict):
        return {"entries": {}}
    entries = payload.get("entries")
    if not isinstance(entries, dict):
        payload["entries"] = {}
    return payload


def save_fetch_cache(path: str | Path | None, cache: dict) -> None:
    if not path:
        return
    _atomic_write_json(path, cache)


class ScoutRequestThrottle:
    def __init__(
        self,
        *,
        max_requests: int = DEFAULT_PUBLIC_API_MAX_REQUESTS_PER_RUN,
        request_delay_seconds: float = DEFAULT_PUBLIC_API_REQUEST_DELAY_SECONDS,
        quota_cooldown_seconds: float = DEFAULT_PUBLIC_API_QUOTA_COOLDOWN_SECONDS,
        quota_cooldown_threshold: int = DEFAULT_PUBLIC_API_QUOTA_COOLDOWN_THRESHOLD,
    ) -> None:
        self.max_requests = max(0, int(max_requests))
        self.request_delay_seconds = max(0.0, float(request_delay_seconds))
        self.quota_cooldown_seconds = max(0.0, float(quota_cooldown_seconds))
        self.quota_cooldown_threshold = max(1, int(quota_cooldown_threshold))
        self.requests_sent = 0
        self.last_request_at = 0.0
        self.next_allowed_at = 0.0
        self.consecutive_quota_errors = 0
        self.cooldowns_triggered = 0

    def acquire(self, status: dict | None = None) -> bool:
        if self.requests_sent >= self.max_requests:
            if isinstance(status, dict):
                status["skipped_budget"] = int(status.get("skipped_budget", 0) or 0) + 1
                status["budget_exhausted"] = True
            return False

        now_ts = time.monotonic()
        wait_until = self.next_allowed_at
        if self.last_request_at > 0 and self.request_delay_seconds > 0:
            wait_until = max(wait_until, self.last_request_at + self.request_delay_seconds)
        remaining = wait_until - now_ts
        if remaining > 0:
            time.sleep(remaining)

        self.requests_sent += 1
        self.last_request_at = time.monotonic()
        if isinstance(status, dict):
            status["network_requests"] = int(status.get("network_requests", 0) or 0) + 1
        return True

    def defer(self, delay_seconds: float) -> None:
        delay = max(0.0, float(delay_seconds))
        self.next_allowed_at = max(self.next_allowed_at, time.monotonic() + delay)

    def record_success(self) -> None:
        self.consecutive_quota_errors = 0

    def record_non_quota_error(self) -> None:
        self.consecutive_quota_errors = 0

    def record_quota_error(self, retry_after: float | None = None) -> float:
        self.consecutive_quota_errors += 1
        wait_time = max(0.0, float(retry_after or 0.0))
        if self.consecutive_quota_errors >= self.quota_cooldown_threshold:
            wait_time = max(wait_time, self.quota_cooldown_seconds)
            self.cooldowns_triggered += 1
        return wait_time


def _scout_warn(message: str) -> None:
    print(f"[scout] Warning: {message}")


def _fetch_url_with_policy(
    url: str,
    *,
    params: dict | None = None,
    headers: dict | None = None,
    timeout: int = DEFAULT_PUBLIC_API_TIMEOUT,
    context: str,
    status: dict,
    throttle: ScoutRequestThrottle | None = None,
    max_retries: int = DEFAULT_PUBLIC_API_MAX_RETRIES,
):
    response = None
    for attempt in range(1, max(1, int(max_retries)) + 1):
        if throttle is not None and not throttle.acquire(status):
            return None
        try:
            response = requests.get(url, params=params, headers=headers, timeout=timeout)
        except requests.exceptions.Timeout as exc:
            if attempt >= max_retries:
                _scout_warn(f"{context} timed out after {attempt} attempts: {exc}")
                return None
            wait_time = compute_backoff_delay(attempt, base_delay=1.0, max_delay=20.0, jitter_ratio=0.15)
            status["retries"] = int(status.get("retries", 0) or 0) + 1
            if throttle is not None:
                throttle.record_non_quota_error()
                throttle.defer(wait_time)
            else:
                time.sleep(wait_time)
            continue
        except requests.RequestException as exc:
            if attempt >= max_retries:
                _scout_warn(f"{context} failed after {attempt} attempts: {exc}")
                return None
            wait_time = compute_backoff_delay(attempt, base_delay=1.0, max_delay=20.0, jitter_ratio=0.15)
            status["retries"] = int(status.get("retries", 0) or 0) + 1
            if throttle is not None:
                throttle.record_non_quota_error()
                throttle.defer(wait_time)
            else:
                time.sleep(wait_time)
            continue

        if 200 <= response.status_code < 300:
            if throttle is not None:
                throttle.record_success()
                status["cooldowns"] = int(status.get("cooldowns", 0) or 0) + int(throttle.cooldowns_triggered)
                throttle.cooldowns_triggered = 0
            return response

        if response.status_code == 429:
            retry_after = parse_retry_after(response.headers.get("Retry-After"), minimum=1.0)
            wait_time = compute_backoff_delay(
                attempt,
                retry_after=retry_after,
                base_delay=1.0,
                max_delay=30.0,
                jitter_ratio=0.15,
            )
            status["quota_limited"] = int(status.get("quota_limited", 0) or 0) + 1
            if throttle is not None:
                wait_time = max(wait_time, throttle.record_quota_error(retry_after))
                status["cooldowns"] = int(status.get("cooldowns", 0) or 0) + int(throttle.cooldowns_triggered)
                throttle.cooldowns_triggered = 0
            if attempt >= max_retries:
                return response
            status["retries"] = int(status.get("retries", 0) or 0) + 1
            if throttle is not None:
                throttle.defer(wait_time)
            else:
                time.sleep(wait_time)
            continue

        if is_retryable_http_status(response.status_code):
            if attempt >= max_retries:
                return response
            wait_time = compute_backoff_delay(attempt, base_delay=1.0, max_delay=20.0, jitter_ratio=0.15)
            status["server_errors"] = int(status.get("server_errors", 0) or 0) + 1
            status["retries"] = int(status.get("retries", 0) or 0) + 1
            if throttle is not None:
                throttle.record_non_quota_error()
                throttle.defer(wait_time)
            else:
                time.sleep(wait_time)
            continue

        if throttle is not None:
            throttle.record_non_quota_error()
        return response

    return response


def _cache_key(namespace: str, key: str) -> str:
    return f"{namespace}:{key}"


def get_cached_fetch(cache: dict, *, namespace: str, key: str, ttl_seconds: int, now_ts: float | None = None):
    entries = cache.get("entries", {})
    entry = entries.get(_cache_key(namespace, key))
    if not isinstance(entry, dict):
        return None
    created_at = float(entry.get("created_at", 0.0) or 0.0)
    now_ts = now_ts if now_ts is not None else time.time()
    if now_ts - created_at > ttl_seconds:
        return None
    return entry.get("payload")


def put_cached_fetch(cache: dict, *, namespace: str, key: str, payload, now_ts: float | None = None) -> None:
    entries = cache.setdefault("entries", {})
    entries[_cache_key(namespace, key)] = {
        "created_at": now_ts if now_ts is not None else time.time(),
        "payload": payload,
    }


def build_report_archive_paths(
    run_timestamp: datetime,
    *,
    archive_root: str | Path = DEFAULT_REPORT_ARCHIVE_ROOT,
    archive_frequency: str = DEFAULT_ARCHIVE_FREQUENCY,
) -> dict[str, Path]:
    archive_root = Path(archive_root)
    day_dir = archive_root / run_timestamp.strftime("%Y-%m-%d")
    if archive_frequency == "run":
        run_dir = day_dir / run_timestamp.strftime("%Hh_%Mp")
    else:
        run_dir = day_dir / run_timestamp.strftime("%Hh")
    return {
        "root": run_dir,
        "markdown": run_dir / "trinh_sat_hang_ngay.md",
        "payload": run_dir / "du_lieu.json",
        "batch": run_dir / "bieu_thuc_da_chon.txt",
        "memory": run_dir / "bo_nho.json",
        "brain_feedback": run_dir / "phan_hoi_brain.json",
        "self_learning": run_dir / "kien_thuc_tu_hoc.json",
    }


def write_report_archive(
    *,
    run_timestamp: datetime,
    markdown: str,
    payload: dict,
    selected_batch_text: str,
    scout_memory: dict,
    brain_feedback_context: dict,
    self_learning_payload: dict,
    archive_root: str | Path = DEFAULT_REPORT_ARCHIVE_ROOT,
    archive_frequency: str = DEFAULT_ARCHIVE_FREQUENCY,
) -> dict[str, str]:
    paths = build_report_archive_paths(
        run_timestamp,
        archive_root=archive_root,
        archive_frequency=archive_frequency,
    )
    paths["root"].mkdir(parents=True, exist_ok=True)
    _atomic_write_text(paths["markdown"], markdown)
    _atomic_write_json(paths["payload"], payload)
    _atomic_write_text(paths["batch"], selected_batch_text)
    _atomic_write_json(paths["memory"], scout_memory)
    _atomic_write_json(paths["brain_feedback"], brain_feedback_context)
    _atomic_write_json(paths["self_learning"], self_learning_payload)
    return {name: str(path) for name, path in paths.items()}


def load_submitted_alphas(path: str | Path | None) -> list[dict]:
    if not path:
        return []
    file_path = Path(path)
    if not file_path.exists():
        return []
    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, list):
        return []

    submitted = []
    for index, item in enumerate(payload, start=1):
        if isinstance(item, str):
            expression = item.strip()
            record = {"expression": expression}
        elif isinstance(item, dict):
            expression = str(item.get("expression", "")).strip()
            record = dict(item)
        else:
            continue
        if not expression:
            continue
        submitted.append(
            {
                "id": record.get("id") or f"submitted_{index}",
                "label": record.get("label") or f"submitted_{index}",
                "expression": expression,
                "settings": record.get("settings"),
                "notes": record.get("notes", ""),
                "source": record.get("source", "manual"),
                "confidence": record.get("confidence", "high"),
            }
        )
    return submitted


def _is_reportable_candidate(
    candidate: dict,
    *,
    report_min_alpha_score: float,
    report_min_confidence_score: float,
    report_min_robustness_score: float,
) -> bool:
    if candidate.get("quality_fail_reasons"):
        return False
    local = candidate.get("local_metrics", {})
    if local.get("verdict") not in {"PASS", "LIKELY_PASS"}:
        return False
    if float(local.get("alpha_score", 0.0) or 0.0) < report_min_alpha_score:
        return False
    if float(candidate.get("confidence_score", 0.0) or 0.0) < report_min_confidence_score:
        return False
    if float(candidate.get("robustness_score", 0.0) or 0.0) < report_min_robustness_score:
        return False
    return True


def select_reportable_candidates(
    candidates: list[dict],
    *,
    target_count: int,
    report_min_alpha_score: float,
    report_min_confidence_score: float,
    report_min_robustness_score: float,
) -> list[dict]:
    ranked_reportable = []
    for candidate in candidates:
        candidate["reportable"] = _is_reportable_candidate(
            candidate,
            report_min_alpha_score=report_min_alpha_score,
            report_min_confidence_score=report_min_confidence_score,
            report_min_robustness_score=report_min_robustness_score,
        )
        if candidate["reportable"]:
            ranked_reportable.append(candidate)

    ranked_reportable.sort(
        key=lambda item: (
            float(item.get("selection_rank_score", 0.0)),
            float(item.get("candidate_score", 0.0)),
            float(item.get("robustness_score", 0.0)),
            float(item.get("confidence_score", 0.0)),
            float(item.get("local_metrics", {}).get("alpha_score", 0.0)),
        ),
        reverse=True,
    )
    if target_count <= 0 or len(ranked_reportable) <= target_count:
        return ranked_reportable

    cutoff_score = float(ranked_reportable[target_count - 1].get("selection_rank_score", 0.0))
    reportable = list(ranked_reportable[:target_count])
    for candidate in ranked_reportable[target_count:]:
        if float(candidate.get("selection_rank_score", 0.0)) >= cutoff_score - REPORT_OVERFLOW_SELECTION_BUFFER:
            reportable.append(candidate)
            continue
        break
    return reportable


def load_seed_store(path: str | Path | None) -> dict:
    if not path:
        return {}
    return load_shared_seed_store(Path(path))


def load_history_records(path: str | Path | None) -> list[dict]:
    if not path:
        return []
    history_path = Path(path)
    if not history_path.exists():
        return []
    records = []
    for line in history_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def build_submitted_alpha_context(records: list[dict]) -> dict:
    if not records:
        return {
            "count": 0,
            "entries": [],
            "skeletons": [],
            "alpha_type_counts": {},
            "style_tag_counts": {},
            "saturated_alpha_types": [],
            "dominant_style_tags": [],
        }

    alpha_type_counts = Counter()
    style_tag_counts = Counter()
    skeletons = set()
    entries = []

    for record in records:
        expression = record.get("expression", "")
        if not expression:
            continue
        try:
            metrics = score_expression(expression)
        except Exception:
            metrics = {"alpha_type": "unknown", "style_tags": []}
        alpha_type = metrics.get("alpha_type", "unknown")
        style_tags = list(metrics.get("style_tags", []))
        skeleton = expression_skeleton(expression)
        alpha_type_counts[alpha_type] += 1
        style_tag_counts.update(style_tags)
        skeletons.add(skeleton)
        entries.append(
            {
                **record,
                "alpha_type": alpha_type,
                "style_tags": style_tags,
                "skeleton": skeleton,
            }
        )

    return {
        "count": len(entries),
        "entries": entries,
        "skeletons": sorted(skeletons),
        "alpha_type_counts": dict(sorted(alpha_type_counts.items())),
        "style_tag_counts": dict(sorted(style_tag_counts.items())),
        "saturated_alpha_types": sorted(alpha_type for alpha_type, count in alpha_type_counts.items() if count >= 2),
        "dominant_style_tags": sorted(tag for tag, count in style_tag_counts.items() if count >= 2),
    }


def summarize_submitted_context(submitted_context: dict) -> dict:
    counts = submitted_context.get("alpha_type_counts", {})
    style_counts = submitted_context.get("style_tag_counts", {})
    top_alpha_types = [f"{name} x {count}" for name, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:4]]
    top_style_tags = [f"{name} x {count}" for name, count in sorted(style_counts.items(), key=lambda item: (-item[1], item[0]))[:6]]
    return {
        "count": int(submitted_context.get("count", 0)),
        "top_alpha_types": top_alpha_types,
        "top_style_tags": top_style_tags,
        "saturated_alpha_types": list(submitted_context.get("saturated_alpha_types", [])),
        "dominant_style_tags": list(submitted_context.get("dominant_style_tags", [])),
    }


def _empty_brain_feedback_stat() -> dict:
    return {
        "attempts": 0,
        "completed": 0,
        "strong_count": 0,
        "weak_count": 0,
        "severe_count": 0,
        "avg_sharpe": 0.0,
        "avg_fitness": 0.0,
        "avg_returns": 0.0,
        "avg_turnover": 0.0,
    }


def load_brain_feedback_rows(csv_path: str | Path | None) -> tuple[list[dict], dict]:
    required_columns = {"regular_code", "sharpe", "fitness", "returns"}
    try:
        resolved = discover_csv(str(csv_path) if csv_path else None)
    except FileNotFoundError:
        return [], {
            "status": "file_missing",
            "path": str(csv_path or ""),
            "message": "No simulation CSV was found.",
            "context_columns_present": [],
            "missing_context_columns": list(BRAIN_CONTEXT_FIELDS),
        }
    try:
        rows = read_rows(resolved)
    except Exception as exc:
        return [], {
            "status": "read_error",
            "path": str(resolved),
            "message": f"Failed to read simulation CSV: {exc}",
            "context_columns_present": [],
            "missing_context_columns": list(BRAIN_CONTEXT_FIELDS),
        }

    if rows:
        available_columns = set(rows[0].keys())
    else:
        available_columns = set()
    missing = sorted(required_columns - available_columns)
    context_columns_present = sorted(column for column in BRAIN_CONTEXT_FIELDS if column in available_columns)
    missing_context_columns = sorted(column for column in BRAIN_CONTEXT_FIELDS if column not in available_columns)
    migration_note = ""
    if not missing and missing_context_columns and resolved:
        migration = migrate_results_csv_context(resolved)
        if migration.get("status") == "migrated":
            rows = read_rows(resolved)
            available_columns = set(rows[0].keys()) if rows else set()
            context_columns_present = sorted(column for column in BRAIN_CONTEXT_FIELDS if column in available_columns)
            missing_context_columns = sorted(column for column in BRAIN_CONTEXT_FIELDS if column not in available_columns)
            migration_note = migration.get("message", "")
    if missing:
        return [], {
            "status": "schema_error",
            "path": str(resolved),
            "message": f"Simulation CSV is missing required columns: {', '.join(missing)}",
            "context_columns_present": context_columns_present,
            "missing_context_columns": missing_context_columns,
        }
    return rows, {
        "status": "ok",
        "path": str(resolved),
        "message": (
            f"Loaded {len(rows)} simulation rows."
            + (f" {migration_note}" if migration_note else "")
        ),
        "context_columns_present": context_columns_present,
        "missing_context_columns": missing_context_columns,
    }


def assess_brain_feedback_health(rows: list[dict], status: dict, brain_feedback_context: dict | None = None) -> dict:
    brain_feedback_context = brain_feedback_context or {}
    current_status = str(status.get("status", "unknown"))
    context_columns_present = list(status.get("context_columns_present", []))
    missing_context_columns = list(status.get("missing_context_columns", []))
    context_row_count = int(brain_feedback_context.get("context_row_count", 0) or 0)
    distinct_context_count = int(brain_feedback_context.get("distinct_context_count", 0) or 0)
    health = "healthy"
    hard_block = False
    reason = "ok"
    recommended_action = "continue"

    if current_status != "ok":
        health = "broken"
        hard_block = True
        reason = current_status
        recommended_action = "halt_and_fix_feedback"
    elif not rows:
        health = "broken"
        hard_block = True
        reason = "empty_feedback"
        recommended_action = "halt_and_fix_feedback"
    elif missing_context_columns:
        health = "degraded"
        hard_block = True
        reason = "missing_context_columns"
        recommended_action = "upgrade_feedback_schema"
    elif context_row_count <= 0 or distinct_context_count <= 0:
        health = "degraded"
        hard_block = True
        reason = "missing_context_values"
        recommended_action = "rerun_with_contextual_feedback"

    message = status.get("message", "")
    if hard_block and missing_context_columns:
        message = (
            f"{message} Missing context columns: {', '.join(missing_context_columns)}."
            if message
            else f"Missing context columns: {', '.join(missing_context_columns)}."
        )

    enriched = dict(status)
    enriched.update(
        {
            "health": health,
            "hard_block": hard_block,
            "reason": reason,
            "recommended_action": recommended_action,
            "context_columns_present": context_columns_present,
            "missing_context_columns": missing_context_columns,
            "context_row_count": context_row_count,
            "distinct_context_count": distinct_context_count,
            "message": message,
        }
    )
    return enriched


def _brain_row_labels(row: dict) -> tuple[bool, bool, bool]:
    sharpe = coerce_float(row.get("sharpe")) or 0.0
    fitness = coerce_float(row.get("fitness")) or 0.0
    returns = coerce_float(row.get("returns")) or 0.0
    failures = set(failing_checks(row))

    strong = (
        sharpe >= 1.25
        and fitness >= 1.0
        and returns > 0.02
        and "HIGH_TURNOVER" not in failures
        and "LOW_SUB_UNIVERSE_SHARPE" not in failures
    )
    weak = (
        sharpe < 0.95
        or fitness < 0.8
        or returns <= 0.0
        or {"LOW_SHARPE", "LOW_FITNESS"} <= failures
    )
    severe = (
        sharpe < 0.7
        or fitness < 0.55
        or returns < -0.03
        or len(failures & {"LOW_SHARPE", "LOW_FITNESS", "LOW_SUB_UNIVERSE_SHARPE"}) >= 2
    )
    return strong, weak, severe


def _update_brain_feedback_stat(stat: dict, *, row: dict) -> None:
    sharpe = coerce_float(row.get("sharpe")) or 0.0
    fitness = coerce_float(row.get("fitness")) or 0.0
    returns = coerce_float(row.get("returns")) or 0.0
    turnover = coerce_float(row.get("turnover")) or 0.0
    pending = row_has_pending_checks(row)
    strong, weak, severe = _brain_row_labels(row)

    count = stat["attempts"]
    stat["avg_sharpe"] = _update_running_average(stat["avg_sharpe"], count, sharpe)
    stat["avg_fitness"] = _update_running_average(stat["avg_fitness"], count, fitness)
    stat["avg_returns"] = _update_running_average(stat["avg_returns"], count, returns)
    stat["avg_turnover"] = _update_running_average(stat["avg_turnover"], count, turnover)
    stat["attempts"] += 1
    stat["completed"] += int(not pending)
    stat["strong_count"] += int(strong)
    stat["weak_count"] += int(weak)
    stat["severe_count"] += int(severe)


def _summarize_brain_feedback_stat(stat: dict, *, exact_skeleton: bool = False) -> dict:
    attempts = int(stat.get("attempts", 0))
    if attempts <= 0:
        return {
            "attempts": 0,
            "completed": 0,
            "strong_count": 0,
            "weak_count": 0,
            "severe_count": 0,
            "strong_rate": 0.0,
            "weak_rate": 0.0,
            "severe_rate": 0.0,
            "avg_sharpe": 0.0,
            "avg_fitness": 0.0,
            "avg_returns": 0.0,
            "avg_turnover": 0.0,
            "severity": 0.0,
            "preference": 0.0,
            "discouraged": False,
            "preferred": False,
            "blocked_exact": False,
        }

    strong_rate = stat["strong_count"] / attempts
    weak_rate = stat["weak_count"] / attempts
    severe_rate = stat["severe_count"] / attempts
    avg_sharpe = float(stat["avg_sharpe"])
    avg_fitness = float(stat["avg_fitness"])
    avg_returns = float(stat["avg_returns"])
    avg_turnover = float(stat["avg_turnover"])

    severity = _clip(
        0.36 * _clip((0.95 - avg_sharpe) / 1.05, 0.0, 1.0)
        + 0.22 * _clip((0.85 - avg_fitness) / 0.85, 0.0, 1.0)
        + 0.16 * _clip((-avg_returns) / 0.10, 0.0, 1.0)
        + 0.16 * weak_rate
        + 0.14 * severe_rate
        - 0.18 * strong_rate,
        0.0,
        1.0,
    )
    preference = _clip(
        0.38 * _clip((avg_sharpe - 1.1) / 0.9, 0.0, 1.0)
        + 0.20 * _clip((avg_fitness - 1.0) / 0.9, 0.0, 1.0)
        + 0.18 * _clip((avg_returns - 0.02) / 0.18, 0.0, 1.0)
        + 0.18 * strong_rate
        - 0.16 * weak_rate,
        0.0,
        1.0,
    )
    discouraged = attempts >= 3 and severity >= 0.42 and strong_rate <= 0.2
    preferred = attempts >= 2 and preference >= 0.32 and strong_rate >= weak_rate
    blocked_exact = exact_skeleton and attempts >= 1 and stat["strong_count"] == 0 and (stat["severe_count"] >= 1 or stat["weak_count"] >= 2)

    return {
        "attempts": attempts,
        "completed": int(stat.get("completed", 0)),
        "strong_count": int(stat.get("strong_count", 0)),
        "weak_count": int(stat.get("weak_count", 0)),
        "severe_count": int(stat.get("severe_count", 0)),
        "strong_rate": _round(strong_rate),
        "weak_rate": _round(weak_rate),
        "severe_rate": _round(severe_rate),
        "avg_sharpe": _round(avg_sharpe),
        "avg_fitness": _round(avg_fitness),
        "avg_returns": _round(avg_returns),
        "avg_turnover": _round(avg_turnover),
        "severity": _round(severity),
        "preference": _round(preference),
        "discouraged": discouraged,
        "preferred": preferred,
        "blocked_exact": blocked_exact,
    }


def _summarize_brain_feedback_rows(rows: list[dict], *, exact_skeleton: bool = False) -> dict:
    stat = _empty_brain_feedback_stat()
    for row in rows:
        _update_brain_feedback_stat(stat, row=row)
    return _summarize_brain_feedback_stat(stat, exact_skeleton=exact_skeleton)


def _resolve_brain_feedback_stat(
    summary_map: dict,
    record_map: dict,
    key: str,
    candidate_context: dict,
    *,
    exact_skeleton: bool = False,
) -> tuple[dict | None, str, float, int]:
    global_stat = summary_map.get(key)
    rows = list(record_map.get(key, []))
    if not rows and not global_stat:
        return None, "no_signal", 0.0, 0

    contextual_rows = [row for row in rows if row.get("_brain_context")]
    matched_rows = [row for row in contextual_rows if _brain_context_matches(row.get("_brain_context", {}), candidate_context)]
    min_exact_attempts = 1 if exact_skeleton else 2
    if len(matched_rows) >= min_exact_attempts:
        factor = 1.0 if exact_skeleton or len(matched_rows) >= 3 else 0.8
        return _summarize_brain_feedback_rows(matched_rows, exact_skeleton=exact_skeleton), "exact_context", factor, len(matched_rows)
    if 0 < len(matched_rows) < min_exact_attempts:
        return _summarize_brain_feedback_rows(matched_rows, exact_skeleton=exact_skeleton), "thin_context", 0.55, len(matched_rows)
    if contextual_rows and candidate_context:
        mismatch_factor = 0.75 if exact_skeleton else 0.45
        return global_stat, "context_mismatch", mismatch_factor, len(contextual_rows)
    legacy_factor = 1.0 if exact_skeleton else 0.70
    return global_stat, "global_legacy", legacy_factor, len(rows)


def build_brain_feedback_context(rows: list[dict]) -> dict:
    if not rows:
        return {
            "count": 0,
            "family_feedback": {},
            "style_tag_feedback": {},
            "alpha_type_feedback": {},
            "skeleton_feedback": {},
            "family_records": {},
            "style_tag_records": {},
            "alpha_type_records": {},
            "skeleton_records": {},
            "weak_families": [],
            "preferred_families": [],
            "weak_style_tags": [],
            "weak_alpha_types": [],
            "blocked_skeletons": [],
            "context_row_count": 0,
            "distinct_context_count": 0,
        }

    family_stats = defaultdict(_empty_brain_feedback_stat)
    style_tag_stats = defaultdict(_empty_brain_feedback_stat)
    alpha_type_stats = defaultdict(_empty_brain_feedback_stat)
    skeleton_stats = defaultdict(_empty_brain_feedback_stat)
    family_records = defaultdict(list)
    style_tag_records = defaultdict(list)
    alpha_type_records = defaultdict(list)
    skeleton_records = defaultdict(list)
    analyzed_rows = 0
    context_signatures = Counter()

    for row in rows:
        expression = str(row.get("regular_code", "")).strip()
        if not expression:
            continue
        analyzed_rows += 1
        enriched_row = dict(row)
        enriched_row["_brain_context"] = extract_brain_context(row)
        if enriched_row["_brain_context"]:
            context_signatures.update([format_brain_context(enriched_row["_brain_context"])])
        family = classify_expression_family(expression)
        skeleton = expression_skeleton(expression)
        try:
            metrics = score_expression(expression)
            style_tags = list(metrics.get("style_tags", []))
            alpha_type = metrics.get("alpha_type", "unknown")
        except Exception:
            style_tags = []
            alpha_type = "unknown"

        _update_brain_feedback_stat(family_stats[family], row=enriched_row)
        _update_brain_feedback_stat(alpha_type_stats[alpha_type], row=enriched_row)
        _update_brain_feedback_stat(skeleton_stats[skeleton], row=enriched_row)
        family_records[family].append(enriched_row)
        alpha_type_records[alpha_type].append(enriched_row)
        skeleton_records[skeleton].append(enriched_row)
        for tag in style_tags:
            _update_brain_feedback_stat(style_tag_stats[tag], row=enriched_row)
            style_tag_records[tag].append(enriched_row)

    family_feedback = {
        name: _summarize_brain_feedback_stat(stat)
        for name, stat in sorted(family_stats.items())
    }
    style_tag_feedback = {
        name: _summarize_brain_feedback_stat(stat)
        for name, stat in sorted(style_tag_stats.items())
    }
    alpha_type_feedback = {
        name: _summarize_brain_feedback_stat(stat)
        for name, stat in sorted(alpha_type_stats.items())
    }
    skeleton_feedback = {
        name: _summarize_brain_feedback_stat(stat, exact_skeleton=True)
        for name, stat in sorted(skeleton_stats.items())
    }

    weak_families = sorted(
        [name for name, stat in family_feedback.items() if stat.get("discouraged")],
        key=lambda name: (
            float(family_feedback[name]["severity"]),
            int(family_feedback[name]["attempts"]),
            name,
        ),
        reverse=True,
    )
    preferred_families = sorted(
        [name for name, stat in family_feedback.items() if stat.get("preferred")],
        key=lambda name: (
            float(family_feedback[name]["preference"]),
            int(family_feedback[name]["attempts"]),
            name,
        ),
        reverse=True,
    )
    weak_style_tags = sorted(
        [
            name
            for name, stat in style_tag_feedback.items()
            if stat.get("discouraged") and stat.get("attempts", 0) >= 4
        ],
        key=lambda name: (
            float(style_tag_feedback[name]["severity"]),
            int(style_tag_feedback[name]["attempts"]),
            name,
        ),
        reverse=True,
    )
    weak_alpha_types = sorted(
        [name for name, stat in alpha_type_feedback.items() if stat.get("discouraged")],
        key=lambda name: (
            float(alpha_type_feedback[name]["severity"]),
            int(alpha_type_feedback[name]["attempts"]),
            name,
        ),
        reverse=True,
    )
    blocked_skeletons = sorted(
        [name for name, stat in skeleton_feedback.items() if stat.get("blocked_exact")],
        key=lambda name: (
            float(skeleton_feedback[name]["severity"]),
            int(skeleton_feedback[name]["attempts"]),
            name,
        ),
        reverse=True,
    )

    return {
        "count": analyzed_rows,
        "family_feedback": family_feedback,
        "style_tag_feedback": style_tag_feedback,
        "alpha_type_feedback": alpha_type_feedback,
        "skeleton_feedback": skeleton_feedback,
        "family_records": {name: records for name, records in family_records.items()},
        "style_tag_records": {name: records for name, records in style_tag_records.items()},
        "alpha_type_records": {name: records for name, records in alpha_type_records.items()},
        "skeleton_records": {name: records for name, records in skeleton_records.items()},
        "weak_families": weak_families,
        "preferred_families": preferred_families,
        "weak_style_tags": weak_style_tags,
        "weak_alpha_types": weak_alpha_types,
        "blocked_skeletons": blocked_skeletons,
        "context_row_count": sum(context_signatures.values()),
        "distinct_context_count": len(context_signatures),
    }


def summarize_brain_feedback_context(brain_feedback_context: dict) -> dict:
    if not brain_feedback_context.get("count"):
        return {
            "count": 0,
            "top_weak_families": [],
            "top_preferred_families": [],
            "top_weak_style_tags": [],
            "top_weak_alpha_types": [],
            "blocked_skeleton_count": 0,
            "context_row_count": 0,
            "distinct_context_count": 0,
        }

    def _format_entries(names: list[str], mapping: dict, key: str) -> list[str]:
        lines = []
        for name in names[:4]:
            stat = mapping.get(name, {})
            lines.append(
                f"{name} (attempts={stat.get('attempts', 0)}, "
                f"avg_sharpe={stat.get('avg_sharpe', 0.0)}, {key}={stat.get(key, 0.0)})"
            )
        return lines

    return {
        "count": int(brain_feedback_context.get("count", 0)),
        "top_weak_families": _format_entries(
            brain_feedback_context.get("weak_families", []),
            brain_feedback_context.get("family_feedback", {}),
            "severity",
        ),
        "top_preferred_families": _format_entries(
            brain_feedback_context.get("preferred_families", []),
            brain_feedback_context.get("family_feedback", {}),
            "preference",
        ),
        "top_weak_style_tags": _format_entries(
            brain_feedback_context.get("weak_style_tags", []),
            brain_feedback_context.get("style_tag_feedback", {}),
            "severity",
        ),
        "top_weak_alpha_types": _format_entries(
            brain_feedback_context.get("weak_alpha_types", []),
            brain_feedback_context.get("alpha_type_feedback", {}),
            "severity",
        ),
        "blocked_skeleton_count": len(brain_feedback_context.get("blocked_skeletons", [])),
        "context_row_count": int(brain_feedback_context.get("context_row_count", 0)),
        "distinct_context_count": int(brain_feedback_context.get("distinct_context_count", 0)),
    }


def compute_brain_feedback_penalty(
    expression: str,
    result: dict,
    family_components: list[str],
    brain_feedback_context: dict | None,
) -> dict:
    brain_feedback_context = brain_feedback_context or {}
    if not brain_feedback_context.get("count"):
        return {
            "penalty_score": 0.0,
            "confidence_drag": 0.0,
            "reasons": [],
            "matched_families": [],
            "matched_style_tags": [],
            "exact_skeleton_match": False,
        }

    penalty_score = 0.0
    reasons = []
    matched_families = []
    matched_style_tags = []
    exact_skeleton_match = False
    context_modes = []
    candidate_context = extract_brain_context(result.get("settings", {}))

    skeleton = expression_skeleton(expression)
    skeleton_feedback, skeleton_mode, skeleton_factor, skeleton_attempts = _resolve_brain_feedback_stat(
        brain_feedback_context.get("skeleton_feedback", {}),
        brain_feedback_context.get("skeleton_records", {}),
        skeleton,
        candidate_context,
        exact_skeleton=True,
    )
    if skeleton_feedback and skeleton_feedback.get("blocked_exact"):
        exact_skeleton_match = True
        penalty_score += (0.18 + 0.18 * float(skeleton_feedback.get("severity", 0.0))) * skeleton_factor
        context_modes.append(skeleton_mode)
        reasons.append(
            f"brain_exact_skeleton avg_sharpe={skeleton_feedback.get('avg_sharpe')} "
            f"avg_fitness={skeleton_feedback.get('avg_fitness')} "
            f"context={skeleton_mode} attempts={skeleton_attempts or skeleton_feedback.get('attempts', 0)}"
        )

    family_feedback_map = brain_feedback_context.get("family_feedback", {})
    family_record_map = brain_feedback_context.get("family_records", {})
    for family_id in family_components:
        family_feedback, family_mode, family_factor, family_attempts = _resolve_brain_feedback_stat(
            family_feedback_map,
            family_record_map,
            family_id,
            candidate_context,
        )
        if not family_feedback:
            continue
        if family_feedback.get("discouraged"):
            matched_families.append(family_id)
            penalty_score += (0.05 + 0.12 * float(family_feedback.get("severity", 0.0))) * family_factor
            context_modes.append(family_mode)
            reasons.append(
                f"brain_family={family_id} avg_sharpe={family_feedback.get('avg_sharpe')} "
                f"severity={family_feedback.get('severity')} "
                f"context={family_mode} attempts={family_attempts or family_feedback.get('attempts', 0)}"
            )
        elif family_feedback.get("preferred"):
            penalty_score -= 0.03 * float(family_feedback.get("preference", 0.0)) * family_factor

    alpha_type = result.get("alpha_type", "unknown")
    alpha_type_feedback, alpha_mode, alpha_factor, alpha_attempts = _resolve_brain_feedback_stat(
        brain_feedback_context.get("alpha_type_feedback", {}),
        brain_feedback_context.get("alpha_type_records", {}),
        alpha_type,
        candidate_context,
    )
    if alpha_type_feedback and alpha_type_feedback.get("discouraged"):
        penalty_score += (0.03 + 0.08 * float(alpha_type_feedback.get("severity", 0.0))) * alpha_factor
        context_modes.append(alpha_mode)
        reasons.append(
            f"brain_alpha_type={alpha_type} avg_sharpe={alpha_type_feedback.get('avg_sharpe')} "
            f"severity={alpha_type_feedback.get('severity')} "
            f"context={alpha_mode} attempts={alpha_attempts or alpha_type_feedback.get('attempts', 0)}"
        )

    style_penalty = 0.0
    style_record_map = brain_feedback_context.get("style_tag_records", {})
    for tag in sorted(set(result.get("style_tags", []))):
        style_feedback, style_mode, style_factor, style_attempts = _resolve_brain_feedback_stat(
            brain_feedback_context.get("style_tag_feedback", {}),
            style_record_map,
            tag,
            candidate_context,
        )
        if not style_feedback or not style_feedback.get("discouraged"):
            continue
        matched_style_tags.append(tag)
        style_penalty += (0.015 + 0.04 * float(style_feedback.get("severity", 0.0))) * style_factor
        context_modes.append(style_mode)
        reasons.append(
            f"brain_style_tag={tag} avg_sharpe={style_feedback.get('avg_sharpe')} "
            f"severity={style_feedback.get('severity')} "
            f"context={style_mode} attempts={style_attempts or style_feedback.get('attempts', 0)}"
        )
    penalty_score += min(0.12, style_penalty)

    penalty_score = _clip(penalty_score, 0.0, 0.48)
    confidence_drag = _clip(penalty_score * 0.85, 0.0, 0.36)
    dominant_mode = Counter(context_modes).most_common(1)[0][0] if context_modes else "no_signal"
    return {
        "penalty_score": _round(penalty_score),
        "confidence_drag": _round(confidence_drag),
        "reasons": reasons[:5],
        "matched_families": matched_families,
        "matched_style_tags": matched_style_tags[:5],
        "exact_skeleton_match": exact_skeleton_match,
        "context_mode": dominant_mode,
        "candidate_context": format_brain_context(candidate_context),
    }


def append_history_records(path: str | Path, records: list[dict]) -> None:
    if not records:
        return
    history_path = Path(path)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def aggregate_scout_memory(records: list[dict]) -> dict:
    source_stats = defaultdict(_empty_stat)
    skeleton_stats = defaultdict(_empty_stat)
    family_horizon_stats = defaultdict(_empty_stat)
    source_family_stats = defaultdict(_empty_stat)

    for record in records:
        source_key = record.get("source_key")
        skeleton = record.get("skeleton")
        selection_status = record.get("selection_status")
        alpha_score = float(record.get("alpha_score", 0.0))
        confidence_score = float(record.get("confidence_score", 0.0))
        source_quality = float(record.get("source_quality_score", 0.0))
        is_selected = selection_status == "selected"
        is_watchlist = selection_status == "watchlist"
        is_strong = is_selected and record.get("verdict") in {"PASS", "LIKELY_PASS"} and alpha_score >= 68.0

        if source_key:
            stat = source_stats[source_key]
            count = stat["attempts"]
            stat["avg_alpha_score"] = _update_running_average(stat["avg_alpha_score"], count, alpha_score)
            stat["avg_confidence_score"] = _update_running_average(stat["avg_confidence_score"], count, confidence_score)
            stat["avg_source_quality"] = _update_running_average(stat["avg_source_quality"], count, source_quality)
            stat["attempts"] += 1
            stat["selected_count"] += int(is_selected)
            stat["watchlist_count"] += int(is_watchlist)
            stat["strong_count"] += int(is_strong)

        if skeleton:
            stat = skeleton_stats[skeleton]
            count = stat["attempts"]
            stat["avg_alpha_score"] = _update_running_average(stat["avg_alpha_score"], count, alpha_score)
            stat["avg_confidence_score"] = _update_running_average(stat["avg_confidence_score"], count, confidence_score)
            stat["avg_source_quality"] = _update_running_average(stat["avg_source_quality"], count, source_quality)
            stat["attempts"] += 1
            stat["selected_count"] += int(is_selected)
            stat["watchlist_count"] += int(is_watchlist)
            stat["strong_count"] += int(is_strong)

        family_components = record.get("family_components", []) or [record.get("thesis_id") or "unknown"]
        horizon = record.get("horizon", "medium")
        for family_id in family_components:
            family_key = f"{family_id}|{horizon}"
            stat = family_horizon_stats[family_key]
            count = stat["attempts"]
            stat["avg_alpha_score"] = _update_running_average(stat["avg_alpha_score"], count, alpha_score)
            stat["avg_confidence_score"] = _update_running_average(stat["avg_confidence_score"], count, confidence_score)
            stat["avg_source_quality"] = _update_running_average(stat["avg_source_quality"], count, source_quality)
            stat["attempts"] += 1
            stat["selected_count"] += int(is_selected)
            stat["watchlist_count"] += int(is_watchlist)
            stat["strong_count"] += int(is_strong)

            if source_key:
                sf_key = f"{source_key}|{family_id}"
                stat = source_family_stats[sf_key]
                count = stat["attempts"]
                stat["avg_alpha_score"] = _update_running_average(stat["avg_alpha_score"], count, alpha_score)
                stat["avg_confidence_score"] = _update_running_average(stat["avg_confidence_score"], count, confidence_score)
                stat["avg_source_quality"] = _update_running_average(stat["avg_source_quality"], count, source_quality)
                stat["attempts"] += 1
                stat["selected_count"] += int(is_selected)
                stat["watchlist_count"] += int(is_watchlist)
                stat["strong_count"] += int(is_strong)

    blocked_source_keys = sorted(
        source_key
        for source_key, stat in source_stats.items()
        if stat["attempts"] >= 2 and stat["selected_count"] == 0 and stat["avg_alpha_score"] < 60.0
    )
    blocked_skeletons = sorted(
        skeleton
        for skeleton, stat in skeleton_stats.items()
        if stat["attempts"] >= 2 and stat["strong_count"] == 0 and stat["avg_alpha_score"] < 58.0
    )
    preferred_source_keys = sorted(
        source_key
        for source_key, stat in source_stats.items()
        if stat["strong_count"] >= 1 and stat["avg_alpha_score"] >= 68.0
    )

    preferred_family_horizons = {}
    grouped_family_horizons = defaultdict(list)
    for key, stat in family_horizon_stats.items():
        family_id, horizon = key.split("|", 1)
        grouped_family_horizons[family_id].append((horizon, stat))

    for family_id, candidates in grouped_family_horizons.items():
        best_horizon, best_stat = max(
            candidates,
            key=lambda item: (
                item[1]["strong_count"],
                item[1]["selected_count"],
                item[1]["avg_alpha_score"],
                item[1]["avg_confidence_score"],
            ),
        )
        if best_stat["selected_count"] >= 1:
            preferred_family_horizons[family_id] = best_horizon

    return {
        "records": len(records),
        "blocked_source_keys": blocked_source_keys,
        "blocked_skeletons": blocked_skeletons,
        "preferred_source_keys": preferred_source_keys,
        "preferred_family_horizons": preferred_family_horizons,
        "source_stats": {key: {name: _round(value) if isinstance(value, float) else value for name, value in stat.items()} for key, stat in source_stats.items()},
        "skeleton_stats": {key: {name: _round(value) if isinstance(value, float) else value for name, value in stat.items()} for key, stat in skeleton_stats.items()},
        "family_horizon_stats": {key: {name: _round(value) if isinstance(value, float) else value for name, value in stat.items()} for key, stat in family_horizon_stats.items()},
        "source_family_stats": {key: {name: _round(value) if isinstance(value, float) else value for name, value in stat.items()} for key, stat in source_family_stats.items()},
    }


def rebuild_openalex_abstract(inverted_index: dict | None) -> str:
    if not inverted_index:
        return ""
    items = []
    for word, positions in inverted_index.items():
        for position in positions:
            items.append((position, word))
    items.sort(key=lambda item: item[0])
    return " ".join(word for _, word in items)


def _append_query_profile(profiles: list[dict], seen_queries: set[str], profile: dict) -> None:
    query = normalize_text(profile.get("query", ""))
    if not query or query in seen_queries:
        return
    profiles.append(
        {
            "query": profile["query"],
            "families": list(profile.get("families", [])),
            "style_tags": list(profile.get("style_tags", [])),
            "horizon": profile.get("horizon", "medium"),
        }
    )
    seen_queries.add(query)


def build_query_profiles(
    memory: dict,
    scout_memory: dict,
    *,
    search_breadth: str = "standard",
    max_profiles: int | None = None,
) -> list[dict]:
    breadth = search_breadth if search_breadth in SEARCH_BREADTH_LIMITS else "standard"
    default_limit = SEARCH_BREADTH_LIMITS[breadth]
    profile_limit = max(6, max_profiles or default_limit)

    profiles = []
    seen_queries: set[str] = set()

    base_seed = BASE_QUERY_PROFILES if breadth != "focused" else BASE_QUERY_PROFILES[:6]
    extended_seed = []
    if breadth == "standard":
        extended_seed = EXTENDED_QUERY_PROFILES[:6]
    elif breadth == "wide":
        extended_seed = EXTENDED_QUERY_PROFILES
    elif breadth == "explore":
        extended_seed = [*EXTENDED_QUERY_PROFILES, *EXPLORATION_QUERY_PROFILES]

    for profile in [*base_seed, *extended_seed]:
        _append_query_profile(profiles, seen_queries, profile)

    top_tag_limit = {"focused": 3, "standard": 5, "wide": 8, "explore": 10}[breadth]
    top_tags = [item.get("tag") for item in memory.get("style_leaders", [])[:top_tag_limit] if item.get("tag")]
    for tag in top_tags:
        expansion_limit = 1 if breadth == "focused" else 2 if breadth in {"standard", "wide"} else 3
        for profile in STYLE_QUERY_EXPANSIONS.get(tag, [])[:expansion_limit]:
            _append_query_profile(profiles, seen_queries, profile)

    preferred_horizons = scout_memory.get("preferred_family_horizons", {})
    family_limit = {"focused": 1, "standard": 2, "wide": 3, "explore": 4}[breadth]
    for family_id, horizon in preferred_horizons.items():
        if family_id not in THESIS_INDEX:
            continue
        label = THESIS_INDEX[family_id]["label"]
        style_tags = sorted({tag for variant in THESIS_INDEX[family_id]["variants"] for tag in variant.get("style_tags", [])})
        _append_query_profile(
            profiles,
            seen_queries,
            {
                "query": f"{label} equity anomaly",
                "families": [family_id],
                "style_tags": style_tags,
                "horizon": horizon,
            },
        )
        for query in FAMILY_QUERY_EXPANSIONS.get(family_id, [])[:family_limit]:
            _append_query_profile(
                profiles,
                seen_queries,
                {
                    "query": query,
                    "families": [family_id],
                    "style_tags": style_tags,
                    "horizon": horizon,
                },
            )

    if breadth in {"wide", "explore"}:
        for family_id, thesis in THESIS_INDEX.items():
            if len(profiles) >= profile_limit:
                break
            style_tags = sorted({tag for variant in thesis["variants"] for tag in variant.get("style_tags", [])})
            for query in FAMILY_QUERY_EXPANSIONS.get(family_id, []):
                _append_query_profile(
                    profiles,
                    seen_queries,
                    {
                        "query": query,
                        "families": [family_id],
                        "style_tags": style_tags,
                        "horizon": scout_memory.get("preferred_family_horizons", {}).get(family_id, "medium"),
                    },
                )
                if len(profiles) >= profile_limit:
                    break
            if breadth == "explore" and len(profiles) < profile_limit:
                _append_query_profile(
                    profiles,
                    seen_queries,
                    {
                        "query": f"{thesis['label']} cross sectional stock alpha",
                        "families": [family_id],
                        "style_tags": style_tags,
                        "horizon": scout_memory.get("preferred_family_horizons", {}).get(family_id, "medium"),
                    },
                )
                _append_query_profile(
                    profiles,
                    seen_queries,
                    {
                        "query": f"{thesis['label']} market neutral anomaly",
                        "families": [family_id],
                        "style_tags": style_tags,
                        "horizon": scout_memory.get("preferred_family_horizons", {}).get(family_id, "medium"),
                    },
                )

    return profiles[:profile_limit]


def compute_source_score(*, year: int | None, citations: int = 0, relevance: float = 0.0, source: str = "openalex") -> float:
    current_year = date.today().year
    recency = 0.42
    if year:
        age = max(0, current_year - year)
        recency = max(0.08, 1.0 - min(age, 12) / 12.0)
    citation_component = min(1.0, max(0, citations) / 200.0)
    relevance_component = min(1.0, max(0.0, relevance) / 1000.0)
    base_by_source = {
        "openalex": 0.36,
        "arxiv": 0.40,
        "github": 0.28,
        "github_readme": 0.34,
        "zip_knowledge": 0.46,
        "finding_alphas_book": 0.44,
    }
    base = base_by_source.get(source, 0.32)
    return round(min(1.0, base + 0.28 * recency + 0.18 * citation_component + 0.12 * relevance_component), 4)


def fetch_openalex_ideas(
    query_profiles: list[dict],
    *,
    per_query: int = 2,
    timeout: int = DEFAULT_PUBLIC_API_TIMEOUT,
    fetch_cache: dict | None = None,
    request_throttle: ScoutRequestThrottle | None = None,
    max_retries: int = DEFAULT_PUBLIC_API_MAX_RETRIES,
) -> tuple[list[dict], dict]:
    ideas = []
    status = {"attempted": len(query_profiles), "succeeded": 0, "failed": 0, "cached": 0, "retries": 0, "quota_limited": 0, "server_errors": 0, "network_requests": 0, "skipped_budget": 0, "cooldowns": 0, "budget_exhausted": False}
    for profile in query_profiles:
        cache_key = f"{profile['query']}|{per_query}"
        cached_payload = get_cached_fetch(fetch_cache or {}, namespace="openalex", key=cache_key, ttl_seconds=FETCH_CACHE_TTLS["openalex"])
        if isinstance(cached_payload, list):
            ideas.extend(cached_payload)
            status["cached"] += 1
            continue
        response = _fetch_url_with_policy(
            OPENALEX_URL,
            params={"search": profile["query"], "per-page": per_query, "sort": "relevance_score:desc"},
            timeout=timeout,
            context=f"OpenAlex fetch for '{profile['query']}'",
            status=status,
            throttle=request_throttle,
            max_retries=max_retries,
        )
        if response is None:
            if status.get("budget_exhausted"):
                break
            status["failed"] += 1
            continue
        if response.status_code < 200 or response.status_code >= 300:
            status["failed"] += 1
            continue

        status["succeeded"] += 1
        payload = response.json()
        cached_ideas = []
        for item in payload.get("results", []):
            title = item.get("display_name") or item.get("title") or ""
            cached_ideas.append(
                {
                    "source": "openalex",
                    "query": profile["query"],
                    "title": title,
                    "summary": rebuild_openalex_abstract(item.get("abstract_inverted_index")),
                    "url": item.get("id") or item.get("primary_location", {}).get("landing_page_url") or "",
                    "openalex_id": item.get("id") or "",
                    "doi": item.get("ids", {}).get("doi") or "",
                    "year": item.get("publication_year"),
                    "citations": item.get("cited_by_count") or 0,
                    "relevance": float(item.get("relevance_score") or 0.0),
                    "bias_families": list(profile.get("families", [])),
                    "bias_style_tags": list(profile.get("style_tags", [])),
                    "bias_horizon": profile.get("horizon", "medium"),
                    "source_score": compute_source_score(
                        year=item.get("publication_year"),
                        citations=item.get("cited_by_count") or 0,
                        relevance=float(item.get("relevance_score") or 0.0),
                        source="openalex",
                    ),
                }
            )
        ideas.extend(cached_ideas)
        if fetch_cache is not None:
            put_cached_fetch(fetch_cache, namespace="openalex", key=cache_key, payload=cached_ideas)
    return ideas, status


def fetch_arxiv_ideas(
    query_profiles: list[dict],
    *,
    per_query: int = 1,
    timeout: int = DEFAULT_PUBLIC_API_TIMEOUT,
    fetch_cache: dict | None = None,
    request_throttle: ScoutRequestThrottle | None = None,
    max_retries: int = DEFAULT_PUBLIC_API_MAX_RETRIES,
) -> tuple[list[dict], dict]:
    ideas = []
    status = {"attempted": len(query_profiles), "succeeded": 0, "failed": 0, "cached": 0, "retries": 0, "quota_limited": 0, "server_errors": 0, "network_requests": 0, "skipped_budget": 0, "cooldowns": 0, "budget_exhausted": False}
    namespace = {"atom": "http://www.w3.org/2005/Atom"}
    for profile in query_profiles:
        cache_key = f"{profile['query']}|{per_query}"
        cached_payload = get_cached_fetch(fetch_cache or {}, namespace="arxiv", key=cache_key, ttl_seconds=FETCH_CACHE_TTLS["arxiv"])
        if isinstance(cached_payload, list):
            ideas.extend(cached_payload)
            status["cached"] += 1
            continue
        response = _fetch_url_with_policy(
            ARXIV_URL,
            params={
                "search_query": f"all:{profile['query']}",
                "start": 0,
                "max_results": per_query,
                "sortBy": "relevance",
                "sortOrder": "descending",
            },
            timeout=timeout,
            context=f"arXiv fetch for '{profile['query']}'",
            status=status,
            throttle=request_throttle,
            max_retries=max_retries,
        )
        if response is None:
            if status.get("budget_exhausted"):
                break
            status["failed"] += 1
            continue
        if response.status_code < 200 or response.status_code >= 300:
            status["failed"] += 1
            continue

        status["succeeded"] += 1
        try:
            root = ET.fromstring(response.text)
        except ET.ParseError:
            status["failed"] += 1
            continue

        cached_ideas = []
        for entry in root.findall("atom:entry", namespace):
            title = (entry.findtext("atom:title", default="", namespaces=namespace) or "").strip()
            summary = (entry.findtext("atom:summary", default="", namespaces=namespace) or "").strip()
            published = entry.findtext("atom:published", default="", namespaces=namespace) or ""
            year = int(published[:4]) if len(published) >= 4 and published[:4].isdigit() else None
            cached_ideas.append(
                {
                    "source": "arxiv",
                    "query": profile["query"],
                    "title": title,
                    "summary": summary,
                    "url": (entry.findtext("atom:id", default="", namespaces=namespace) or "").strip(),
                    "arxiv_id": (entry.findtext("atom:id", default="", namespaces=namespace) or "").strip(),
                    "doi": "",
                    "year": year,
                    "citations": 0,
                    "relevance": 0.0,
                    "bias_families": list(profile.get("families", [])),
                    "bias_style_tags": list(profile.get("style_tags", [])),
                    "bias_horizon": profile.get("horizon", "medium"),
                    "source_score": compute_source_score(year=year, citations=0, relevance=0.0, source="arxiv"),
                }
            )
        ideas.extend(cached_ideas)
        if fetch_cache is not None:
            put_cached_fetch(fetch_cache, namespace="arxiv", key=cache_key, payload=cached_ideas)
    return ideas, status


def _github_headers() -> dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "brain-learn-scout",
    }
    github_token = os.getenv("GITHUB_TOKEN", "").strip()
    if github_token:
        headers["Authorization"] = f"Bearer {github_token}"
    return headers


def fetch_github_ideas(
    query_profiles: list[dict],
    *,
    per_query: int = 1,
    timeout: int = DEFAULT_PUBLIC_API_TIMEOUT,
    max_queries: int = DEFAULT_GITHUB_QUERY_LIMIT,
    fetch_cache: dict | None = None,
    request_throttle: ScoutRequestThrottle | None = None,
    max_retries: int = DEFAULT_PUBLIC_API_MAX_RETRIES,
) -> tuple[list[dict], dict]:
    ideas = []
    effective_profiles = list(query_profiles[: max(0, max_queries)])
    status = {"attempted": len(effective_profiles), "succeeded": 0, "failed": 0, "cached": 0, "retries": 0, "quota_limited": 0, "server_errors": 0, "network_requests": 0, "skipped_budget": 0, "cooldowns": 0, "budget_exhausted": False}
    if not effective_profiles or per_query <= 0:
        return ideas, status

    headers = _github_headers()

    for profile in effective_profiles:
        query = f"{profile['query']} in:name,description,readme"
        cache_key = f"{profile['query']}|{per_query}"
        cached_payload = get_cached_fetch(fetch_cache or {}, namespace="github", key=cache_key, ttl_seconds=FETCH_CACHE_TTLS["github"])
        if isinstance(cached_payload, list):
            ideas.extend(cached_payload)
            status["cached"] += 1
            continue
        response = _fetch_url_with_policy(
            GITHUB_SEARCH_URL,
            params={
                "q": query,
                "sort": "stars",
                "order": "desc",
                "per_page": per_query,
            },
            headers=headers,
            timeout=timeout,
            context=f"GitHub search for '{profile['query']}'",
            status=status,
            throttle=request_throttle,
            max_retries=max_retries,
        )
        if response is None:
            if status.get("budget_exhausted"):
                break
            status["failed"] += 1
            continue
        if response.status_code < 200 or response.status_code >= 300:
            status["failed"] += 1
            continue

        status["succeeded"] += 1
        payload = response.json()
        cached_ideas = []
        for item in payload.get("items", []):
            title = item.get("full_name") or item.get("name") or ""
            description = item.get("description") or ""
            topics = item.get("topics") or []
            summary = ". ".join(part for part in [description, " ".join(topics)] if part).strip()
            updated_at = item.get("updated_at") or ""
            year = int(updated_at[:4]) if len(updated_at) >= 4 and updated_at[:4].isdigit() else None
            relevance = float(item.get("score") or 0.0) * 100.0
            stars = int(item.get("stargazers_count") or 0)
            cached_ideas.append(
                {
                    "source": "github",
                    "query": profile["query"],
                    "title": title,
                    "summary": summary or description,
                    "url": item.get("html_url") or "",
                    "github_full_name": item.get("full_name") or "",
                    "doi": "",
                    "year": year,
                    "citations": stars,
                    "relevance": relevance,
                    "bias_families": list(profile.get("families", [])),
                    "bias_style_tags": list(profile.get("style_tags", [])),
                    "bias_horizon": profile.get("horizon", "medium"),
                    "source_score": compute_source_score(
                        year=year,
                        citations=stars,
                        relevance=relevance,
                        source="github",
                    ),
                }
            )
        ideas.extend(cached_ideas)
        if fetch_cache is not None:
            put_cached_fetch(fetch_cache, namespace="github", key=cache_key, payload=cached_ideas)
    return ideas, status


def _extract_expression_parameters(expression: str) -> list[dict]:
    parameters = []
    for match in re.finditer(r"(?<=[,()\s])(-?\d*\.?\d+)(?![a-zA-Z])", expression):
        try:
            value = float(match.group(1))
        except ValueError:
            continue
        parameters.append(
            {
                "value": value,
                "start": match.start(1),
                "end": match.end(1),
                "is_integer": value.is_integer(),
            }
        )
    return parameters


def build_expression_parameter_variants(expression: str, *, max_variants: int = 4) -> list[str]:
    variants = [expression]
    seen = {expression}
    parameters = _extract_expression_parameters(expression)
    for param in parameters[:2]:
        base_value = param["value"]
        if param["is_integer"] and 2 <= abs(base_value) <= 252:
            step = 5 if abs(base_value) <= 40 else 10
            candidate_values = [int(base_value - step), int(base_value + step)]
        elif 0.0 < abs(base_value) < 1.0:
            candidate_values = [round(base_value * 0.75, 2), round(min(1.0, base_value * 1.25), 2)]
        else:
            continue
        for candidate_value in candidate_values:
            if isinstance(candidate_value, int) and candidate_value < 2:
                continue
            replacement = str(candidate_value)
            mutated = expression[: param["start"]] + replacement + expression[param["end"] :]
            if mutated in seen:
                continue
            seen.add(mutated)
            variants.append(mutated)
            if len(variants) >= max_variants:
                return variants
    return variants


def _infer_expression_horizon(expression: str) -> str:
    numeric_values = [
        abs(item["value"])
        for item in _extract_expression_parameters(expression)
        if 1.0 <= abs(item["value"]) <= 252.0
    ]
    if not numeric_values:
        return "medium"
    max_window = max(numeric_values)
    if max_window <= 12:
        return "short"
    if max_window <= 63:
        return "medium"
    return "long"


def _guess_supported_family(expression: str, style_tags: list[str]) -> str:
    family = classify_expression_family(expression)
    if family in THESIS_INDEX:
        return family
    style_tag_set = set(style_tags)
    if "volatility" in style_tag_set:
        return "shock_response"
    if "vwap" in style_tag_set:
        return "vwap_dislocation"
    if "residual" in style_tag_set or "beta" in style_tag_set:
        return "residual_beta"
    if "volume" in style_tag_set or "liquidity" in style_tag_set:
        return "pv_divergence"
    if "reversal" in style_tag_set:
        return "reversal_conditioned"
    if "simple" in style_tag_set or "ratio_like" in style_tag_set:
        return "simple_price_patterns"
    return "technical_indicator"


def _clean_learning_text(text: str, *, max_length: int = 2200) -> str:
    cleaned = re.sub(r"```.*?```", " ", text or "", flags=re.DOTALL)
    cleaned = re.sub(r"`([^`]+)`", r"\1", cleaned)
    cleaned = re.sub(r"!\[[^\]]*\]\([^)]+\)", " ", cleaned)
    cleaned = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", cleaned)
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    cleaned = re.sub(r"[_#>*~\-]{1,}", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned[:max_length]


def extract_expression_candidates_from_text(text: str, *, max_items: int = 4) -> list[str]:
    expression_hints = ("rank(", "ts_", "zscore(", "winsorize(", "group_", "trade_when(", "ts_mean(", "ts_delta(", "ts_std_dev(")
    snippets = re.findall(r"```(?:[\w+-]+)?\n(.*?)```", text or "", flags=re.DOTALL)
    snippets.extend((text or "").splitlines())
    candidates = []
    seen = set()
    for snippet in snippets:
        for raw_line in str(snippet).splitlines():
            line = raw_line.strip().strip("`").strip()
            if not line:
                continue
            line = re.sub(r"^(?:alpha|signal)\s*=\s*", "", line, flags=re.IGNORECASE)
            line = line.rstrip(";")
            if len(line) < 12 or len(line) > 220:
                continue
            if not any(token in line for token in expression_hints):
                continue
            if line.count("(") != line.count(")"):
                continue
            if not re.fullmatch(r"[A-Za-z0-9_(),.+\-/*<>=?:\s]+", line):
                continue
            if line in seen:
                continue
            seen.add(line)
            candidates.append(line)
            if len(candidates) >= max_items:
                return candidates
    return candidates


def fetch_zip_learned_ideas(zip_path: str | Path | None, *, max_seed_ideas: int = DEFAULT_ZIP_SEED_LIMIT) -> tuple[list[dict], dict]:
    ideas = []
    status = {
        "status": "disabled" if not zip_path else "missing",
        "path": str(zip_path or ""),
        "seed_expression_count": 0,
        "operator_count": 0,
        "imported_ideas": 0,
    }
    if not zip_path:
        return ideas, status

    path = Path(zip_path)
    status["path"] = str(path)
    if not path.exists():
        return ideas, status

    status["status"] = "ok"
    try:
        with zipfile.ZipFile(path) as archive:
            operator_candidates = [
                name
                for name in archive.namelist()
                if name.lower().endswith("operatorraw.json")
            ]
            if operator_candidates:
                try:
                    operators = json.loads(archive.read(operator_candidates[0]).decode("utf-8", "ignore"))
                    if isinstance(operators, list):
                        status["operator_count"] = len(operators)
                except Exception:
                    status["operator_count"] = 0

            expression_paths = [
                name
                for name in archive.namelist()
                if name.lower().endswith("mined_expressions.json")
            ]
            if not expression_paths:
                status["status"] = "no_mined_expressions"
                return ideas, status

            raw_records = json.loads(archive.read(expression_paths[0]).decode("utf-8", "ignore"))
            if not isinstance(raw_records, list):
                status["status"] = "invalid_mined_expressions"
                return ideas, status

            for record in raw_records[: max(0, max_seed_ideas)]:
                expression = str(record.get("expression", "") if isinstance(record, dict) else "").strip()
                if not expression:
                    continue
                metrics = score_expression(expression)
                style_tags = list(metrics.get("style_tags", [])) or ["technical"]
                family_id = _guess_supported_family(expression, style_tags)
                result_settings = (record.get("result") or {}).get("settings", {}) if isinstance(record, dict) else {}
                settings_bits = [
                    f"region={result_settings.get('region')}" if result_settings.get("region") else "",
                    f"universe={result_settings.get('universe')}" if result_settings.get("universe") else "",
                    f"neutralization={result_settings.get('neutralization')}" if result_settings.get("neutralization") else "",
                ]
                settings_text = ", ".join(bit for bit in settings_bits if bit)
                seed_variants = build_expression_parameter_variants(expression)
                idea = {
                    "source": "zip_knowledge",
                    "query": f"worldquant miner learned alpha seed {family_id}",
                    "title": f"ZIP Seed: {expression[:88]}",
                    "summary": (
                        f"Learned equity alpha seed expression from worldquant-miner. "
                        f"Family hint: {family_id}. Styles: {', '.join(style_tags[:5])}. "
                        f"Original settings: {settings_text or 'not provided'}."
                    ),
                    "url": str(path),
                    "year": date.today().year,
                    "citations": 0,
                    "relevance": 980.0,
                    "bias_families": [family_id],
                    "bias_style_tags": style_tags[:6],
                    "bias_horizon": _infer_expression_horizon(expression),
                    "source_score": compute_source_score(
                        year=date.today().year,
                        citations=0,
                        relevance=980.0,
                        source="zip_knowledge",
                    ),
                    "seed_expressions": seed_variants,
                    "seed_expression_origin": expression_paths[0],
                    "seed_expression_operator_count": len(re.findall(r"\b[a-z_][a-z0-9_]*\s*\(", expression)),
                }
                ideas.append(idea)

            status["seed_expression_count"] = len(ideas)
            status["imported_ideas"] = len(ideas)
    except zipfile.BadZipFile:
        status["status"] = "bad_zip"
    except Exception:
        status["status"] = "error"
    return ideas, status


def fetch_github_readme_ideas(
    repo_ideas: list[dict],
    *,
    limit: int = DEFAULT_GITHUB_README_LIMIT,
    timeout: int = DEFAULT_PUBLIC_API_TIMEOUT,
    fetch_cache: dict | None = None,
    request_throttle: ScoutRequestThrottle | None = None,
    max_retries: int = DEFAULT_PUBLIC_API_MAX_RETRIES,
) -> tuple[list[dict], dict]:
    ideas = []
    status = {"attempted": 0, "succeeded": 0, "failed": 0, "cached": 0, "seed_expression_count": 0, "retries": 0, "quota_limited": 0, "server_errors": 0, "network_requests": 0, "skipped_budget": 0, "cooldowns": 0, "budget_exhausted": False}
    if limit <= 0:
        return ideas, status

    seen_repos = set()
    headers = _github_headers()
    candidates = sorted(
        [item for item in repo_ideas if item.get("source") == "github" and item.get("github_full_name")],
        key=lambda item: float(item.get("source_score", 0.0)),
        reverse=True,
    )

    for repo_idea in candidates:
        repo_name = repo_idea.get("github_full_name") or ""
        if not repo_name or repo_name in seen_repos:
            continue
        seen_repos.add(repo_name)
        status["attempted"] += 1
        cached_payload = get_cached_fetch(fetch_cache or {}, namespace="github_readme", key=repo_name, ttl_seconds=FETCH_CACHE_TTLS["github_readme"])
        if isinstance(cached_payload, dict):
            ideas.append(cached_payload)
            status["cached"] += 1
            status["seed_expression_count"] += len(cached_payload.get("seed_expressions", []))
            if status["attempted"] >= limit:
                break
            continue
        response = _fetch_url_with_policy(
            GITHUB_README_URL.format(repo=repo_name),
            headers=headers,
            timeout=timeout,
            context=f"GitHub README fetch for '{repo_name}'",
            status=status,
            throttle=request_throttle,
            max_retries=max_retries,
        )
        if response is None:
            if status.get("budget_exhausted"):
                break
            status["failed"] += 1
            if status["attempted"] >= limit:
                break
            continue
        if response.status_code < 200 or response.status_code >= 300:
            status["failed"] += 1
            if status["attempted"] >= limit:
                break
            continue
        try:
            payload = response.json()
            raw_content = payload.get("content") or ""
            decoded = base64.b64decode(raw_content).decode("utf-8", "ignore") if raw_content else ""
        except Exception:
            status["failed"] += 1
            if status["attempted"] >= limit:
                break
            continue

        summary = _clean_learning_text(decoded)
        if not summary:
            if status["attempted"] >= limit:
                break
            continue

        seed_expressions = extract_expression_candidates_from_text(decoded)
        learned_idea = {
                "source": "github_readme",
                "query": repo_idea.get("query"),
                "title": f"{repo_name} README",
                "summary": summary,
                "url": payload.get("html_url") or repo_idea.get("url") or "",
                "github_full_name": repo_name,
                "year": repo_idea.get("year"),
                "citations": int(repo_idea.get("citations") or 0),
                "relevance": float(repo_idea.get("relevance") or 0.0),
                "bias_families": list(repo_idea.get("bias_families", [])),
                "bias_style_tags": list(repo_idea.get("bias_style_tags", [])),
                "bias_horizon": repo_idea.get("bias_horizon", "medium"),
                "source_score": compute_source_score(
                    year=repo_idea.get("year"),
                    citations=int(repo_idea.get("citations") or 0),
                    relevance=float(repo_idea.get("relevance") or 0.0),
                    source="github_readme",
                ),
                "seed_expressions": seed_expressions,
            }
        ideas.append(learned_idea)
        status["succeeded"] += 1
        status["seed_expression_count"] += len(seed_expressions)
        if fetch_cache is not None:
            put_cached_fetch(fetch_cache, namespace="github_readme", key=repo_name, payload=learned_idea)
        if status["attempted"] >= limit:
            break
    return ideas, status


def build_fallback_ideas() -> list[dict]:
    ideas = []
    for thesis in THESIS_LIBRARY:
        style_tags = sorted({tag for variant in thesis.get("variants", []) for tag in variant.get("style_tags", [])})
        ideas.append(
            {
                "source": "local_library",
                "query": thesis["label"],
                "title": thesis["label"],
                "summary": thesis["why"],
                "url": "",
                "year": None,
                "citations": 0,
                "relevance": 0.0,
                "bias_families": [thesis["id"]],
                "bias_style_tags": style_tags,
                "bias_horizon": "medium",
                "source_score": 0.24,
            }
        )
    return ideas


def build_book_ideas() -> list[dict]:
    ideas = []
    for item in BOOK_IDEA_BLUEPRINTS:
        ideas.append(
            {
                "source": "finding_alphas_book",
                "query": item["title"],
                "title": item["title"],
                "summary": item["summary"],
                "url": "local:WorldQuant_FindingAlphas.pdf",
                "year": 2015,
                "citations": 0,
                "relevance": 1000.0,
                "bias_families": list(item["bias_families"]),
                "bias_style_tags": list(item["bias_style_tags"]),
                "bias_horizon": item["bias_horizon"],
                "source_score": 0.86,
            }
        )
    return ideas


def dedupe_ideas(ideas: Iterable[dict]) -> list[dict]:
    deduped = []
    seen = set()
    for idea in ideas:
        key = source_key_from_idea(idea)
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(idea)
    return deduped


def _count_term_hits(text: str, terms: Iterable[str]) -> int:
    return sum(1 for term in terms if term in text)


def infer_idea_profile(idea: dict, scout_memory: dict | None = None) -> dict:
    scout_memory = scout_memory or {}
    is_github_source = idea.get("source") in {"github", "github_readme"}
    source_text = normalize_text(" ".join([idea.get("title", ""), idea.get("summary", "")]))
    query_text = normalize_text(idea.get("query", ""))
    source_key = source_key_from_idea(idea)

    family_scores = Counter({family_id: 0.25 for family_id in idea.get("bias_families", [])})
    style_tags = set(idea.get("bias_style_tags", []))
    reasoning_bits = []

    equity_hits = _count_term_hits(source_text, ("equity", "equities", "stock", "stocks", "cross-sectional", "cross sectional", "return", "returns", "anomaly", "factor", "alpha"))
    equity_hits += 0.5 * _count_term_hits(query_text, ("equity", "equities", "stock", "stocks", "anomaly", "factor"))
    broad_hits = _count_term_hits(source_text, ("mutual fund", "asset allocation", "portfolio choice", "fund performance", "household", "benchmark", "strategic asset allocation"))
    non_equity_hits = _count_term_hits(source_text, ("bond", "fixed income", "commodity", "crypto", "forex", "currency"))
    source_theme_hits = 0
    query_theme_hits = 0

    def source_match_any(*terms: str) -> bool:
        return any(term in source_text for term in terms)

    def query_match_any(*terms: str) -> bool:
        return any(term in query_text for term in terms)

    if source_match_any("momentum", "trend", "persistence", "continuation", "relative strength"):
        family_scores["technical_indicator"] += 2.0
        style_tags.update({"momentum", "trend", "technical"})
        reasoning_bits.append("paper nghieng ve momentum/trend")
        source_theme_hits += 1
    elif query_match_any("momentum", "trend", "persistence", "continuation", "relative strength"):
        family_scores["technical_indicator"] += 0.35
        query_theme_hits += 1
    if source_match_any("reversal", "mean reversion", "contrarian", "overreaction", "underreaction"):
        family_scores["reversal_conditioned"] += 2.0
        family_scores["vwap_dislocation"] += 1.0
        style_tags.update({"reversal", "technical"})
        reasoning_bits.append("paper nghieng ve mean reversion")
        source_theme_hits += 1
    elif query_match_any("reversal", "mean reversion", "contrarian"):
        family_scores["reversal_conditioned"] += 0.35
        family_scores["vwap_dislocation"] += 0.18
        query_theme_hits += 1
    if source_match_any("volume", "liquidity", "order flow", "trading activity", "turnover", "flow imbalance"):
        family_scores["pv_divergence"] += 1.8
        family_scores["shock_response"] += 1.0
        style_tags.update({"volume", "liquidity"})
        reasoning_bits.append("paper co dau hieu volume/liquidity")
        source_theme_hits += 1
    elif query_match_any("volume", "liquidity", "order flow"):
        family_scores["pv_divergence"] += 0.3
        family_scores["shock_response"] += 0.18
        query_theme_hits += 1
    if source_match_any("vwap", "intraday", "microstructure", "dislocation"):
        family_scores["vwap_dislocation"] += 2.0
        style_tags.update({"vwap", "liquidity"})
        reasoning_bits.append("paper co mau intraday/vwap")
        source_theme_hits += 1
    elif query_match_any("vwap", "intraday", "microstructure", "dislocation"):
        family_scores["vwap_dislocation"] += 0.35
        query_theme_hits += 1
    if source_match_any("volatility", "variance", "shock", "dispersion", "risk regime"):
        family_scores["shock_response"] += 1.9
        style_tags.update({"volatility"})
        reasoning_bits.append("paper nghieng ve volatility/shock")
        source_theme_hits += 1
    elif query_match_any("volatility", "variance", "shock", "dispersion"):
        family_scores["shock_response"] += 0.35
        query_theme_hits += 1
    if source_match_any("beta", "residual", "idiosyncratic", "market neutral", "systematic risk"):
        family_scores["residual_beta"] += 2.0
        style_tags.update({"residual", "beta", "correlation"})
        reasoning_bits.append("paper co thesis residual/beta")
        source_theme_hits += 1
    elif query_match_any("beta", "residual", "idiosyncratic", "market neutral"):
        family_scores["residual_beta"] += 0.35
        query_theme_hits += 1
    if source_match_any("correlation", "co-movement", "co movement"):
        family_scores["pv_divergence"] += 0.9
        family_scores["residual_beta"] += 0.7
        style_tags.update({"correlation"})
        source_theme_hits += 1
    if source_match_any("ratio-like", "ratio like", "compare current values to historical values", "historical values", "simple hypothesis", "simple expression"):
        family_scores["simple_price_patterns"] += 1.9
        style_tags.update({"simple", "ratio_like", "book_alpha_design"})
        reasoning_bits.append("paper nhan manh bien doi ratio-like va simple hypothesis")
        source_theme_hits += 1
    elif query_match_any("ratio like", "simple price hypothesis"):
        family_scores["simple_price_patterns"] += 0.4
        query_theme_hits += 1
    if source_match_any("price-delay", "price delay", "delay(price", "lagged price", "inverse price", "1/price"):
        family_scores["simple_price_patterns"] += 2.0
        style_tags.update({"simple", "ratio_like", "reversal", "trend", "book_alpha_design"})
        reasoning_bits.append("paper dua ra cac mau price-delay / inverse-price")
        source_theme_hits += 1
    elif query_match_any("price delay", "inverse price"):
        family_scores["simple_price_patterns"] += 0.4
        query_theme_hits += 1

    github_theme_support = source_theme_hits
    if is_github_source and query_theme_hits:
        github_theme_support = source_theme_hits + min(1.0, 0.75 * query_theme_hits)
    if source_match_any("rank", "ranking", "z-scoring", "z-scoring", "winsorization", "winsorizing", "truncation", "decay", "neutralization"):
        family_scores["simple_price_patterns"] += 0.8
        family_scores["technical_indicator"] += 0.4
        style_tags.update({"rank", "normalization", "winsorize", "book_alpha_design"})
        reasoning_bits.append("paper nhan manh rank / decay / neutralization / winsorize")
        source_theme_hits += 1

    theme_strength = max(family_scores.values(), default=0.0)
    specificity_score = _clip(0.08 + 0.10 * len(style_tags) + 0.10 * source_theme_hits + 0.04 * theme_strength - 0.12 * broad_hits, 0.0, 1.0)
    equity_factor_relevance = _clip(0.06 + 0.08 * equity_hits + 0.08 * source_theme_hits + 0.04 * theme_strength - 0.16 * broad_hits - 0.20 * non_equity_hits, 0.0, 1.0)

    if source_key in set(scout_memory.get("blocked_source_keys", [])):
        specificity_score = _clip(specificity_score - 0.15, 0.0, 1.0)
        equity_factor_relevance = _clip(equity_factor_relevance - 0.12, 0.0, 1.0)

    horizon = idea.get("bias_horizon", "medium")
    if source_match_any("intraday", "short-term", "1-day", "daily", "weekly"):
        horizon = "short"
    elif source_match_any("quarterly", "persistent", "long-term", "12-month", "annual"):
        horizon = "long"

    ranked_families = [family_id for family_id, score in family_scores.most_common() if score >= 1.2]
    if not ranked_families:
        ranked_families = list(idea.get("bias_families", [])) or ["technical_indicator"]

    source_quality_score = _clip(
        0.40 * float(idea.get("source_score", 0.0))
        + 0.35 * equity_factor_relevance
        + 0.25 * specificity_score,
        0.0,
        1.0,
    )
    if idea.get("source") == "finding_alphas_book":
        source_quality_score = _clip(source_quality_score + 0.10, 0.0, 1.0)
    if is_github_source:
        generation_ok = (
            source_quality_score >= 0.44
            and specificity_score >= 0.38
            and equity_factor_relevance >= 0.24
            and github_theme_support >= 0.75
        )
    else:
        generation_ok = source_quality_score >= 0.46 and (equity_factor_relevance >= 0.36 or source_theme_hits >= 2)
    if source_theme_hits == 0 and (not is_github_source or github_theme_support < 0.75):
        generation_ok = False
    if broad_hits and source_theme_hits < 2:
        generation_ok = False
    if non_equity_hits >= 2:
        generation_ok = False

    return {
        "families": ranked_families[:2],
        "family_scores": {family_id: _round(score) for family_id, score in family_scores.items()},
        "style_tags": sorted(style_tags),
        "horizon": horizon,
        "reasoning": ", ".join(reasoning_bits) if reasoning_bits else "paper duoc map sang supported template library bang weighted heuristic",
        "source_quality_score": _round(source_quality_score),
        "specificity_score": _round(specificity_score),
        "equity_factor_relevance": _round(equity_factor_relevance),
        "theme_strength": _round(theme_strength),
        "generation_ok": generation_ok,
        "source_key": source_key,
    }


def filter_relevant_ideas(ideas: list[dict], scout_memory: dict) -> list[dict]:
    filtered = []
    for idea in ideas:
        profile = infer_idea_profile(idea, scout_memory=scout_memory)
        if not profile["generation_ok"]:
            continue
        enriched = dict(idea)
        enriched["idea_profile"] = profile
        filtered.append(enriched)
    return filtered


def adjust_horizon_from_memory(horizon: str, family_id: str, memory: dict, scout_memory: dict) -> str:
    preferred_horizon = scout_memory.get("preferred_family_horizons", {}).get(family_id)
    if preferred_horizon:
        horizon = preferred_horizon

    failure_counts = memory.get("failure_counts", {})
    high_turnover = int(failure_counts.get("HIGH_TURNOVER", 0))
    low_turnover = int(failure_counts.get("LOW_TURNOVER", 0))
    order = ["short", "medium", "long"]
    index = order.index(horizon)
    if high_turnover >= low_turnover + 2 and index < 2:
        return order[index + 1]
    if low_turnover >= high_turnover + 2 and index > 0:
        return order[index - 1]
    return horizon


def retarget_token(token: str, horizon: str) -> str:
    for prefix, mapping in WINDOW_GROUPS.items():
        if token.startswith(f"{prefix}_"):
            return mapping[horizon]
    return token


def _validated_variants(candidates: list[list[str]]) -> list[list[str]]:
    deduped = []
    seen = set()
    for candidate in candidates:
        key = tuple(candidate)
        if key in seen:
            continue
        try:
            validate_token_program(candidate)
        except Exception:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped


def _append_if_missing(program: list[str], token: str) -> list[str]:
    if token in program:
        return list(program)
    return list(program) + [token]


def mutate_token_program(token_program: list[str], *, family_id: str, horizon: str, style_tags: list[str]) -> list[list[str]]:
    base = list(token_program)
    current = [retarget_token(token, horizon) for token in base]
    shorter = [retarget_token(token, "short") for token in base]
    longer = [retarget_token(token, "long") for token in base]
    surprise_token = WINDOW_GROUPS["SURPRISE"][horizon]
    tsz_token = WINDOW_GROUPS["TSZ"][horizon]

    variants = [base, current]
    if horizon != "short":
        variants.append(shorter)
    if horizon != "long":
        variants.append(longer)

    variants.append(_append_if_missing(current, "RANK"))
    variants.append(_append_if_missing(current, "WINSORIZE"))
    variants.append(_append_if_missing(current, "ZSCORE"))
    if not any(token.startswith("SURPRISE_") for token in current):
        variants.append(current + [surprise_token, "RANK"])

    if family_id in {"technical_indicator", "reversal_conditioned", "vwap_dislocation"}:
        variants.append(current + ["TSZ_21", "RANK"])
        variants.append(longer + ["WINSORIZE", "RANK"])
    elif family_id in {"pv_divergence", "shock_response"}:
        variants.append(current + ["WINSORIZE", "RANK"])
        variants.append(shorter + ["TSZ_21", "RANK"])
    elif family_id == "residual_beta":
        variants.append(current + ["ZSCORE", "RANK"])
        variants.append(longer + ["WINSORIZE", "RANK"])
    elif family_id == "simple_price_patterns":
        variants.append(current + [tsz_token, "RANK"])
        variants.append(current + [surprise_token, "WINSORIZE", "RANK"])
        variants.append(longer + ["WINSORIZE", "RANK"])

    if "volatility" in style_tags:
        variants.append(current + ["WINSORIZE", "RANK"])
    if "normalization" not in style_tags:
        variants.append(current + ["TSZ_21", "RANK"])
    if "ratio_like" not in style_tags:
        variants.append(current + [surprise_token, "RANK"])
    if "book_alpha_design" in style_tags:
        variants.append(current + [surprise_token, tsz_token, "RANK"])
    if "reversal" in style_tags and "momentum" in style_tags:
        variants.append(shorter + ["WINSORIZE", "RANK"])

    return _validated_variants(variants)


def build_blend_programs(idea: dict, families: list[str], horizons: dict[str, str]) -> list[tuple[list[str], dict]]:
    if len(families) < 2:
        return []
    first, second = families[:2]
    first_thesis = THESIS_INDEX.get(first)
    second_thesis = THESIS_INDEX.get(second)
    if not first_thesis or not second_thesis:
        return []

    first_program = [retarget_token(token, horizons[first]) for token in first_thesis["variants"][0]["token_program"]]
    second_program = [retarget_token(token, horizons[second]) for token in second_thesis["variants"][0]["token_program"]]
    candidates = [
        (first_program + second_program + ["ADD", "RANK"], {"families": [first, second], "label": f"{first_thesis['label']} + {second_thesis['label']} blend"}),
        (first_program + second_program + ["MUL", "WINSORIZE", "RANK"], {"families": [first, second], "label": f"{first_thesis['label']} x {second_thesis['label']} blend"}),
    ]

    blended = []
    for token_program, metadata in candidates:
        try:
            validate_token_program(token_program)
        except Exception:
            continue
        blended.append((token_program, metadata))
    return blended


def recommend_settings_profiles(family_ids: list[str], horizon: str, style_tags: list[str]) -> list[str]:
    primary_family = family_ids[0] if family_ids else "technical_indicator"
    region, universe, delay, decay, truncation, neutralization = FAMILY_SETTINGS_BASE.get(
        primary_family,
        ("USA", "TOP3000", 1, 5, 0.05, "Market"),
    )

    if horizon == "short":
        decay = max(2, decay - 2)
        truncation = min(truncation, 0.03)
    elif horizon == "long":
        decay = min(10, decay + 2)

    profiles = [
        (region, universe, delay, decay, truncation, neutralization),
        (region, universe, delay, min(10, decay + 2), max(0.01, truncation - 0.01), "Subindustry" if neutralization != "Subindustry" else neutralization),
        (region, universe, delay, max(1, decay - 1), min(0.08, truncation + 0.01), neutralization),
    ]

    if "volatility" in style_tags or "liquidity" in style_tags:
        profiles.append((region, universe, delay, min(10, decay + 1), max(0.01, truncation - 0.01), "Industry"))
    if primary_family == "residual_beta":
        profiles.append(("USA", "TOP3000", 1, min(10, decay + 1), 0.05, "Industry"))
    if primary_family == "vwap_dislocation":
        profiles.append(("USA", "TOP200", 1, max(2, decay - 1), 0.02, "Subindustry"))

    formatted = []
    seen = set()
    for item in profiles:
        settings_text = (
            f"{item[0]}, {item[1]}, Decay {item[3]}, Delay {item[2]}, "
            f"Truncation {item[4]:.2f}, Neutralization {item[5]}"
        )
        if settings_text in seen:
            continue
        seen.add(settings_text)
        formatted.append(settings_text)
    return formatted[:4]


def _verdict_rank(verdict: str) -> int:
    return {"PASS": 3, "LIKELY_PASS": 2, "BORDERLINE": 1, "FAIL": 0}.get(verdict, 0)


def _confidence_rank(confidence: str) -> int:
    return {"HIGH": 3, "MEDIUM": 2, "LOW": 1}.get(confidence, 0)


def evaluate_expression_across_settings(
    expression: str,
    *,
    settings_profiles: list[str],
    history_index: HistoryIndex,
) -> tuple[dict, list[dict]]:
    evaluations = []
    best_result = None
    best_key = None

    try:
        results = score_expressions_batch(
            [expression] * len(settings_profiles),
            history_index=history_index,
            settings_list=settings_profiles,
        )
    except Exception:
        results = [
            score_expression(expression, history_index=history_index, settings=settings)
            for settings in settings_profiles
        ]

    for settings, result in zip(settings_profiles, results):
        evaluations.append(
            {
                "settings": settings,
                "verdict": result["verdict"],
                "alpha_score": result["alpha_score"],
                "fitness": result["fitness"],
                "sharpe": result["sharpe"],
                "uniqueness_proxy": result["uniqueness_proxy"],
            }
        )
        current_key = (
            _verdict_rank(result["verdict"]),
            float(result["alpha_score"]),
            _confidence_rank(result["confidence"]),
            float(result["fitness"]),
            float(result["uniqueness_proxy"]),
            float(result["sharpe"]),
        )
        if best_result is None or current_key > best_key:
            best_result = result
            best_key = current_key

    return best_result, evaluations


def summarize_settings_robustness(evaluations: list[dict]) -> dict:
    if not evaluations:
        return {
            "settings_count": 0,
            "pass_rate": 0.0,
            "avg_alpha_score": 0.0,
            "min_alpha_score": 0.0,
            "max_alpha_score": 0.0,
            "alpha_spread": 0.0,
            "robustness_score": 0.0,
        }

    verdict_weights = {"PASS": 1.0, "LIKELY_PASS": 0.72, "BORDERLINE": 0.40, "FAIL": 0.0}
    alpha_scores = [float(item.get("alpha_score", 0.0)) for item in evaluations]
    verdict_strength = sum(verdict_weights.get(item.get("verdict"), 0.0) for item in evaluations) / len(evaluations)
    pass_rate = sum(1 for item in evaluations if item.get("verdict") in {"PASS", "LIKELY_PASS"}) / len(evaluations)
    avg_alpha_score = sum(alpha_scores) / len(alpha_scores)
    min_alpha_score = min(alpha_scores)
    max_alpha_score = max(alpha_scores)
    alpha_spread = max_alpha_score - min_alpha_score
    consistency_component = 1.0 - _clip(alpha_spread / 24.0, 0.0, 1.0)
    floor_component = _clip(min_alpha_score / 100.0, 0.0, 1.0)
    robustness_score = _clip(
        0.30 * verdict_strength
        + 0.26 * pass_rate
        + 0.22 * _clip(avg_alpha_score / 100.0, 0.0, 1.0)
        + 0.10 * floor_component
        + 0.12 * consistency_component,
        0.0,
        1.0,
    )

    return {
        "settings_count": len(evaluations),
        "pass_rate": _round(pass_rate),
        "avg_alpha_score": _round(avg_alpha_score),
        "min_alpha_score": _round(min_alpha_score),
        "max_alpha_score": _round(max_alpha_score),
        "alpha_spread": _round(alpha_spread),
        "robustness_score": _round(robustness_score),
    }


def _source_history_adjustment(source_key: str, family_id: str, scout_memory: dict) -> float:
    source_stat = scout_memory.get("source_stats", {}).get(source_key, {})
    pair_stat = scout_memory.get("source_family_stats", {}).get(f"{source_key}|{family_id}", {})
    adjustment = 0.0
    if source_stat.get("strong_count", 0) >= 1:
        adjustment += 0.05
    elif source_stat.get("attempts", 0) >= 2 and source_stat.get("selected_count", 0) == 0:
        adjustment -= 0.10
    if pair_stat.get("strong_count", 0) >= 1:
        adjustment += 0.04
    elif pair_stat.get("attempts", 0) >= 2 and pair_stat.get("selected_count", 0) == 0:
        adjustment -= 0.08
    return adjustment


def _family_history_snapshot(family_id: str, scout_memory: dict, *, horizon: str | None = None) -> dict:
    stats = scout_memory.get("family_horizon_stats", {})
    matches = []
    for key, stat in stats.items():
        base_family, _, base_horizon = key.partition("|")
        if base_family != family_id:
            continue
        if horizon and base_horizon and base_horizon != horizon:
            continue
        matches.append(stat)

    if not matches and horizon is not None:
        return _family_history_snapshot(family_id, scout_memory, horizon=None)
    if not matches:
        return {
            "attempts": 0,
            "selected_count": 0,
            "watchlist_count": 0,
            "strong_count": 0,
            "avg_alpha_score": 0.0,
            "avg_confidence_score": 0.0,
        }

    attempts = sum(int(item.get("attempts", 0) or 0) for item in matches)
    selected_count = sum(int(item.get("selected_count", 0) or 0) for item in matches)
    watchlist_count = sum(int(item.get("watchlist_count", 0) or 0) for item in matches)
    strong_count = sum(int(item.get("strong_count", 0) or 0) for item in matches)
    total_alpha_weight = sum(float(item.get("avg_alpha_score", 0.0) or 0.0) * int(item.get("attempts", 0) or 0) for item in matches)
    total_conf_weight = sum(float(item.get("avg_confidence_score", 0.0) or 0.0) * int(item.get("attempts", 0) or 0) for item in matches)
    return {
        "attempts": attempts,
        "selected_count": selected_count,
        "watchlist_count": watchlist_count,
        "strong_count": strong_count,
        "avg_alpha_score": _round(total_alpha_weight / attempts) if attempts else 0.0,
        "avg_confidence_score": _round(total_conf_weight / attempts) if attempts else 0.0,
    }


def _select_generation_families(
    idea_profile: dict,
    *,
    scout_memory: dict,
    brain_feedback_context: dict | None = None,
) -> list[str]:
    brain_feedback_context = brain_feedback_context or {}
    family_scores = Counter(idea_profile.get("family_scores", {}))
    families = list(idea_profile.get("families", []))
    if not families:
        families = ["technical_indicator"]
    horizon = idea_profile.get("horizon", "medium")
    style_tags = set(idea_profile.get("style_tags", []))
    reactive_theme = bool(style_tags & {"reversal", "volume", "vwap", "technical", "momentum", "trend"})
    crowded_reactive = [
        family_id
        for family_id in families
        if family_id in REACTIVE_FAMILIES
        and _family_history_snapshot(family_id, scout_memory, horizon=horizon).get("attempts", 0) >= 2
    ]
    family_feedback = brain_feedback_context.get("family_feedback", {})
    preferred_contrasts = []
    for family_id in ("residual_beta", "shock_response"):
        if family_id in families or family_id not in THESIS_INDEX:
            continue
        feedback = family_feedback.get(family_id, {})
        if feedback.get("discouraged"):
            continue
        history = _family_history_snapshot(family_id, scout_memory, horizon=horizon)
        contrast_bonus = 0.0
        if family_id == "residual_beta" and reactive_theme:
            contrast_bonus += 0.36
        if family_id == "shock_response" and style_tags & {"reversal", "volume", "liquidity", "technical"}:
            contrast_bonus += 0.26
        if history.get("attempts", 0) == 0:
            contrast_bonus += 0.10
        elif history.get("strong_count", 0) >= 1:
            contrast_bonus += 0.06
        preferred_contrasts.append((contrast_bonus, family_id))

    if crowded_reactive or (reactive_theme and not (set(families) & CONTRAST_FAMILIES)):
        for _, family_id in sorted(preferred_contrasts, reverse=True):
            if family_id not in families:
                families.append(family_id)
            if len(families) >= 3:
                break

    def _priority(family_id: str) -> tuple[float, float, int, str]:
        history = _family_history_snapshot(family_id, scout_memory, horizon=horizon)
        feedback = family_feedback.get(family_id, {})
        contrast_bonus = 0.10 if family_id in CONTRAST_FAMILIES and reactive_theme else 0.0
        history_bonus = 0.04 if history.get("strong_count", 0) >= 1 else 0.0
        freshness_bonus = 0.06 if history.get("attempts", 0) == 0 else 0.0
        crowd_penalty = 0.08 if history.get("attempts", 0) >= 3 and history.get("strong_count", 0) == 0 else 0.0
        discouraged_penalty = 0.12 * float(feedback.get("severity", 0.0) or 0.0) if feedback.get("discouraged") else 0.0
        preferred_bonus = 0.05 * float(feedback.get("preference", 0.0) or 0.0) if feedback.get("preferred") else 0.0
        score = float(family_scores.get(family_id, 0.0)) + contrast_bonus + history_bonus + freshness_bonus + preferred_bonus - crowd_penalty - discouraged_penalty
        return (
            _round(score),
            float(history.get("avg_alpha_score", 0.0)),
            int(history.get("strong_count", 0)),
            family_id,
        )

    ranked = sorted(dict.fromkeys(families), key=_priority, reverse=True)
    return ranked[:3]


def compute_family_diversity_profile(
    family_components: list[str],
    *,
    horizon: str,
    style_tags: Iterable[str],
    scout_memory: dict,
    brain_feedback_context: dict | None = None,
) -> dict:
    brain_feedback_context = brain_feedback_context or {}
    style_tags = set(style_tags)
    score = 0.52
    reasons = []
    crowded_families = []
    fresh_families = []
    family_feedback = brain_feedback_context.get("family_feedback", {})
    family_set = set(family_components)
    reactive_theme = bool(style_tags & {"reversal", "volume", "vwap", "technical", "momentum", "trend"})

    for family_id in family_components:
        history = _family_history_snapshot(family_id, scout_memory, horizon=horizon)
        feedback = family_feedback.get(family_id, {})
        if history.get("attempts", 0) == 0:
            score += 0.06
            fresh_families.append(family_id)
            reasons.append(f"fresh_family={family_id}")
        elif history.get("attempts", 0) >= 3 and history.get("strong_count", 0) == 0:
            score -= 0.08
            crowded_families.append(family_id)
            reasons.append(f"crowded_family={family_id}")
        elif history.get("strong_count", 0) >= 1:
            score += 0.03

        if feedback.get("discouraged"):
            severity = float(feedback.get("severity", 0.0) or 0.0)
            score -= 0.10 + (0.06 * severity)
            crowded_families.append(family_id)
            reasons.append(f"discouraged_family={family_id}")
        elif feedback.get("preferred"):
            score += 0.04

    if reactive_theme and family_set & CONTRAST_FAMILIES:
        score += 0.10
        reasons.append("reactive_theme_with_contrast_family")
    elif reactive_theme and family_set and family_set <= REACTIVE_FAMILIES:
        score -= 0.08
        reasons.append("reactive_theme_without_contrast_family")

    if len(family_set) >= 2 and (family_set & CONTRAST_FAMILIES):
        score += 0.05
    if family_set & CONTRAST_FAMILIES:
        score += 0.04

    score = _clip(score, 0.0, 1.0)
    return {
        "score": _round(score),
        "reasons": list(dict.fromkeys(reasons))[:6],
        "crowded_families": sorted(set(crowded_families)),
        "fresh_families": sorted(set(fresh_families)),
    }


def infer_data_categories(style_tags: Iterable[str]) -> list[str]:
    categories = set()
    tags = set(style_tags)
    if tags & {"simple", "ratio_like", "reversal", "momentum", "trend", "technical"}:
        categories.add("price")
    if tags & {"volume", "liquidity", "vwap"}:
        categories.add("flow")
    if tags & {"residual", "beta", "volatility"}:
        categories.add("risk")
    if tags & {"correlation"}:
        categories.add("dependence")
    return sorted(categories)


def summarize_surrogate_shadow(result: dict) -> dict:
    shadow = result.get("surrogate_shadow") or {}
    summary = {
        "status": shadow.get("status", "unavailable"),
        "preview_verdict": shadow.get("preview_verdict", "UNAVAILABLE"),
        "alignment": shadow.get("alignment", "unknown"),
        "penalty_score": 0.0,
        "confidence_drag": 0.0,
        "reasons": [],
        "hard_signal": "none",
    }
    if summary["status"] != "ready":
        return summary

    local_verdict = str(result.get("verdict", "") or "").upper()
    if local_verdict not in {"PASS", "LIKELY_PASS"}:
        return summary

    heuristic_fitness = coerce_float(result.get("fitness")) or 0.0
    heuristic_sharpe = coerce_float(result.get("sharpe")) or 0.0
    predicted_fitness = coerce_float(shadow.get("predicted_fitness"))
    predicted_sharpe = coerce_float(shadow.get("predicted_sharpe"))
    if predicted_fitness is None or predicted_sharpe is None:
        return summary

    preview_verdict = str(summary["preview_verdict"] or "UNAVAILABLE").upper()
    alignment = str(summary["alignment"] or "unknown")
    fitness_gap = max(0.0, heuristic_fitness - predicted_fitness)
    sharpe_gap = max(0.0, heuristic_sharpe - predicted_sharpe)
    fitness_gap_component = _clip(fitness_gap / 1.40, 0.0, 1.0)
    sharpe_gap_component = _clip(sharpe_gap / 1.80, 0.0, 1.0)

    penalty = 0.0
    reasons = []
    if alignment == "more_cautious":
        penalty += 0.04
        reasons.append("surrogate_more_cautious")
    elif alignment == "mixed":
        penalty += 0.02
        reasons.append("surrogate_mixed_signal")

    if preview_verdict == "FAIL":
        penalty += 0.09
        reasons.append("surrogate_preview_fail")
    elif preview_verdict == "BORDERLINE":
        penalty += 0.05
        reasons.append("surrogate_preview_borderline")
    elif preview_verdict == "LIKELY_PASS" and local_verdict == "PASS":
        penalty += 0.02
        reasons.append("surrogate_less_confident_than_heuristic")

    if fitness_gap_component >= 0.25:
        penalty += 0.04 * fitness_gap_component
        reasons.append(f"fitness_gap={_round(fitness_gap)}")
    if sharpe_gap_component >= 0.25:
        penalty += 0.03 * sharpe_gap_component
        reasons.append(f"sharpe_gap={_round(sharpe_gap)}")

    penalty = _clip(penalty, 0.0, SURROGATE_SHADOW_MAX_PENALTY)
    confidence_drag = _clip(
        (0.65 * penalty) + (0.02 * fitness_gap_component) + (0.02 * sharpe_gap_component),
        0.0,
        SURROGATE_SHADOW_MAX_CONFIDENCE_DRAG,
    )

    hard_signal = "none"
    if preview_verdict == "FAIL" and alignment == "more_cautious" and (fitness_gap_component + sharpe_gap_component) >= 1.10:
        hard_signal = "severe_mismatch"
    elif penalty >= 0.10:
        hard_signal = "soft_mismatch"

    summary.update(
        {
            "penalty_score": _round(penalty),
            "confidence_drag": _round(confidence_drag),
            "reasons": list(dict.fromkeys(reasons)),
            "hard_signal": hard_signal,
            "predicted_fitness": predicted_fitness,
            "predicted_sharpe": predicted_sharpe,
            "fitness_gap": _round(fitness_gap),
            "sharpe_gap": _round(sharpe_gap),
        }
    )
    return summary


def build_risk_tags(
    result: dict,
    source_quality_score: float,
    *,
    data_category_count: int,
    settings_robustness: dict | None = None,
    submitted_overlap: dict | None = None,
    brain_feedback: dict | None = None,
    surrogate_shadow: dict | None = None,
    scout_blocked_skeleton: bool = False,
) -> list[str]:
    settings_robustness = settings_robustness or {}
    submitted_overlap = submitted_overlap or {}
    brain_feedback = brain_feedback or {}
    surrogate_shadow = surrogate_shadow or {}
    risk_tags = []
    if result.get("HIGH_TURNOVER") == "FAIL":
        risk_tags.append("turnover_risk")
    if result.get("CONCENTRATED_WEIGHT") == "FAIL":
        risk_tags.append("weight_risk")
    if result.get("OUT_OF_SAMPLE_ALIGNMENT") == "FAIL":
        risk_tags.append("out_of_sample_risk")
    if brain_feedback.get("exact_skeleton_match"):
        risk_tags.append("brain_exact_skeleton_risk")
    if float(brain_feedback.get("penalty_score", 0.0)) >= 0.16:
        risk_tags.append("brain_feedback_risk")
    if result.get("SELF_CORRELATION") == "FAIL" or result.get("MATCHES_COMPETITION") == "FAIL":
        risk_tags.append("similarity_risk")
    if result.get("verdict") in {"BORDERLINE", "FAIL"} or source_quality_score < 0.48:
        risk_tags.append("unproven_style")
    if data_category_count > 3:
        risk_tags.append("category_overload_risk")
    if settings_robustness.get("settings_count", 0) > 1 and settings_robustness.get("robustness_score", 0.0) < 0.55:
        risk_tags.append("fragile_across_settings")
    if submitted_overlap.get("exact_skeleton_match"):
        risk_tags.append("submitted_skeleton_risk")
    if submitted_overlap.get("overlap_score", 0.0) >= 0.58:
        risk_tags.append("submitted_overlap_risk")
    if scout_blocked_skeleton:
        risk_tags.append("scout_blocked_skeleton_risk")
    if surrogate_shadow.get("hard_signal") == "severe_mismatch":
        risk_tags.append("surrogate_shadow_risk")
    return sorted(set(risk_tags))


def compute_submitted_overlap(expression: str, result: dict, submitted_context: dict | None) -> dict:
    submitted_context = submitted_context or {}
    if not submitted_context.get("count"):
        return {
            "overlap_score": 0.0,
            "diversity_score": 1.0,
            "exact_skeleton_match": False,
            "alpha_type_count": 0,
            "matched_style_tags": [],
            "reasons": [],
        }

    skeleton = expression_skeleton(expression)
    exact_skeleton_match = skeleton in set(submitted_context.get("skeletons", []))
    alpha_type = result.get("alpha_type", "unknown")
    alpha_type_count = int(submitted_context.get("alpha_type_counts", {}).get(alpha_type, 0))
    alpha_type_pressure = min(1.0, alpha_type_count / 3.0)
    style_tags = set(result.get("style_tags", []))
    dominant_style_tags = set(submitted_context.get("dominant_style_tags", []))
    matched_style_tags = sorted(style_tags & dominant_style_tags)
    style_overlap_score = min(1.0, len(matched_style_tags) / 3.0)
    overlap_score = 0.0
    if exact_skeleton_match:
        overlap_score += 0.45
    overlap_score += 0.30 * alpha_type_pressure
    overlap_score += 0.20 * style_overlap_score
    if alpha_type in set(submitted_context.get("saturated_alpha_types", [])) and matched_style_tags:
        overlap_score += 0.15
    overlap_score = _clip(overlap_score, 0.0, 1.0)

    reasons = []
    if exact_skeleton_match:
        reasons.append("submitted_skeleton_match")
    if alpha_type_count:
        reasons.append(f"submitted_alpha_type={alpha_type}x{alpha_type_count}")
    if matched_style_tags:
        reasons.append(f"submitted_style_overlap={','.join(matched_style_tags[:4])}")

    return {
        "overlap_score": _round(overlap_score),
        "diversity_score": _round(1.0 - overlap_score),
        "exact_skeleton_match": exact_skeleton_match,
        "alpha_type_count": alpha_type_count,
        "matched_style_tags": matched_style_tags,
        "reasons": reasons,
    }


def compute_submitted_contrast(result: dict, submitted_context: dict | None) -> dict:
    submitted_context = submitted_context or {}
    if not submitted_context.get("count"):
        return {
            "contrast_score": 0.5,
            "fresh_style_tags": [],
            "matched_style_tags": [],
            "reasons": [],
        }

    alpha_type = result.get("alpha_type", "unknown")
    alpha_type_count = int(submitted_context.get("alpha_type_counts", {}).get(alpha_type, 0))
    style_tags = set(result.get("style_tags", []))
    dominant_style_tags = set(submitted_context.get("dominant_style_tags", []))
    fresh_style_tags = sorted(style_tags - dominant_style_tags)
    matched_style_tags = sorted(style_tags & dominant_style_tags)
    alpha_component = 1.0 - min(1.0, alpha_type_count / 3.0)
    if style_tags:
        style_component = len(fresh_style_tags) / len(style_tags)
    else:
        style_component = 0.5
    contrast_score = _clip(0.58 * alpha_component + 0.42 * style_component, 0.0, 1.0)

    reasons = []
    if alpha_type_count == 0:
        reasons.append(f"fresh_alpha_type={alpha_type}")
    elif alpha_type not in set(submitted_context.get("saturated_alpha_types", [])):
        reasons.append(f"less_saturated_alpha_type={alpha_type}x{alpha_type_count}")
    if fresh_style_tags:
        reasons.append(f"fresh_style_tags={','.join(fresh_style_tags[:4])}")

    return {
        "contrast_score": _round(contrast_score),
        "fresh_style_tags": fresh_style_tags,
        "matched_style_tags": matched_style_tags,
        "reasons": reasons,
    }


def summarize_sources(idea: dict) -> str:
    title = idea.get("title") or idea.get("query") or "unknown source"
    source = idea.get("source", "unknown")
    year = idea.get("year")
    if year:
        return f"{title} ({source}, {year})"
    return f"{title} ({source})"


def build_candidate_from_program(
    *,
    thesis_id: str,
    thesis_label: str,
    family_components: list[str],
    why_label: str,
    token_program: list[str] | None = None,
    idea: dict,
    idea_profile: dict,
    memory: dict,
    scout_memory: dict,
    history_index: HistoryIndex,
    submitted_context: dict | None = None,
    brain_feedback_context: dict | None = None,
    expression_override: str | None = None,
) -> dict | None:
    if expression_override:
        expression = expression_override.strip()
        if not expression:
            return None
    else:
        try:
            expression = render_token_program(token_program or [])
        except Exception:
            return None

    settings_profiles = recommend_settings_profiles(family_components, idea_profile["horizon"], idea_profile.get("style_tags", []))
    result, evaluations = evaluate_expression_across_settings(expression, settings_profiles=settings_profiles, history_index=history_index)
    if result is None:
        return None

    style_tags = sorted(set(idea_profile.get("style_tags", [])))
    data_categories = infer_data_categories(style_tags)
    data_category_count = len(data_categories)
    book_alignment_component = 1.0 if "book_alpha_design" in style_tags or "simple" in style_tags else 0.0
    category_focus_component = 1.0 if data_category_count <= 2 else 0.82 if data_category_count == 3 else 0.55
    style_alignment_raw = score_style_alignment(style_tags, memory)
    style_alignment_component = _clip(style_alignment_raw * 4.0, 0.0, 1.0)
    source_history_bonus = _source_history_adjustment(idea_profile["source_key"], family_components[0], scout_memory)
    source_quality_score = _clip(idea_profile["source_quality_score"] + source_history_bonus, 0.0, 1.0)
    family_diversity = compute_family_diversity_profile(
        family_components,
        horizon=idea_profile["horizon"],
        style_tags=style_tags,
        scout_memory=scout_memory,
        brain_feedback_context=brain_feedback_context,
    )
    family_diversity_score = float(family_diversity["score"])
    blocked_scout_skeleton = expression_skeleton(expression) in set(scout_memory.get("blocked_skeletons", []))
    novelty_score = _clip(
        0.52 * float(result.get("uniqueness_proxy", 0.0))
        + 0.18 * source_quality_score
        + 0.15 * (1.0 if expression_skeleton(expression) not in set(scout_memory.get("blocked_skeletons", [])) else 0.0)
        + 0.15 * max(0.0, 1.0 - history_index.skeleton_match_count(expression) * 0.12),
        0.0,
        1.0,
    )
    settings_robustness = summarize_settings_robustness(evaluations)
    robustness_score = float(settings_robustness["robustness_score"])
    submitted_overlap = compute_submitted_overlap(expression, result, submitted_context)
    submitted_contrast = compute_submitted_contrast(result, submitted_context)
    submitted_overlap_score = float(submitted_overlap["overlap_score"])
    submitted_diversity_score = float(submitted_overlap["diversity_score"])
    submitted_contrast_score = float(submitted_contrast["contrast_score"])
    brain_feedback = compute_brain_feedback_penalty(expression, result, family_components, brain_feedback_context)
    surrogate_shadow = summarize_surrogate_shadow(result)

    alpha_component = float(result.get("alpha_score", 0.0)) / 100.0
    verdict_component = {"PASS": 1.0, "LIKELY_PASS": 0.82, "BORDERLINE": 0.55, "FAIL": 0.20}.get(result.get("verdict"), 0.20)
    confidence_component = {"HIGH": 1.0, "MEDIUM": 0.72, "LOW": 0.42}.get(result.get("confidence"), 0.42)

    candidate_score = _clip(
        0.32 * alpha_component
        + 0.16 * source_quality_score
        + 0.10 * idea_profile["specificity_score"]
        + 0.10 * idea_profile["equity_factor_relevance"]
        + 0.04 * style_alignment_component
        + 0.07 * category_focus_component
        + 0.05 * book_alignment_component
        + 0.08 * novelty_score
        + 0.08 * robustness_score,
        0.0,
        1.0,
    )
    candidate_score = _clip(candidate_score + (0.10 * (submitted_contrast_score - 0.5)), 0.0, 1.0)
    candidate_score = _clip(candidate_score + (0.10 * (family_diversity_score - 0.5)), 0.0, 1.0)
    candidate_score = _clip(candidate_score - (0.14 * submitted_overlap_score), 0.0, 1.0)
    candidate_score = _clip(candidate_score - float(brain_feedback["penalty_score"]), 0.0, 1.0)
    candidate_score = _clip(candidate_score - float(surrogate_shadow["penalty_score"]), 0.0, 1.0)
    if blocked_scout_skeleton:
        candidate_score = _clip(candidate_score - 0.12, 0.0, 1.0)
    confidence_score = _clip(
        0.30 * alpha_component
        + 0.16 * source_quality_score
        + 0.14 * verdict_component
        + 0.10 * confidence_component
        + 0.05 * style_alignment_component
        + 0.07 * category_focus_component
        + 0.05 * book_alignment_component
        + 0.05 * novelty_score
        + 0.08 * robustness_score,
        0.0,
        1.0,
    )
    confidence_score = _clip(confidence_score + (0.06 * (submitted_contrast_score - 0.5)), 0.0, 1.0)
    confidence_score = _clip(confidence_score + (0.05 * (family_diversity_score - 0.5)), 0.0, 1.0)
    confidence_score = _clip(confidence_score - (0.10 * submitted_overlap_score), 0.0, 1.0)
    confidence_score = _clip(confidence_score - float(brain_feedback["confidence_drag"]), 0.0, 1.0)
    confidence_score = _clip(confidence_score - float(surrogate_shadow["confidence_drag"]), 0.0, 1.0)
    if blocked_scout_skeleton:
        confidence_score = _clip(confidence_score - 0.08, 0.0, 1.0)

    risk_tags = build_risk_tags(
        result,
        source_quality_score,
        data_category_count=data_category_count,
        settings_robustness=settings_robustness,
        submitted_overlap=submitted_overlap,
        brain_feedback=brain_feedback,
        surrogate_shadow=surrogate_shadow,
        scout_blocked_skeleton=blocked_scout_skeleton,
    )
    source_reference = idea.get("title") or idea.get("query")
    inspiration_prefix = (
        "Learned from"
        if expression_override or idea.get("source") in {"zip_knowledge", "github_readme"}
        else "Inspired by public research around"
    )
    candidate = {
        "variant_id": source_reference,
        "thesis_id": thesis_id,
        "thesis": thesis_label,
        "thesis_family_ids": family_components,
        "source_kind": idea.get("source", "unknown"),
        "learned_seed": bool(expression_override or idea.get("source") in {"zip_knowledge", "github_readme"}),
        "why": (
            f"{inspiration_prefix} '{source_reference}'. "
            f"Mapped to {why_label} because {idea_profile['reasoning']}."
        ),
        "expression": expression,
        "token_program": list(token_program or []),
        "novelty_score": _round(novelty_score),
        "robustness_score": _round(robustness_score),
        "family_diversity_score": _round(family_diversity_score),
        "family_diversity_reasons": family_diversity["reasons"],
        "family_crowded_families": family_diversity["crowded_families"],
        "family_fresh_families": family_diversity["fresh_families"],
        "submitted_overlap_score": _round(submitted_overlap_score),
        "submitted_diversity_score": _round(submitted_diversity_score),
        "submitted_overlap_reasons": submitted_overlap["reasons"],
        "submitted_style_overlap_tags": submitted_overlap["matched_style_tags"],
        "submitted_contrast_score": _round(submitted_contrast_score),
        "submitted_contrast_reasons": submitted_contrast["reasons"],
        "submitted_fresh_style_tags": submitted_contrast["fresh_style_tags"],
        "brain_feedback_penalty": brain_feedback["penalty_score"],
        "brain_feedback_confidence_drag": brain_feedback["confidence_drag"],
        "brain_feedback_reasons": brain_feedback["reasons"],
        "brain_feedback_matched_families": brain_feedback["matched_families"],
        "brain_feedback_matched_style_tags": brain_feedback["matched_style_tags"],
        "brain_feedback_exact_skeleton_match": brain_feedback["exact_skeleton_match"],
        "brain_feedback_context_mode": brain_feedback.get("context_mode", "no_signal"),
        "brain_feedback_candidate_context": brain_feedback.get("candidate_context", "global"),
        "surrogate_shadow_penalty": surrogate_shadow["penalty_score"],
        "surrogate_shadow_confidence_drag": surrogate_shadow["confidence_drag"],
        "surrogate_shadow_reasons": surrogate_shadow["reasons"],
        "surrogate_shadow_preview_verdict": surrogate_shadow["preview_verdict"],
        "surrogate_shadow_alignment": surrogate_shadow["alignment"],
        "surrogate_shadow_hard_signal": surrogate_shadow["hard_signal"],
        "settings_robustness": settings_robustness,
        "style_alignment_score": _round(style_alignment_component),
        "style_alignment_raw": _round(style_alignment_raw),
        "scout_blocked_skeleton": blocked_scout_skeleton,
        "risk_tags": risk_tags,
        "candidate_score": _round(candidate_score),
        "confidence_score": _round(confidence_score),
        "source_key": idea_profile["source_key"],
        "source_title": idea.get("title") or idea.get("query"),
        "source_query": idea.get("query"),
        "source_ideas": [summarize_sources(idea)],
        "source_quality_score": _round(source_quality_score),
        "source_specificity_score": idea_profile["specificity_score"],
        "source_equity_relevance": idea_profile["equity_factor_relevance"],
        "data_categories": data_categories,
        "book_alignment_score": _round(book_alignment_component),
        "horizon": idea_profile["horizon"],
        "settings": result["settings"]["label"],
        "settings_evaluated": evaluations,
        "local_metrics": result,
        "thinking": (
            f"Chon {thesis_id} vi source co quality={idea_profile['source_quality_score']} va "
            f"theme_strength={idea_profile['theme_strength']}. Internal verdict={result['verdict']}, "
            f"alpha_score={result['alpha_score']}, robustness={_round(robustness_score)}, "
            f"family_diversity={_round(family_diversity_score)}, "
            f"submitted_overlap={_round(submitted_overlap_score)}, brain_feedback={brain_feedback['penalty_score']}, "
            f"surrogate_shadow={surrogate_shadow['penalty_score']}, "
            f"novelty={_round(novelty_score)}, "
            f"book_alignment={_round(book_alignment_component)}."
        ),
    }
    candidate["lineage"] = ensure_candidate_lineage(
        candidate,
        stage_source="scout",
        source_detail=idea.get("source"),
        default_hypothesis_id=thesis_id,
        default_hypothesis_label=thesis_label,
        default_family=family_components[0] if family_components else thesis_id,
        default_family_components=family_components,
        default_generation_reason=why_label,
    )
    return candidate


def apply_source_crowding_adjustments(candidates: list[dict]) -> list[dict]:
    by_source: dict[str, list[dict]] = defaultdict(list)
    by_source_family: dict[tuple[str, str], list[dict]] = defaultdict(list)

    for candidate in candidates:
        source_key = str(candidate.get("source_key") or "__unknown__")
        thesis_id = str(candidate.get("thesis_id") or "__unknown__")
        by_source[source_key].append(candidate)
        by_source_family[(source_key, thesis_id)].append(candidate)

    for group in by_source.values():
        group.sort(key=_candidate_rank_key, reverse=True)
    for group in by_source_family.values():
        group.sort(key=_candidate_rank_key, reverse=True)

    for source_key, group in by_source.items():
        total_source_count = len(group)
        for source_rank, candidate in enumerate(group, start=1):
            candidate["source_variant_rank"] = source_rank
            candidate["source_candidate_count"] = total_source_count

    for source_family_key, group in by_source_family.items():
        total_source_family_count = len(group)
        for source_family_rank, candidate in enumerate(group, start=1):
            source_crowding_penalty = float(candidate.get("source_crowding_penalty", 0.0) or 0.0)
            if total_source_family_count > 2:
                source_crowding_penalty += min(0.12, max(0, source_family_rank - 2) * 0.05)
            if candidate.get("source_candidate_count", 0) > 4:
                source_crowding_penalty += min(0.06, max(0, int(candidate.get("source_variant_rank", 1)) - 4) * 0.02)

            candidate["source_family_variant_rank"] = source_family_rank
            candidate["source_family_candidate_count"] = total_source_family_count
            candidate["source_crowding_penalty"] = _round(source_crowding_penalty)
            if source_crowding_penalty > 0.0:
                candidate["candidate_score"] = _round(
                    _clip(float(candidate.get("candidate_score", 0.0)) - source_crowding_penalty, 0.0, 1.0)
                )
                candidate["confidence_score"] = _round(
                    _clip(float(candidate.get("confidence_score", 0.0)) - min(0.08, source_crowding_penalty * 0.7), 0.0, 1.0)
                )
                risk_tags = set(candidate.get("risk_tags", []))
                risk_tags.add("source_crowding_risk")
                candidate["risk_tags"] = sorted(risk_tags)

    return candidates


def build_candidates(
    ideas: list[dict],
    *,
    memory: dict,
    scout_memory: dict,
    history_index: HistoryIndex,
    submitted_context: dict | None = None,
    brain_feedback_context: dict | None = None,
) -> list[dict]:
    candidates = []
    seen_expressions = set()

    for idea in ideas:
        idea_profile = dict(idea.get("idea_profile") or infer_idea_profile(idea, scout_memory=scout_memory))
        if not idea_profile.get("generation_ok", True):
            continue
        generation_families = _select_generation_families(
            idea_profile,
            scout_memory=scout_memory,
            brain_feedback_context=brain_feedback_context,
        )
        idea_profile["families"] = generation_families
        is_learned_source = idea.get("source") in {"zip_knowledge", "github_readme"} or bool(idea.get("seed_expressions"))
        if (
            not is_learned_source
            and idea_profile["source_key"] in set(scout_memory.get("blocked_source_keys", []))
            and idea_profile["source_quality_score"] < 0.72
        ):
            continue

        family_horizons = {}
        seed_expressions = [expr for expr in idea.get("seed_expressions", []) if isinstance(expr, str) and expr.strip()]
        if seed_expressions:
            seed_family = idea_profile["families"][0] if idea_profile.get("families") else (idea.get("bias_families", ["technical_indicator"])[0])
            family_horizons[seed_family] = adjust_horizon_from_memory(
                idea_profile["horizon"],
                seed_family,
                memory,
                scout_memory,
            )
            for expression in seed_expressions:
                candidate = build_candidate_from_program(
                    thesis_id=f"seed__{seed_family}",
                    thesis_label=f"Learned Seed: {THESIS_INDEX.get(seed_family, {}).get('label', seed_family)}",
                    family_components=[seed_family],
                    why_label="Learned seed expression adaptation",
                    token_program=None,
                    expression_override=expression,
                    idea=idea,
                    idea_profile={**idea_profile, "horizon": family_horizons[seed_family]},
                    memory=memory,
                    scout_memory=scout_memory,
                    history_index=history_index,
                    submitted_context=submitted_context,
                    brain_feedback_context=brain_feedback_context,
                )
                if candidate is None or candidate["expression"] in seen_expressions:
                    continue
                seen_expressions.add(candidate["expression"])
                candidates.append(candidate)
                history_index.observe_expression(candidate["expression"], candidate["local_metrics"])

        for family_id in idea_profile["families"]:
            family_horizons[family_id] = adjust_horizon_from_memory(
                idea_profile["horizon"],
                family_id,
                memory,
                scout_memory,
            )
            thesis = THESIS_INDEX.get(family_id)
            if not thesis:
                continue
            for variant in thesis.get("variants", []):
                variant_style_tags = sorted(set(variant.get("style_tags", [])) | set(idea_profile.get("style_tags", [])))
                for token_program in mutate_token_program(
                    list(variant["token_program"]),
                    family_id=family_id,
                    horizon=family_horizons[family_id],
                    style_tags=variant_style_tags,
                ):
                    candidate = build_candidate_from_program(
                        thesis_id=family_id,
                        thesis_label=thesis["label"],
                        family_components=[family_id],
                        why_label=thesis["label"],
                        token_program=token_program,
                        idea=idea,
                        idea_profile={**idea_profile, "horizon": family_horizons[family_id], "style_tags": variant_style_tags},
                        memory=memory,
                        scout_memory=scout_memory,
                        history_index=history_index,
                        submitted_context=submitted_context,
                        brain_feedback_context=brain_feedback_context,
                    )
                    if candidate is None or candidate["expression"] in seen_expressions:
                        continue
                    seen_expressions.add(candidate["expression"])
                    candidates.append(candidate)
                    history_index.observe_expression(candidate["expression"], candidate["local_metrics"])

        if len(idea_profile["families"]) >= 2 and idea_profile["theme_strength"] >= 2.2:
            for token_program, metadata in build_blend_programs(idea, idea_profile["families"], family_horizons):
                candidate = build_candidate_from_program(
                    thesis_id=f"blend__{'__'.join(metadata['families'])}",
                    thesis_label=metadata["label"],
                    family_components=metadata["families"],
                    why_label=metadata["label"],
                    token_program=token_program,
                    idea=idea,
                    idea_profile={**idea_profile, "style_tags": sorted(set(idea_profile.get("style_tags", [])) | {"normalization"})},
                    memory=memory,
                    scout_memory=scout_memory,
                    history_index=history_index,
                    submitted_context=submitted_context,
                    brain_feedback_context=brain_feedback_context,
                )
                if candidate is None or candidate["expression"] in seen_expressions:
                    continue
                seen_expressions.add(candidate["expression"])
                candidates.append(candidate)
                history_index.observe_expression(candidate["expression"], candidate["local_metrics"])

    candidates.sort(
        key=lambda item: (
            float(item.get("candidate_score", 0.0)),
            float(item.get("submitted_diversity_score", 1.0)),
            float(item.get("submitted_contrast_score", 0.5)),
            float(item.get("robustness_score", 0.0)),
            float(item.get("confidence_score", 0.0)),
            float(item.get("local_metrics", {}).get("alpha_score", 0.0)),
            float(item.get("novelty_score", 0.0)),
        ),
        reverse=True,
    )
    candidates = apply_source_crowding_adjustments(candidates)
    candidates.sort(
        key=lambda item: (
            float(item.get("candidate_score", 0.0)),
            float(item.get("submitted_diversity_score", 1.0)),
            float(item.get("submitted_contrast_score", 0.5)),
            float(item.get("robustness_score", 0.0)),
            float(item.get("confidence_score", 0.0)),
            float(item.get("local_metrics", {}).get("alpha_score", 0.0)),
            -float(item.get("source_crowding_penalty", 0.0)),
            float(item.get("novelty_score", 0.0)),
        ),
        reverse=True,
    )
    return candidates


def assess_candidate_quality(candidate: dict, *, min_alpha_score: float, min_confidence_score: float) -> tuple[bool, list[str]]:
    local = candidate.get("local_metrics", {})
    reasons = []

    if local.get("verdict") not in {"PASS", "LIKELY_PASS"}:
        reasons.append(f"verdict={local.get('verdict')}")
    if float(local.get("alpha_score", 0.0)) < min_alpha_score:
        reasons.append(f"alpha_score<{min_alpha_score}")
    if float(candidate.get("confidence_score", 0.0)) < min_confidence_score:
        reasons.append(f"confidence_score<{min_confidence_score}")
    if float(local.get("uniqueness_proxy", 0.0)) < 0.45:
        reasons.append("uniqueness_proxy<0.45")
    if "similarity_risk" in set(candidate.get("risk_tags", [])):
        reasons.append("similarity_risk")
    if "category_overload_risk" in set(candidate.get("risk_tags", [])):
        reasons.append("category_overload_risk")
    if "submitted_skeleton_risk" in set(candidate.get("risk_tags", [])):
        reasons.append("submitted_skeleton_risk")
    if "brain_exact_skeleton_risk" in set(candidate.get("risk_tags", [])):
        reasons.append("brain_exact_skeleton_risk")
    if "scout_blocked_skeleton_risk" in set(candidate.get("risk_tags", [])):
        reasons.append("scout_blocked_skeleton_risk")
    if "out_of_sample_risk" in set(candidate.get("risk_tags", [])):
        reasons.append("out_of_sample_risk")
    if candidate.get("surrogate_shadow_hard_signal") == "severe_mismatch":
        reasons.append("surrogate_shadow_risk")
    if float(candidate.get("brain_feedback_penalty", 0.0)) >= 0.22:
        reasons.append("brain_feedback_penalty")
    if float(candidate.get("submitted_overlap_score", 0.0)) >= 0.72:
        reasons.append("submitted_overlap_risk")

    for check_name in ("LOW_SHARPE", "LOW_FITNESS", "LOW_SUB_UNIVERSE_SHARPE"):
        if local.get(check_name) == "FAIL":
            reasons.append(check_name)

    return (len(reasons) == 0), reasons


def _candidate_rank_key(item: dict) -> tuple:
    return (
        _verdict_rank(item.get("local_metrics", {}).get("verdict")),
        float(item.get("family_diversity_score", 0.5)),
        float(item.get("submitted_diversity_score", 1.0)),
        float(item.get("robustness_score", 0.0)),
        float(item.get("local_metrics", {}).get("alpha_score", 0.0)),
        float(item.get("confidence_score", 0.0)),
        float(item.get("candidate_score", 0.0)),
        float(item.get("source_quality_score", 0.0)),
    )


def _candidate_selection_score(candidate: dict) -> float:
    local = candidate.get("local_metrics", {})
    verdict_component = {
        "PASS": 1.0,
        "LIKELY_PASS": 0.82,
        "BORDERLINE": 0.48,
        "FAIL": 0.16,
    }.get(local.get("verdict"), 0.16)
    return _clip(
        0.34 * float(candidate.get("candidate_score", 0.0))
        + 0.16 * float(candidate.get("robustness_score", 0.0))
        + 0.12 * float(candidate.get("confidence_score", 0.0))
        + 0.12 * (float(local.get("alpha_score", 0.0)) / 100.0)
        + 0.10 * float(candidate.get("family_diversity_score", 0.5))
        + 0.10 * float(candidate.get("submitted_diversity_score", 1.0))
        + 0.08 * float(candidate.get("source_quality_score", 0.0))
        + 0.08 * verdict_component,
        0.0,
        1.0,
    )


def _jaccard_distance(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return 1.0 - (len(left & right) / len(union))


def _candidate_distance(left: dict, right: dict) -> float:
    left_skeleton = expression_skeleton(left.get("expression", ""))
    right_skeleton = expression_skeleton(right.get("expression", ""))
    skeleton_distance = 1.0 - SequenceMatcher(None, left_skeleton, right_skeleton).ratio()
    family_distance = _jaccard_distance(
        set(left.get("thesis_family_ids", []) or [left.get("thesis_id")]),
        set(right.get("thesis_family_ids", []) or [right.get("thesis_id")]),
    )
    style_distance = _jaccard_distance(
        set(left.get("local_metrics", {}).get("style_tags", [])),
        set(right.get("local_metrics", {}).get("style_tags", [])),
    )
    source_distance = 0.0 if left.get("source_key") == right.get("source_key") else 1.0
    return _clip(
        0.46 * skeleton_distance
        + 0.24 * family_distance
        + 0.20 * style_distance
        + 0.10 * source_distance,
        0.0,
        1.0,
    )


def _min_candidate_distance(candidate: dict, selected: list[dict]) -> float:
    if not selected:
        return 1.0
    return min(_candidate_distance(candidate, existing) for existing in selected)


def _family_selection_adjustment(candidate: dict, thesis_counts: Counter) -> float:
    family_ids = candidate.get("thesis_family_ids", []) or [candidate.get("thesis_id")]
    family_counts = [thesis_counts[family_id] for family_id in family_ids if family_id]
    max_family_count = max(family_counts, default=0)
    family_diversity_score = float(candidate.get("family_diversity_score", 0.5))
    adjustment = 0.06 * (family_diversity_score - 0.5)
    if max_family_count >= 1:
        adjustment -= min(0.16, max_family_count * (0.08 + max(0.0, 0.55 - family_diversity_score) * 0.12))
    elif set(family_ids) & CONTRAST_FAMILIES:
        adjustment += 0.03
    return adjustment


def _pick_diverse_candidate(
    pool: list[dict],
    *,
    selected: list[dict],
    diversity_weight: float,
    thesis_counts: Counter,
) -> dict | None:
    if not pool:
        return None
    best = None
    best_score = None
    for candidate in pool:
        score = (
            _candidate_selection_score(candidate)
            + (diversity_weight * _min_candidate_distance(candidate, selected))
            + _family_selection_adjustment(candidate, thesis_counts)
        )
        if best is None or score > best_score:
            best = candidate
            best_score = score
    return best


def _selection_reasons_for_state(
    candidate: dict,
    *,
    selected_skeletons: set[str],
    thesis_counts: Counter,
    source_counts: Counter,
    source_family_counts: Counter,
    existing_seed_store: dict,
) -> list[str]:
    reasons = list(candidate.get("quality_fail_reasons", []))
    skeleton = expression_skeleton(candidate.get("expression", ""))
    family_ids = candidate.get("thesis_family_ids", []) or [candidate.get("thesis_id")]
    source_key = candidate.get("source_key")
    source_family_key = (source_key, candidate.get("thesis_id"))

    if candidate.get("expression") in existing_seed_store:
        reasons.append("already_in_seed_store")
    if skeleton in selected_skeletons:
        reasons.append("duplicate_skeleton_in_top_picks")
    if any(thesis_counts[family_id] >= 2 for family_id in family_ids if family_id):
        reasons.append("thesis_family_cap")
    if source_family_counts[source_family_key] >= 1:
        reasons.append("source_family_cap")
    if source_counts[source_key] >= 2:
        reasons.append("source_paper_cap")
    return list(dict.fromkeys(reasons))


def _is_relaxed_fill_candidate(
    candidate: dict,
    reasons: list[str],
    *,
    min_alpha_score: float,
    min_confidence_score: float,
) -> bool:
    local = candidate.get("local_metrics", {})
    risk_tags = set(candidate.get("risk_tags", []))
    if local.get("verdict") not in {"PASS", "LIKELY_PASS"}:
        return False
    if risk_tags & {
        "similarity_risk",
        "submitted_skeleton_risk",
        "brain_exact_skeleton_risk",
        "scout_blocked_skeleton_risk",
        "out_of_sample_risk",
        "category_overload_risk",
        "surrogate_shadow_risk",
    }:
        return False

    for reason in reasons:
        if reason == "thesis_family_cap":
            continue
        if reason == "source_family_cap":
            return False
        if reason == "source_paper_cap":
            return False
        if reason == "brain_feedback_penalty":
            if float(candidate.get("brain_feedback_penalty", 0.0)) > RELAXED_BRAIN_FEEDBACK_CAP:
                return False
            continue
        if reason.startswith("confidence_score<"):
            if float(candidate.get("confidence_score", 0.0)) < max(0.0, min_confidence_score - RELAXED_CONFIDENCE_BUFFER):
                return False
            continue
        if reason.startswith("alpha_score<"):
            if float(local.get("alpha_score", 0.0)) < max(0.0, min_alpha_score - RELAXED_ALPHA_BUFFER):
                return False
            continue
        return False
    return bool(reasons)


def _is_watchlist_candidate(
    candidate: dict,
    reasons: list[str],
    *,
    min_alpha_score: float,
    min_confidence_score: float,
) -> bool:
    local = candidate.get("local_metrics", {})
    risk_tags = set(candidate.get("risk_tags", []))
    if local.get("verdict") not in {"PASS", "LIKELY_PASS", "BORDERLINE"}:
        return False
    if risk_tags & {
        "similarity_risk",
        "submitted_skeleton_risk",
        "brain_exact_skeleton_risk",
        "scout_blocked_skeleton_risk",
        "out_of_sample_risk",
        "category_overload_risk",
        "surrogate_shadow_risk",
    }:
        return False

    for reason in reasons:
        if reason == "source_family_cap":
            return False
        if reason == "source_paper_cap":
            continue
        if reason == "brain_feedback_penalty":
            if float(candidate.get("brain_feedback_penalty", 0.0)) > WATCHLIST_BRAIN_FEEDBACK_CAP:
                return False
            continue
        if reason.startswith("confidence_score<"):
            if float(candidate.get("confidence_score", 0.0)) < max(0.0, min_confidence_score - WATCHLIST_CONFIDENCE_BUFFER):
                return False
            continue
        if reason.startswith("alpha_score<"):
            if float(local.get("alpha_score", 0.0)) < max(0.0, min_alpha_score - WATCHLIST_ALPHA_BUFFER):
                return False
            continue
        if reason == "surrogate_shadow_risk":
            return False
        return False
    return True


def select_daily_picks(
    candidates: list[dict],
    *,
    count: int,
    min_alpha_score: float,
    min_confidence_score: float,
    include_watchlist: bool,
    existing_seed_store: dict | None = None,
    diversity_weight: float = DEFAULT_DIVERSITY_WEIGHT,
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    existing_seed_store = existing_seed_store or {}
    selected = []
    annotated = []

    selected_skeletons = set()
    thesis_counts = Counter()
    source_counts = Counter()
    source_family_counts = Counter()

    ranked = sorted(candidates, key=_candidate_rank_key, reverse=True)

    for candidate in ranked:
        candidate = dict(candidate)
        quality_ok, quality_reasons = assess_candidate_quality(
            candidate,
            min_alpha_score=min_alpha_score,
            min_confidence_score=min_confidence_score,
        )
        candidate["quality_ok"] = quality_ok
        candidate["quality_fail_reasons"] = quality_reasons
        candidate["selection_rank_score"] = _round(_candidate_selection_score(candidate))
        annotated.append(candidate)

    remaining = list(annotated)
    while len(selected) < count:
        eligible = []
        for candidate in remaining:
            reasons = _selection_reasons_for_state(
                candidate,
                selected_skeletons=selected_skeletons,
                thesis_counts=thesis_counts,
                source_counts=source_counts,
                source_family_counts=source_family_counts,
                existing_seed_store=existing_seed_store,
            )
            if not reasons:
                eligible.append(candidate)
        chosen = _pick_diverse_candidate(
            eligible,
            selected=selected,
            diversity_weight=diversity_weight,
            thesis_counts=thesis_counts,
        )
        if chosen is None:
            break

        family_ids = chosen.get("thesis_family_ids", []) or [chosen.get("thesis_id")]
        chosen["selection_status"] = "selected"
        chosen["selection_mode"] = "strict"
        chosen["selection_reasons"] = []
        chosen["selection_reason"] = "passed_quality_gate"
        chosen["qualified"] = True
        chosen["quality_label"] = "qualified"
        chosen["lineage"] = ensure_candidate_lineage(
            chosen,
            stage_source="scout",
            source_detail=chosen.get("source_kind"),
            default_hypothesis_id=chosen.get("thesis_id"),
            default_hypothesis_label=chosen.get("thesis"),
            default_family_components=chosen.get("thesis_family_ids", []),
            default_generation_reason=chosen.get("selection_reason"),
        )
        selected.append(chosen)
        selected_skeletons.add(expression_skeleton(chosen.get("expression", "")))
        for family_id in family_ids:
            if family_id:
                thesis_counts[family_id] += 1
        source_counts[chosen.get("source_key")] += 1
        source_family_counts[(chosen.get("source_key"), chosen.get("thesis_id"))] += 1
        remaining.remove(chosen)

    relaxed_limit = min(max(0, count - len(selected)), max(1, int(round(count * RELAXED_FILL_RATIO)))) if count > 0 else 0
    relaxed_promotions = 0
    while relaxed_promotions < relaxed_limit:
        eligible = []
        for candidate in remaining:
            reasons = _selection_reasons_for_state(
                candidate,
                selected_skeletons=selected_skeletons,
                thesis_counts=thesis_counts,
                source_counts=source_counts,
                source_family_counts=source_family_counts,
                existing_seed_store=existing_seed_store,
            )
            if _is_relaxed_fill_candidate(
                candidate,
                reasons,
                min_alpha_score=min_alpha_score,
                min_confidence_score=min_confidence_score,
            ):
                candidate["selection_reasons"] = reasons
                eligible.append(candidate)
        chosen = _pick_diverse_candidate(
            eligible,
            selected=selected,
            diversity_weight=diversity_weight,
            thesis_counts=thesis_counts,
        )
        if chosen is None:
            break

        family_ids = chosen.get("thesis_family_ids", []) or [chosen.get("thesis_id")]
        original_reason = ", ".join(chosen.get("selection_reasons", [])) or "near_threshold"
        chosen["selection_status"] = "selected"
        chosen["selection_mode"] = "relaxed"
        chosen["selection_reason"] = f"relaxed_fill_after_strict_shortfall: {original_reason}"
        chosen["qualified"] = True
        chosen["quality_label"] = "relaxed_fill"
        chosen["lineage"] = ensure_candidate_lineage(
            chosen,
            stage_source="scout",
            source_detail=chosen.get("source_kind"),
            default_hypothesis_id=chosen.get("thesis_id"),
            default_hypothesis_label=chosen.get("thesis"),
            default_family_components=chosen.get("thesis_family_ids", []),
            default_generation_reason=chosen.get("selection_reason"),
        )
        selected.append(chosen)
        selected_skeletons.add(expression_skeleton(chosen.get("expression", "")))
        for family_id in family_ids:
            if family_id:
                thesis_counts[family_id] += 1
        source_counts[chosen.get("source_key")] += 1
        source_family_counts[(chosen.get("source_key"), chosen.get("thesis_id"))] += 1
        remaining.remove(chosen)
        relaxed_promotions += 1

    watchlist = []
    rejected = []
    for candidate in annotated:
        if candidate.get("selection_status") == "selected":
            continue
        local = candidate.get("local_metrics", {})
        state_reasons = _selection_reasons_for_state(
            candidate,
            selected_skeletons=selected_skeletons,
            thesis_counts=thesis_counts,
            source_counts=source_counts,
            source_family_counts=source_family_counts,
            existing_seed_store=existing_seed_store,
        )
        quality_reasons = list(candidate.get("quality_fail_reasons", []))
        reasons = list(dict.fromkeys(quality_reasons + state_reasons))
        candidate["selection_reasons"] = reasons
        if include_watchlist and _is_watchlist_candidate(
            candidate,
            reasons,
            min_alpha_score=min_alpha_score,
            min_confidence_score=min_confidence_score,
        ):
            candidate["selection_status"] = "watchlist"
            candidate["selection_mode"] = "watchlist"
            candidate["selection_reason"] = ", ".join(reasons) if reasons else "below_final_cut"
            candidate["qualified"] = False
            candidate["quality_label"] = "watchlist"
            candidate["lineage"] = ensure_candidate_lineage(
                candidate,
                stage_source="scout",
                source_detail=candidate.get("source_kind"),
                default_hypothesis_id=candidate.get("thesis_id"),
                default_hypothesis_label=candidate.get("thesis"),
                default_family_components=candidate.get("thesis_family_ids", []),
                default_generation_reason=candidate.get("selection_reason"),
            )
            watchlist.append(candidate)
        else:
            candidate["selection_status"] = "rejected"
            candidate["selection_mode"] = "rejected"
            candidate["selection_reason"] = ", ".join(reasons) if reasons else "below_quality_gate"
            candidate["qualified"] = False
            candidate["quality_label"] = "rejected"
            candidate["lineage"] = ensure_candidate_lineage(
                candidate,
                stage_source="scout",
                source_detail=candidate.get("source_kind"),
                default_hypothesis_id=candidate.get("thesis_id"),
                default_hypothesis_label=candidate.get("thesis"),
                default_family_components=candidate.get("thesis_family_ids", []),
                default_generation_reason=candidate.get("selection_reason"),
            )
            rejected.append(
                {
                    "expression": candidate.get("expression"),
                    "reason": candidate["selection_reason"],
                    "source_title": candidate.get("source_title"),
                }
            )

    return selected, watchlist, rejected, annotated


def build_payload(
    *,
    ideas: list[dict],
    candidates: list[dict],
    memory: dict,
    count: int,
    seed_store: dict,
    submitted_context: dict | None = None,
    brain_feedback_context: dict | None = None,
    search_breadth: str = "standard",
    query_profile_count: int = 0,
    min_alpha_score: float = 68.0,
    min_confidence_score: float = 0.58,
    report_min_alpha_score: float = DEFAULT_REPORT_MIN_ALPHA_SCORE,
    report_min_confidence_score: float = DEFAULT_REPORT_MIN_CONFIDENCE_SCORE,
    report_min_robustness_score: float = DEFAULT_REPORT_MIN_ROBUSTNESS_SCORE,
    include_watchlist: bool = False,
    fallback_mode: bool = False,
    diversity_weight: float = DEFAULT_DIVERSITY_WEIGHT,
    brain_feedback_status: dict | None = None,
    self_learning_status: dict | None = None,
    report_status: dict | None = None,
    run_timestamp: str | None = None,
    archive_frequency: str = DEFAULT_ARCHIVE_FREQUENCY,
    archived_paths: dict | None = None,
    archive_status: dict | None = None,
) -> tuple[dict, list[dict], list[dict], list[dict]]:
    submitted_context = submitted_context or {}
    brain_feedback_context = brain_feedback_context or {}
    brain_feedback_status = brain_feedback_status or {"status": "unknown", "path": "", "message": ""}
    self_learning_status = self_learning_status or {}
    report_status = report_status or {}
    archived_paths = archived_paths or {}
    archive_status = archive_status or {"written": False, "reason": "", "frequency": archive_frequency}
    selected, watchlist, rejected, annotated_candidates = select_daily_picks(
        candidates,
        count=count,
        min_alpha_score=min_alpha_score,
        min_confidence_score=min_confidence_score,
        include_watchlist=include_watchlist,
        existing_seed_store=seed_store,
        diversity_weight=diversity_weight,
    )
    reportable_selected = select_reportable_candidates(
        annotated_candidates,
        target_count=count,
        report_min_alpha_score=report_min_alpha_score,
        report_min_confidence_score=report_min_confidence_score,
        report_min_robustness_score=report_min_robustness_score,
    )
    strict_selected = [candidate for candidate in selected if candidate.get("selection_mode") == "strict"]
    relaxed_fill_count = sum(1 for item in selected if item.get("selection_mode") == "relaxed")
    payload = {
        "memory": memory,
        "submitted_context": summarize_submitted_context(submitted_context),
        "brain_feedback": summarize_brain_feedback_context(brain_feedback_context),
        "brain_feedback_status": brain_feedback_status,
        "self_learning": self_learning_status,
        "report_status": report_status,
        "run_timestamp": run_timestamp,
        "archived_paths": archived_paths,
        "archive_status": archive_status,
        "ideas": ideas,
        "batch": {
            "candidates": annotated_candidates,
            "qualified_count": len(selected),
            "strict_count": len(strict_selected),
            "reportable_count": len(reportable_selected),
            "watchlist_count": len(watchlist),
            "relaxed_fill_count": relaxed_fill_count,
            "source_inputs": sorted({idea.get("source", "unknown") for idea in ideas}),
            "notes": [
                "Public APIs used for scouting: OpenAlex, arXiv, GitHub, and local learned seeds when available.",
                "All alpha expressions were generated and scored locally; nothing was sent to WorldQuant.",
                "Daily picks use a quality-first selector with a diversity-aware greedy final pass.",
                "When strict selection returns too few picks, scout may promote a small number of near-threshold exploration ideas.",
                "When a submitted alpha library is present, scout penalizes candidates that overlap too much with that library.",
                "When real Brain simulation history is available, scout penalizes families and skeletons that already underperformed on TEST.",
                "When surrogate shadow disagrees strongly with the heuristic score, scout now downranks that candidate before final selection.",
                "Scout can now learn from local worldquant-miner ZIP seeds and GitHub READMEs before generating alpha candidates.",
            ],
            "fallback_mode": fallback_mode,
        },
        "settings": {
            "count": count,
            "search_breadth": search_breadth,
            "query_profile_count": query_profile_count,
            "min_alpha_score": min_alpha_score,
            "min_confidence_score": min_confidence_score,
            "report_min_alpha_score": report_min_alpha_score,
            "report_min_confidence_score": report_min_confidence_score,
            "report_min_robustness_score": report_min_robustness_score,
            "include_watchlist": include_watchlist,
            "diversity_weight": diversity_weight,
            "archive_frequency": archive_frequency,
        },
    }
    payload["selected"] = selected
    payload["reportable_selected"] = reportable_selected
    payload["watchlist"] = watchlist
    payload["rejected"] = rejected
    return payload, selected, watchlist, rejected


def build_history_records(
    candidates: list[dict],
    *,
    run_date: str,
) -> list[dict]:
    records = []
    for candidate in candidates:
        local = candidate.get("local_metrics", {})
        records.append(
            {
                "date": run_date,
                "source_key": candidate.get("source_key"),
                "source_title": candidate.get("source_title"),
                "source_query": candidate.get("source_query"),
                "source_ideas": candidate.get("source_ideas", []),
                "source_quality_score": candidate.get("source_quality_score"),
                "source_specificity_score": candidate.get("source_specificity_score"),
                "source_equity_relevance": candidate.get("source_equity_relevance"),
                "thesis_id": candidate.get("thesis_id"),
                "family_components": candidate.get("thesis_family_ids", []),
                "horizon": candidate.get("horizon"),
                "expression": candidate.get("expression"),
                "skeleton": expression_skeleton(candidate.get("expression", "")),
                "settings": candidate.get("settings"),
                "robustness_score": candidate.get("robustness_score"),
                "settings_pass_rate": candidate.get("settings_robustness", {}).get("pass_rate"),
                "submitted_overlap_score": candidate.get("submitted_overlap_score"),
                "surrogate_shadow_penalty": candidate.get("surrogate_shadow_penalty"),
                "surrogate_shadow_preview_verdict": candidate.get("surrogate_shadow_preview_verdict"),
                "surrogate_shadow_alignment": candidate.get("surrogate_shadow_alignment"),
                "alpha_score": local.get("alpha_score"),
                "confidence_score": candidate.get("confidence_score"),
                "verdict": local.get("verdict"),
                "selection_status": candidate.get("selection_status"),
                "rejection_reason": candidate.get("selection_reason"),
            }
        )
    return records


def split_top_level_args(text: str) -> list[str]:
    args = []
    current = []
    depth = 0
    for char in text:
        if char == "," and depth == 0:
            chunk = "".join(current).strip()
            if chunk:
                args.append(chunk)
            current = []
            continue
        if char == "(":
            depth += 1
        elif char == ")" and depth > 0:
            depth -= 1
        current.append(char)
    tail = "".join(current).strip()
    if tail:
        args.append(tail)
    return args


def parse_function_call(expression: str) -> tuple[str, str] | None:
    text = expression.strip()
    match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\(", text)
    if not match or not text.endswith(")"):
        return None
    name = match.group(1)
    open_index = len(name)
    depth = 0
    closing_index = None
    for index in range(open_index, len(text)):
        char = text[index]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                closing_index = index
                break
    if closing_index != len(text) - 1:
        return None
    return name, text[open_index + 1 : closing_index]


def pretty_format_expression(expression: str, *, indent: int = 0) -> str:
    parsed = parse_function_call(expression)
    if not parsed:
        return expression.strip()

    name, args_text = parsed
    args = split_top_level_args(args_text)
    if not args:
        return f"{name}()"

    indent_text = "  " * indent
    inner_indent = "  " * (indent + 1)
    rendered_args = ",\n".join(f"{inner_indent}{pretty_format_expression(arg, indent=indent + 1)}" for arg in args)
    return f"{name}(\n{rendered_args}\n{indent_text})"


def parse_settings_label(settings: str) -> dict:
    if not settings:
        return {}
    parts = [item.strip() for item in settings.split(",") if item.strip()]
    parsed = {}
    if len(parts) >= 2:
        parsed["Universe"] = f"{parts[0]}, {parts[1]}"
    for part in parts[2:]:
        pieces = part.split(" ", 1)
        if len(pieces) == 2:
            parsed[pieces[0]] = pieces[1].strip()
    return parsed


def display_metric(value, default: str = "n/a") -> str:
    if value is None or value == "":
        return default
    return str(value)


def append_candidate_markdown_block(lines: list[str], candidate: dict, *, heading: str) -> None:
    local = candidate.get("local_metrics", {})
    robustness = candidate.get("settings_robustness", {})
    settings = parse_settings_label(candidate.get("settings") or "")
    lines.append(heading)
    lines.append("")
    lines.append("**Alpha**")
    lines.append("```text")
    lines.append(pretty_format_expression(candidate.get("expression") or ""))
    lines.append("```")
    lines.append("")
    lines.append("**Settings**")
    if settings:
        for label in ("Universe", "Decay", "Delay", "Truncation", "Neutralization"):
            if settings.get(label):
                lines.append(f"- {label}: {settings[label]}")
    else:
        lines.append(f"- {candidate.get('settings')}")
    lines.append("")
    lines.append("**Scores**")
    lines.append(f"- Verdict: {local.get('verdict')} ({local.get('confidence')})")
    lines.append(f"- Alpha score: {local.get('alpha_score')}")
    lines.append(f"- Sharpe: {local.get('sharpe')}")
    lines.append(f"- Fitness: {local.get('fitness')}")
    shadow = local.get("surrogate_shadow", {})
    if shadow.get("status") == "ready":
        lines.append(
            f"- Surrogate shadow: {shadow.get('preview_verdict')} "
            f"(fitness={shadow.get('predicted_fitness')}, "
            f"sharpe={shadow.get('predicted_sharpe')}, "
            f"rows={shadow.get('training_rows')}, "
            f"alignment={shadow.get('alignment')})"
        )
    else:
        lines.append(f"- Surrogate shadow: {shadow.get('status', 'unavailable')}")
    lines.append(f"- Surrogate selector penalty: {display_metric(candidate.get('surrogate_shadow_penalty'), default='0.0')}")
    lines.append(f"- Confidence score: {candidate.get('confidence_score')}")
    lines.append(f"- Candidate score: {candidate.get('candidate_score')}")
    lines.append(f"- Novelty score: {candidate.get('novelty_score')}")
    selection_mode = candidate.get("selection_mode")
    if selection_mode and selection_mode != "strict":
        if selection_mode == "relaxed":
            lines.append(f"- Selection mode: relaxed exploration ({candidate.get('selection_reason')})")
        else:
            lines.append(f"- Selection mode: {selection_mode} ({candidate.get('selection_reason')})")
    lines.append(
        f"- Settings robustness: {display_metric(candidate.get('robustness_score'))} "
        f"(pass_rate={display_metric(robustness.get('pass_rate'))}, "
        f"alpha_range={display_metric(robustness.get('min_alpha_score'))}-{display_metric(robustness.get('max_alpha_score'))})"
    )
    lines.append(
        f"- Submitted overlap: {display_metric(candidate.get('submitted_overlap_score'))} "
        f"(diversity={display_metric(candidate.get('submitted_diversity_score'))})"
    )
    lines.append(f"- Submitted contrast: {display_metric(candidate.get('submitted_contrast_score'))}")
    lines.append(f"- Brain feedback penalty: {display_metric(candidate.get('brain_feedback_penalty'))}")
    lines.append(f"- Source crowding penalty: {display_metric(candidate.get('source_crowding_penalty'), default='0.0')}")
    lines.append("")
    lines.append("**Context**")
    lines.append(f"- Source quality: {candidate.get('source_quality_score')}")
    lines.append(f"- Source specificity: {candidate.get('source_specificity_score')}")
    lines.append(f"- Style alignment: {candidate.get('style_alignment_score')}")
    lines.append(f"- Source ideas: {', '.join(candidate.get('source_ideas', []))}")
    lines.append(f"- Risk tags: {', '.join(candidate.get('risk_tags', [])) or 'none'}")
    lines.append(
        f"- Brain feedback context: {candidate.get('brain_feedback_context_mode', 'no_signal')} "
        f"({candidate.get('brain_feedback_candidate_context', 'global')})"
    )
    overlap_reasons = ", ".join(candidate.get("submitted_overlap_reasons", [])) or "none"
    lines.append(f"- Submitted overlap reasons: {overlap_reasons}")
    contrast_reasons = ", ".join(candidate.get("submitted_contrast_reasons", [])) or "none"
    lines.append(f"- Submitted contrast reasons: {contrast_reasons}")
    brain_reasons = ", ".join(candidate.get("brain_feedback_reasons", [])) or "none"
    lines.append(f"- Brain feedback reasons: {brain_reasons}")
    surrogate_reasons = ", ".join(candidate.get("surrogate_shadow_reasons", [])) or "none"
    lines.append(f"- Surrogate shadow reasons: {surrogate_reasons}")
    lines.append("")
    lines.append("**Why**")
    lines.append(candidate.get("why") or "")
    lines.append("")
    lines.append("**Thinking**")
    lines.append(candidate.get("thinking") or "")
    lines.append("")


def _learned_seed_candidates(payload: dict, selected: list[dict], watchlist: list[dict], top: int) -> list[dict]:
    selected_expressions = {item.get("expression") for item in [*selected, *watchlist]}
    learned = []
    for candidate in payload.get("batch", {}).get("candidates", []):
        if not candidate.get("learned_seed"):
            continue
        if candidate.get("expression") in selected_expressions:
            continue
        learned.append(candidate)
    learned.sort(
        key=lambda item: (
            float(item.get("selection_rank_score", 0.0)),
            float(item.get("candidate_score", 0.0)),
            float(item.get("confidence_score", 0.0)),
            float(item.get("local_metrics", {}).get("alpha_score", 0.0)),
        ),
        reverse=True,
    )
    return learned[:top]


def render_markdown(
    payload: dict,
    selected: list[dict],
    rejected: list[dict],
    *,
    top: int,
    watchlist: list[dict] | None = None,
) -> str:
    watchlist = watchlist or []
    strict_selected = [candidate for candidate in selected if candidate.get("selection_mode") == "strict"]
    relaxed_selected = [candidate for candidate in selected if candidate.get("selection_mode") == "relaxed"]
    reportable_selected = payload.get("reportable_selected") or []
    withheld_strict_count = max(0, len(strict_selected) - len(reportable_selected))
    show_secondary_sections = not reportable_selected
    learned_seed_candidates = _learned_seed_candidates(payload, selected, watchlist, top)
    lines = ["# Daily Scout Alpha Picks", ""]
    lines.append(f"- Source Inputs: {', '.join(payload.get('batch', {}).get('source_inputs', []))}")
    lines.append("- Remote WorldQuant submission: disabled")
    lines.append(f"- Ideas fetched: {len(payload.get('ideas', []))}")
    lines.append(f"- Candidates scored locally: {len(payload.get('batch', {}).get('candidates', []))}")
    lines.append(f"- Top picks requested: {top}")
    lines.append(f"- Top picks returned: {len(selected)}")
    lines.append(f"- Strict picks returned: {payload.get('batch', {}).get('strict_count', len(strict_selected))}")
    lines.append(f"- Reportable picks: {payload.get('batch', {}).get('reportable_count', len(reportable_selected))}")
    if len(reportable_selected) > top > 0:
        lines.append(f"- Report overflow picks: {len(reportable_selected) - top}")
    lines.append(f"- Relaxed exploration picks: {payload.get('batch', {}).get('relaxed_fill_count', 0)}")
    lines.append(f"- Search breadth: {payload.get('settings', {}).get('search_breadth', 'standard')}")
    lines.append(f"- Query profiles used: {payload.get('settings', {}).get('query_profile_count', 0)}")
    lines.append(f"- Diversity weight: {payload.get('settings', {}).get('diversity_weight', DEFAULT_DIVERSITY_WEIGHT)}")
    lines.append(
        "- Report gate: "
        f"alpha_score>={payload.get('settings', {}).get('report_min_alpha_score', DEFAULT_REPORT_MIN_ALPHA_SCORE)}, "
        f"confidence_score>={payload.get('settings', {}).get('report_min_confidence_score', DEFAULT_REPORT_MIN_CONFIDENCE_SCORE)}, "
        f"robustness_score>={payload.get('settings', {}).get('report_min_robustness_score', DEFAULT_REPORT_MIN_ROBUSTNESS_SCORE)}"
    )
    lines.append(f"- Archive frequency: {payload.get('settings', {}).get('archive_frequency', DEFAULT_ARCHIVE_FREQUENCY)}")
    lines.append(f"- Fallback mode: {payload.get('batch', {}).get('fallback_mode')}")
    if payload.get("run_timestamp"):
        lines.append(f"- Run time: {payload.get('run_timestamp')}")
    lines.append(f"- History archive: {DEFAULT_HISTORY}")
    lines.append(f"- Scout memory: {DEFAULT_MEMORY}")
    brain_feedback_status = payload.get("brain_feedback_status", {})
    lines.append(f"- Brain feedback status: {brain_feedback_status.get('status', 'unknown')}")
    if brain_feedback_status.get("health"):
        lines.append(f"- Brain feedback health: {brain_feedback_status.get('health')}")
    if brain_feedback_status.get("message"):
        lines.append(f"- Brain feedback note: {brain_feedback_status.get('message')}")
    report_status = payload.get("report_status", {})
    if report_status:
        lines.append(
            f"- Report status: {'published' if report_status.get('published') else 'skipped'}"
            + (f" ({report_status.get('reason')})" if report_status.get("reason") else "")
        )
        if report_status.get("interval_minutes"):
            lines.append(f"- Report interval (minutes): {report_status.get('interval_minutes')}")
    archive_status = payload.get("archive_status", {})
    if archive_status:
        lines.append(
            f"- Archive status: {'written' if archive_status.get('written') else 'skipped'}"
            + (f" ({archive_status.get('reason')})" if archive_status.get("reason") else "")
        )
    lines.append("")

    self_learning = payload.get("self_learning", {})
    if self_learning:
        lines.append("## Self-Learning")
        zip_learning = self_learning.get("zip", {})
        if zip_learning:
            lines.append(
                f"- ZIP learning: {zip_learning.get('status', 'unknown')} "
                f"(seed expressions={zip_learning.get('seed_expression_count', 0)}, operators={zip_learning.get('operator_count', 0)})"
            )
        github_readme = self_learning.get("github_readme", {})
        if github_readme:
            lines.append(
                f"- GitHub README learning: attempted={github_readme.get('attempted', 0)}, "
                f"succeeded={github_readme.get('succeeded', 0)}, failed={github_readme.get('failed', 0)}, "
                f"seed expressions={github_readme.get('seed_expression_count', 0)}"
            )
        archived_paths = payload.get("archived_paths", {})
        if archive_status.get("written") and archived_paths.get("root"):
            lines.append(f"- Archived run folder: {archived_paths.get('root')}")
        lines.append("")

    submitted_summary = payload.get("submitted_context", {})
    if submitted_summary.get("count"):
        lines.append("## Submitted Library Pressure")
        lines.append(f"- Submitted alphas loaded: {submitted_summary.get('count')}")
        if submitted_summary.get("top_alpha_types"):
            lines.append(f"- Alpha types: {', '.join(submitted_summary.get('top_alpha_types', []))}")
        if submitted_summary.get("top_style_tags"):
            lines.append(f"- Dominant style tags: {', '.join(submitted_summary.get('top_style_tags', []))}")
        lines.append("")

    brain_feedback_summary = payload.get("brain_feedback", {})
    if brain_feedback_summary.get("count"):
        lines.append("## Real Brain Feedback")
        lines.append(f"- Simulation rows analyzed: {brain_feedback_summary.get('count')}")
        lines.append(f"- Blocked exact skeletons: {brain_feedback_summary.get('blocked_skeleton_count')}")
        lines.append(f"- Context-aware rows: {brain_feedback_summary.get('context_row_count', 0)}")
        lines.append(f"- Distinct settings contexts: {brain_feedback_summary.get('distinct_context_count', 0)}")
        if brain_feedback_summary.get("top_weak_families"):
            lines.append(f"- Weak families: {', '.join(brain_feedback_summary.get('top_weak_families', []))}")
        if brain_feedback_summary.get("top_preferred_families"):
            lines.append(f"- Preferred families: {', '.join(brain_feedback_summary.get('top_preferred_families', []))}")
        if brain_feedback_summary.get("top_weak_alpha_types"):
            lines.append(f"- Weak alpha types: {', '.join(brain_feedback_summary.get('top_weak_alpha_types', []))}")
        if brain_feedback_summary.get("top_weak_style_tags"):
            lines.append(f"- Weak style tags: {', '.join(brain_feedback_summary.get('top_weak_style_tags', []))}")
        lines.append("")

    lines.append("## Ready To Submit")
    if not reportable_selected:
        lines.append("No alpha cleared the report gate this round.")
        if withheld_strict_count:
            lines.append(f"- Strict picks held back from reporting: {withheld_strict_count}")
        lines.append("")
    else:
        lines.append(f"- Reportable picks: {len(reportable_selected)}")
        if withheld_strict_count:
            lines.append(f"- Strict picks held back from reporting: {withheld_strict_count}")
        lines.append("")
        for index, candidate in enumerate(reportable_selected, start=1):
            append_candidate_markdown_block(lines, candidate, heading=f"### #{index} {candidate.get('thesis')}")
        lines.append("")

    if relaxed_selected and show_secondary_sections:
        lines.append("## Explore / Verify First")
        lines.append(f"- Relaxed picks: {len(relaxed_selected)}")
        lines.append("- These candidates filled a strict shortfall and should be verified first in BRAIN.")
        lines.append("")
        for index, candidate in enumerate(relaxed_selected[:top], start=1):
            append_candidate_markdown_block(lines, candidate, heading=f"### #{index} {candidate.get('thesis')}")
        lines.append("")

    if watchlist and show_secondary_sections:
        lines.append("## Watchlist")
        for candidate in watchlist[:top]:
            local = candidate.get("local_metrics", {})
            robustness = candidate.get("settings_robustness", {})
            settings = parse_settings_label(candidate.get("settings") or "")
            lines.append(f"### {candidate.get('thesis')}")
            lines.append("")
            lines.append("```text")
            lines.append(pretty_format_expression(candidate.get("expression") or ""))
            lines.append("```")
            if settings:
                lines.append(f"- Universe: {settings.get('Universe')}")
                if settings.get("Decay"):
                    lines.append(f"- Decay: {settings.get('Decay')}")
                if settings.get("Delay"):
                    lines.append(f"- Delay: {settings.get('Delay')}")
                if settings.get("Truncation"):
                    lines.append(f"- Truncation: {settings.get('Truncation')}")
                if settings.get("Neutralization"):
                    lines.append(f"- Neutralization: {settings.get('Neutralization')}")
            lines.append(f"- Verdict: {local.get('verdict')} ({local.get('confidence')})")
            lines.append(f"- Alpha score: {local.get('alpha_score')}")
            shadow = local.get("surrogate_shadow", {})
            if shadow.get("status") == "ready":
                lines.append(
                    f"- Surrogate shadow: {shadow.get('preview_verdict')} "
                    f"(fitness={shadow.get('predicted_fitness')}, sharpe={shadow.get('predicted_sharpe')})"
                )
            else:
                lines.append(f"- Surrogate shadow: {shadow.get('status', 'unavailable')}")
            lines.append(f"- Surrogate selector penalty: {display_metric(candidate.get('surrogate_shadow_penalty'), default='0.0')}")
            lines.append(f"- Confidence score: {candidate.get('confidence_score')}")
            lines.append(f"- Settings robustness: {display_metric(candidate.get('robustness_score'))} (pass_rate={display_metric(robustness.get('pass_rate'))})")
            lines.append(f"- Submitted overlap: {display_metric(candidate.get('submitted_overlap_score'))}")
            lines.append(f"- Submitted contrast: {display_metric(candidate.get('submitted_contrast_score'))}")
            lines.append(f"- Brain feedback penalty: {display_metric(candidate.get('brain_feedback_penalty'))}")
            lines.append(f"- Source crowding penalty: {display_metric(candidate.get('source_crowding_penalty'), default='0.0')}")
            lines.append(f"- Reason: {candidate.get('selection_reason')}")
            lines.append("")
        lines.append("")

    if learned_seed_candidates and show_secondary_sections:
        lines.append("## Learned Seeds / Verify First")
        lines.append("- These candidates were learned from local ZIP seeds or GitHub README knowledge before normal template generation.")
        lines.append("")
        for index, candidate in enumerate(learned_seed_candidates, start=1):
            append_candidate_markdown_block(lines, candidate, heading=f"### #{index} {candidate.get('thesis')}")
        lines.append("")

    if rejected and show_secondary_sections:
        lines.append("## Held Out")
        for item in rejected[:8]:
            lines.append(f"- {item.get('expression')}: {item.get('reason')}")
        lines.append("")

    lines.append("## Files")
    lines.append("- artifacts/trinh_sat_hang_ngay.md")
    lines.append("- artifacts/_trinh_sat/du_lieu.json")
    lines.append("- artifacts/_trinh_sat/bieu_thuc_da_chon.txt")
    lines.append("- artifacts/_trinh_sat/lich_su.jsonl")
    lines.append("- artifacts/_trinh_sat/bo_nho.json")
    lines.append("- artifacts/_trinh_sat/phan_hoi_brain.json")
    lines.append("- artifacts/_trinh_sat/kien_thuc_tu_hoc.json")
    lines.append("- artifacts/_trinh_sat/mo_hinh_surrogate_shadow_v1.joblib")
    lines.append("- artifacts/alpha_da_gui.json")
    return "\n".join(lines)


def render_feedback_health_failure_markdown(*, run_timestamp: str, brain_feedback_status: dict) -> str:
    lines = ["# Daily Scout Alpha Picks", ""]
    lines.append(f"- Run time: {run_timestamp}")
    lines.append(f"- Brain feedback status: {brain_feedback_status.get('status', 'unknown')}")
    lines.append(f"- Brain feedback health: {brain_feedback_status.get('health', 'unknown')}")
    if brain_feedback_status.get("message"):
        lines.append(f"- Brain feedback note: {brain_feedback_status.get('message')}")
    if brain_feedback_status.get("missing_context_columns"):
        lines.append(
            "- Missing context columns: "
            + ", ".join(brain_feedback_status.get("missing_context_columns", []))
        )
    lines.append("")
    lines.append("## Feedback Health Check")
    lines.append("- Scout halted before retrieval/generation because Brain feedback is not healthy enough for loop mode.")
    lines.append(f"- Recommended action: {brain_feedback_status.get('recommended_action', 'fix_feedback')}")
    lines.append("")
    lines.append("## Files")
    lines.append("- artifacts/trinh_sat_hang_ngay.md")
    lines.append("- artifacts/_trinh_sat/du_lieu.json")
    lines.append("- artifacts/_trinh_sat/phan_hoi_brain.json")
    return "\n".join(lines)


def load_history_index(csv_path: str | None) -> HistoryIndex:
    if csv_path:
        return HistoryIndex.from_csv(csv_path)
    try:
        resolved = discover_csv(None)
    except FileNotFoundError:
        return HistoryIndex()
    return HistoryIndex.from_csv(resolved)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Scout public factor ideas via APIs, convert them into local alpha candidates, and rank strict daily picks.",
    )
    parser.add_argument("--count", type=int, default=6, help="Maximum number of daily picks to return.")
    parser.add_argument("--openalex-per-query", type=int, default=2, help="How many OpenAlex works to fetch per query.")
    parser.add_argument("--arxiv-per-query", type=int, default=1, help="How many arXiv papers to fetch per query.")
    parser.add_argument("--github-per-query", type=int, default=1, help="How many GitHub repositories to fetch per query.")
    parser.add_argument("--github-query-limit", type=int, default=DEFAULT_GITHUB_QUERY_LIMIT, help="Maximum number of query profiles to send to GitHub search.")
    parser.add_argument("--github-readme-limit", type=int, default=DEFAULT_GITHUB_README_LIMIT, help="How many GitHub READMEs to learn from after repo discovery.")
    parser.add_argument("--public-api-timeout", type=int, default=DEFAULT_PUBLIC_API_TIMEOUT, help="Timeout in seconds for each public API request.")
    parser.add_argument("--public-api-max-retries", type=int, default=DEFAULT_PUBLIC_API_MAX_RETRIES, help="Maximum retries for retryable public API failures.")
    parser.add_argument("--request-delay-seconds", type=float, default=DEFAULT_PUBLIC_API_REQUEST_DELAY_SECONDS, help="Minimum delay between uncached public API requests.")
    parser.add_argument("--max-api-requests-per-run", type=int, default=DEFAULT_PUBLIC_API_MAX_REQUESTS_PER_RUN, help="Maximum uncached public API requests allowed in a single scout round.")
    parser.add_argument("--quota-cooldown-seconds", type=float, default=DEFAULT_PUBLIC_API_QUOTA_COOLDOWN_SECONDS, help="Cooldown applied after repeated 429 responses from public APIs.")
    parser.add_argument(
        "--learn-zip-path",
        default=str(DEFAULT_WORLDQUANT_MINER_ZIP) if DEFAULT_WORLDQUANT_MINER_ZIP.exists() else "",
        help="Optional ZIP file to mine for learned alpha seeds, operators, and docs.",
    )
    parser.add_argument("--zip-seed-limit", type=int, default=DEFAULT_ZIP_SEED_LIMIT, help="Maximum number of mined seed expressions to import from the ZIP source.")
    parser.add_argument("--search-breadth", choices=["focused", "standard", "wide", "explore"], default="standard", help="How broad the public-idea search space should be.")
    parser.add_argument("--max-query-profiles", type=int, help="Optional cap on how many query profiles to send to OpenAlex/arXiv.")
    parser.add_argument("--memory", default="artifacts/bo_nho_nghien_cuu.json", help="Optional research memory JSON.")
    parser.add_argument("--memory-path", default=str(DEFAULT_MEMORY), help="Scout memory output path.")
    parser.add_argument("--history-path", default=str(DEFAULT_HISTORY), help="Scout history archive path.")
    parser.add_argument("--seed-store", default="initial-population.pkl", help="Optional existing seed store path for duplicate avoidance.")
    parser.add_argument("--submitted-alphas", default=str(DEFAULT_SUBMITTED_ALPHAS), help="Optional JSON file describing already-submitted alpha expressions to avoid.")
    parser.add_argument("--write-brain-feedback", default=str(DEFAULT_BRAIN_FEEDBACK), help="JSON summary of real Brain feedback learned from simulation history.")
    parser.add_argument("--csv", help="Optional local simulation CSV for novelty context.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Markdown output path.")
    parser.add_argument("--write-plan", default=str(DEFAULT_PLAN), help="JSON output path.")
    parser.add_argument("--write-batch", default=str(DEFAULT_BATCH), help="Candidate expressions output path.")
    parser.add_argument("--write-learning", default=str(DEFAULT_SELF_LEARNING), help="JSON output path for learned external knowledge.")
    parser.add_argument("--fetch-cache", default=str(DEFAULT_FETCH_CACHE), help="JSON cache for public API fetch results.")
    parser.add_argument("--report-state-path", default=str(DEFAULT_REPORT_STATE), help="JSON state used to enforce report publishing cadence across scout runs.")
    parser.add_argument("--archive-root", default=str(DEFAULT_REPORT_ARCHIVE_ROOT), help="Folder root for timestamped scout run archives.")
    parser.add_argument(
        "--archive-frequency",
        choices=["hour", "run"],
        default=DEFAULT_ARCHIVE_FREQUENCY,
        help="How often archived scout folders should be rotated. 'hour' keeps one folder per hour, 'run' keeps one per run.",
    )
    parser.add_argument(
        "--report-interval-minutes",
        type=int,
        default=DEFAULT_REPORT_INTERVAL_MINUTES,
        help="Minimum minutes between published scout reports. 0 publishes every run.",
    )
    parser.add_argument("--min-alpha-score", type=float, default=68.0, help="Minimum alpha_score for strict daily picks.")
    parser.add_argument("--min-confidence-score", type=float, default=0.58, help="Minimum confidence_score for strict daily picks.")
    parser.add_argument(
        "--report-min-alpha-score",
        type=float,
        default=DEFAULT_REPORT_MIN_ALPHA_SCORE,
        help="Minimum alpha_score required before a strict pick is shown in the report or archived batch.",
    )
    parser.add_argument(
        "--report-min-confidence-score",
        type=float,
        default=DEFAULT_REPORT_MIN_CONFIDENCE_SCORE,
        help="Minimum confidence_score required before a strict pick is shown in the report or archived batch.",
    )
    parser.add_argument(
        "--report-min-robustness-score",
        type=float,
        default=DEFAULT_REPORT_MIN_ROBUSTNESS_SCORE,
        help="Minimum robustness_score required before a strict pick is shown in the report or archived batch.",
    )
    parser.add_argument("--diversity-weight", type=float, default=DEFAULT_DIVERSITY_WEIGHT, help="How strongly final selection should favor diverse families, style tags, and skeletons.")
    parser.add_argument("--include-watchlist", action="store_true", help="Include lower-confidence watchlist candidates in the markdown output.")
    parser.add_argument(
        "--require-feedback-healthy",
        action="store_true",
        help="Halt before retrieval/generation when Brain feedback is missing, malformed, or missing context columns.",
    )
    args = parser.parse_args()

    run_dt = datetime.now()
    run_label = run_dt.strftime("%Y-%m-%d %Hh %M'")
    run_date = run_dt.strftime("%Y-%m-%d %H:%M:%S")
    generic_memory = load_memory(args.memory)
    historical_records = load_history_records(args.history_path)
    scout_memory = aggregate_scout_memory(historical_records) if historical_records else load_json(args.memory_path)
    submitted_alphas = load_submitted_alphas(args.submitted_alphas)
    submitted_context = build_submitted_alpha_context(submitted_alphas)
    brain_feedback_rows, brain_feedback_status = load_brain_feedback_rows(args.csv)
    brain_feedback_context = build_brain_feedback_context(brain_feedback_rows)
    brain_feedback_status = assess_brain_feedback_health(brain_feedback_rows, brain_feedback_status, brain_feedback_context)

    if args.require_feedback_healthy and brain_feedback_status.get("hard_block"):
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plan_path = Path(args.write_plan)
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        batch_path = Path(args.write_batch)
        batch_path.parent.mkdir(parents=True, exist_ok=True)
        memory_path = Path(args.memory_path)
        memory_path.parent.mkdir(parents=True, exist_ok=True)
        brain_feedback_path = Path(args.write_brain_feedback)
        brain_feedback_path.parent.mkdir(parents=True, exist_ok=True)
        learning_path = Path(args.write_learning)
        learning_path.parent.mkdir(parents=True, exist_ok=True)
        report_state_path = Path(args.report_state_path)
        report_state_path.parent.mkdir(parents=True, exist_ok=True)

        learning_payload = {
            "generated_at": run_date,
            "zip": {"status": "skipped"},
            "github_readme": {"status": "skipped", "attempted": 0, "succeeded": 0, "failed": 0, "seed_expression_count": 0},
            "sources": [],
        }
        failure_report_status = {
            "published": True,
            "reason": "feedback_health_failure",
            "interval_minutes": max(0, int(args.report_interval_minutes)),
            "last_published_at": run_dt.isoformat(timespec="seconds"),
            "next_publish_after": "",
            "minutes_until_next": 0,
        }
        payload, selected, watchlist, rejected = build_payload(
            ideas=[],
            candidates=[],
            memory=generic_memory,
            count=args.count,
            seed_store={},
            submitted_context=submitted_context,
            brain_feedback_context=brain_feedback_context,
            search_breadth=args.search_breadth,
            query_profile_count=0,
            min_alpha_score=args.min_alpha_score,
            min_confidence_score=args.min_confidence_score,
            report_min_alpha_score=args.report_min_alpha_score,
            report_min_confidence_score=args.report_min_confidence_score,
            report_min_robustness_score=args.report_min_robustness_score,
            include_watchlist=args.include_watchlist,
            fallback_mode=False,
            diversity_weight=_clip(args.diversity_weight, 0.0, 1.0),
            brain_feedback_status=brain_feedback_status,
            self_learning_status={"zip": learning_payload["zip"], "github_readme": learning_payload["github_readme"]},
            report_status=failure_report_status,
            run_timestamp=run_label,
            archive_frequency=args.archive_frequency,
        )
        payload["halted"] = True
        payload["halt_reason"] = "feedback_health_check"
        payload["batch"]["notes"] = [
            "Scout halted before retrieval/generation because Brain feedback failed the required health check.",
            "Fix the simulation CSV schema/context coverage before rerunning loop mode.",
        ]
        payload["archived_paths"] = {}
        payload["archive_status"] = {
            "written": False,
            "reason": "feedback_health_failure",
            "frequency": args.archive_frequency,
        }
        markdown = render_feedback_health_failure_markdown(
            run_timestamp=run_label,
            brain_feedback_status=brain_feedback_status,
        )
        try:
            archive_paths = write_report_archive(
                run_timestamp=run_dt,
                markdown=markdown,
                payload={
                    **payload,
                    "archived_paths": {
                        name: str(path)
                        for name, path in build_report_archive_paths(
                            run_dt,
                            archive_root=args.archive_root,
                            archive_frequency=args.archive_frequency,
                        ).items()
                    },
                    "archive_status": {
                        "written": True,
                        "reason": "feedback_health_failure",
                        "frequency": args.archive_frequency,
                    },
                },
                selected_batch_text="",
                scout_memory=scout_memory,
                brain_feedback_context=brain_feedback_context,
                self_learning_payload=learning_payload,
                archive_root=args.archive_root,
                archive_frequency=args.archive_frequency,
            )
        except Exception as exc:
            print(f"[scout] Archive write skipped ({type(exc).__name__}: {exc})")
            payload["archived_paths"] = {}
            payload["archive_status"] = {
                "written": False,
                "reason": "archive_write_failed",
                "frequency": args.archive_frequency,
            }
        else:
            payload["archived_paths"] = archive_paths
            payload["archive_status"] = {
                "written": True,
                "reason": "feedback_health_failure",
                "frequency": args.archive_frequency,
            }

        _atomic_write_text(output_path, markdown)
        save_json(plan_path, payload)
        _atomic_write_text(batch_path, "")
        save_json(memory_path, scout_memory)
        save_json(brain_feedback_path, brain_feedback_context)
        save_json(learning_path, learning_payload)
        save_json(report_state_path, build_report_state(run_dt, report_status=failure_report_status))
        print(markdown)
        return 2

    query_profiles = build_query_profiles(
        generic_memory,
        scout_memory,
        search_breadth=args.search_breadth,
        max_profiles=args.max_query_profiles,
    )
    fetch_cache = load_fetch_cache(args.fetch_cache)
    openalex_floor = 4 if args.search_breadth == "explore" else 3 if args.search_breadth == "wide" else 2
    arxiv_floor = 3 if args.search_breadth == "explore" else 2 if args.search_breadth == "wide" else 1
    github_floor = 2 if args.search_breadth == "explore" else 0
    github_query_floor = 10 if args.search_breadth == "explore" else 0
    github_readme_floor = 6 if args.search_breadth == "explore" else 0
    openalex_per_query = max(args.openalex_per_query, openalex_floor)
    arxiv_per_query = max(args.arxiv_per_query, arxiv_floor)
    github_per_query = max(github_floor, args.github_per_query)
    github_query_limit = max(github_query_floor, args.github_query_limit)
    github_readme_limit = max(github_readme_floor, args.github_readme_limit)
    request_throttle = ScoutRequestThrottle(
        max_requests=args.max_api_requests_per_run,
        request_delay_seconds=args.request_delay_seconds,
        quota_cooldown_seconds=args.quota_cooldown_seconds,
    )

    openalex_ideas, openalex_status = fetch_openalex_ideas(
        query_profiles,
        per_query=openalex_per_query,
        timeout=args.public_api_timeout,
        fetch_cache=fetch_cache,
        request_throttle=request_throttle,
        max_retries=args.public_api_max_retries,
    )
    arxiv_ideas, arxiv_status = fetch_arxiv_ideas(
        query_profiles,
        per_query=arxiv_per_query,
        timeout=args.public_api_timeout,
        fetch_cache=fetch_cache,
        request_throttle=request_throttle,
        max_retries=args.public_api_max_retries,
    )
    github_ideas, github_status = fetch_github_ideas(
        query_profiles,
        per_query=github_per_query,
        timeout=args.public_api_timeout,
        max_queries=github_query_limit,
        fetch_cache=fetch_cache,
        request_throttle=request_throttle,
        max_retries=args.public_api_max_retries,
    )
    github_readme_ideas, github_readme_status = fetch_github_readme_ideas(
        github_ideas,
        limit=github_readme_limit,
        timeout=args.public_api_timeout,
        fetch_cache=fetch_cache,
        request_throttle=request_throttle,
        max_retries=args.public_api_max_retries,
    )
    zip_ideas, zip_learning_status = fetch_zip_learned_ideas(args.learn_zip_path, max_seed_ideas=max(0, args.zip_seed_limit))
    book_ideas = build_book_ideas()
    api_ideas = dedupe_ideas(book_ideas + openalex_ideas + arxiv_ideas + github_ideas + github_readme_ideas + zip_ideas)
    fallback_mode = False
    if not api_ideas:
        api_ideas = build_fallback_ideas()
        fallback_mode = True

    ideas = filter_relevant_ideas(api_ideas, scout_memory)
    if not ideas:
        ideas = build_fallback_ideas()
        fallback_mode = True
        for idea in ideas:
            idea["idea_profile"] = infer_idea_profile(idea, scout_memory=scout_memory)

    history_index = load_history_index(args.csv)
    seed_store = load_seed_store(args.seed_store)
    candidates = build_candidates(
        ideas,
        memory=generic_memory,
        scout_memory=scout_memory,
        history_index=history_index,
        submitted_context=submitted_context,
        brain_feedback_context=brain_feedback_context,
    )
    payload, selected, watchlist, rejected = build_payload(
        ideas=ideas,
        candidates=candidates,
        memory=generic_memory,
        count=args.count,
        seed_store=seed_store,
        submitted_context=submitted_context,
        brain_feedback_context=brain_feedback_context,
        search_breadth=args.search_breadth,
        query_profile_count=len(query_profiles),
        min_alpha_score=args.min_alpha_score,
        min_confidence_score=args.min_confidence_score,
        report_min_alpha_score=args.report_min_alpha_score,
        report_min_confidence_score=args.report_min_confidence_score,
        report_min_robustness_score=args.report_min_robustness_score,
        include_watchlist=args.include_watchlist,
        fallback_mode=fallback_mode,
        diversity_weight=_clip(args.diversity_weight, 0.0, 1.0),
        brain_feedback_status=brain_feedback_status,
        self_learning_status={"zip": zip_learning_status, "github_readme": github_readme_status},
        run_timestamp=run_label,
        archive_frequency=args.archive_frequency,
    )
    payload["api_status"] = {
        "openalex": openalex_status,
        "arxiv": arxiv_status,
        "github": github_status,
        "github_readme": github_readme_status,
        "zip_learning": zip_learning_status,
    }

    history_records = build_history_records(payload["batch"]["candidates"], run_date=run_date)
    append_history_records(args.history_path, history_records)
    updated_scout_memory = aggregate_scout_memory(load_history_records(args.history_path))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path = Path(args.write_plan)
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    batch_path = Path(args.write_batch)
    batch_path.parent.mkdir(parents=True, exist_ok=True)
    memory_path = Path(args.memory_path)
    memory_path.parent.mkdir(parents=True, exist_ok=True)
    brain_feedback_path = Path(args.write_brain_feedback)
    brain_feedback_path.parent.mkdir(parents=True, exist_ok=True)
    learning_path = Path(args.write_learning)
    learning_path.parent.mkdir(parents=True, exist_ok=True)
    fetch_cache_path = Path(args.fetch_cache)
    fetch_cache_path.parent.mkdir(parents=True, exist_ok=True)
    report_state_path = Path(args.report_state_path)
    report_state_path.parent.mkdir(parents=True, exist_ok=True)

    payload["scout_memory"] = updated_scout_memory
    reportable_selected = payload.get("reportable_selected", [])
    report_state = load_json(report_state_path)
    report_status = assess_report_publish_status(
        run_dt,
        reportable_count=len(reportable_selected),
        report_interval_minutes=args.report_interval_minutes,
        report_state=report_state,
    )
    payload["report_status"] = report_status
    selected_batch_text = "\n".join(item.get("expression") or "" for item in reportable_selected) + ("\n" if reportable_selected else "")
    learning_payload = {
        "generated_at": run_date,
        "zip": zip_learning_status,
        "github_readme": github_readme_status,
        "sources": [
            {
                "source": idea.get("source"),
                "title": idea.get("title"),
                "query": idea.get("query"),
                "seed_expression_count": len(idea.get("seed_expressions", [])),
            }
            for idea in [*zip_ideas[: args.zip_seed_limit], *github_readme_ideas[: github_readme_limit]]
        ],
    }
    payload["run_timestamp"] = run_label
    payload["archived_paths"] = {}
    payload["archive_status"] = {
        "written": False,
        "reason": report_status.get("reason", "no_reportable_pick"),
        "frequency": args.archive_frequency,
    }
    markdown = render_markdown(payload, selected, rejected, top=args.count, watchlist=watchlist if args.include_watchlist else [])

    if report_status.get("published") and reportable_selected:
        candidate_archive_paths = build_report_archive_paths(
            run_dt,
            archive_root=args.archive_root,
            archive_frequency=args.archive_frequency,
        )
        archived_payload = dict(payload)
        archived_payload["archived_paths"] = {name: str(path) for name, path in candidate_archive_paths.items()}
        archived_payload["archive_status"] = {
            "written": True,
            "reason": report_status.get("reason", "reportable_pick_found"),
            "frequency": args.archive_frequency,
        }
        archived_markdown = render_markdown(
            archived_payload,
            selected,
            rejected,
            top=args.count,
            watchlist=watchlist if args.include_watchlist else [],
        )
        try:
            archive_paths = write_report_archive(
                run_timestamp=run_dt,
                markdown=archived_markdown,
                payload=archived_payload,
                selected_batch_text=selected_batch_text,
                scout_memory=updated_scout_memory,
                brain_feedback_context=brain_feedback_context,
                self_learning_payload=learning_payload,
                archive_root=args.archive_root,
                archive_frequency=args.archive_frequency,
            )
        except Exception as exc:
            print(f"[scout] Archive write skipped ({type(exc).__name__}: {exc})")
            payload["archived_paths"] = {}
            payload["archive_status"] = {
                "written": False,
                "reason": "archive_write_failed",
                "frequency": args.archive_frequency,
            }
            markdown = render_markdown(
                payload,
                selected,
                rejected,
                top=args.count,
                watchlist=watchlist if args.include_watchlist else [],
            )
        else:
            archived_payload["archived_paths"] = archive_paths
            payload = archived_payload
            markdown = archived_markdown

    save_json(plan_path, payload)
    save_json(memory_path, updated_scout_memory)
    save_json(brain_feedback_path, brain_feedback_context)
    save_json(learning_path, learning_payload)
    save_fetch_cache(fetch_cache_path, fetch_cache)
    if report_status.get("published"):
        _atomic_write_text(output_path, markdown)
        _atomic_write_text(batch_path, selected_batch_text)
        save_json(report_state_path, build_report_state(run_dt, report_status=report_status))

    if report_status.get("published"):
        print(markdown)
    else:
        next_publish_after = report_status.get("next_publish_after") or "later"
        print(
            "[scout] Report skipped "
            f"({report_status.get('reason', 'unknown')}); "
            f"reportable_picks={len(reportable_selected)}, "
            f"next_publish_after={next_publish_after}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
