"""Helpers for consistent candidate lineage across planner, scout, fix, evaluate, and seed flows."""

from __future__ import annotations

from copy import deepcopy


def _clean_text(value) -> str | None:
    text = str(value or "").strip()
    return text or None


def _clean_list(values) -> list[str]:
    if values is None:
        return []
    if isinstance(values, (str, bytes)):
        values = [values]
    elif not isinstance(values, (list, tuple, set)):
        values = [values]

    items = []
    seen = set()
    for value in values:
        text = _clean_text(value)
        if not text or text in seen:
            continue
        seen.add(text)
        items.append(text)
    return items


def _compact_dict(payload: dict) -> dict:
    compact = {}
    for key, value in payload.items():
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        if isinstance(value, (list, dict, tuple, set)) and not value:
            continue
        compact[key] = value
    return compact


def _is_blank(value) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, dict, tuple, set)):
        return not value
    return False


def _unique_dicts(items: list[dict]) -> list[dict]:
    unique = []
    seen = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        normalized = _compact_dict(item)
        if not normalized:
            continue
        key = tuple(sorted((name, str(value)) for name, value in normalized.items()))
        if key in seen:
            continue
        seen.add(key)
        unique.append(normalized)
    return unique


def canonical_lineage_origin(source: str | None, *, seed_source: str | None = None) -> str:
    source_text = (_clean_text(seed_source) or _clean_text(source) or "unknown").lower()
    if seed_source:
        return "seed"
    if source_text in {"planner"}:
        return "planner"
    if source_text in {"scout"}:
        return "scout"
    if source_text in {"auto_fix_rewrite", "auto_fix", "fix", "rewrite"}:
        return "fix"
    if source_text in {"seed", "submit_ready_report", "seed_store"}:
        return "seed"
    return source_text


def infer_family_components(candidate: dict) -> list[str]:
    components = candidate.get("family_components")
    if not components:
        components = candidate.get("thesis_family_ids")
    if not components:
        primary = candidate.get("family_id") or candidate.get("thesis_id")
        components = [primary] if primary else []
    return _clean_list(components)


def _parent_entry(
    candidate: dict,
    *,
    default_parent_expression: str | None = None,
    default_parent_alpha_id: str | None = None,
    default_parent_candidate_id: str | None = None,
    default_parent_signature: str | None = None,
    default_parent_source: str | None = None,
) -> dict | None:
    parent = _compact_dict(
        {
            "candidate_id": candidate.get("parent_candidate_id") or default_parent_candidate_id,
            "candidate_signature": candidate.get("parent_candidate_signature") or default_parent_signature,
            "expression": candidate.get("parent_expression") or candidate.get("source_expression") or default_parent_expression,
            "alpha_id": candidate.get("parent_alpha_id") or candidate.get("source_alpha_id") or default_parent_alpha_id,
            "source": candidate.get("parent_source") or default_parent_source,
        }
    )
    return parent or None


def _merge_stage_result(primary: dict | None, secondary: dict | None) -> dict:
    merged = deepcopy(secondary) if isinstance(secondary, dict) else {}
    if isinstance(primary, dict):
        for key, value in primary.items():
            if key == "fail_reasons":
                continue
            if value is None:
                continue
            if isinstance(value, str) and not value.strip():
                continue
            if isinstance(value, (list, dict, tuple, set)) and not value:
                continue
            merged[key] = deepcopy(value)
    fail_reasons = _clean_list((secondary or {}).get("fail_reasons"))
    fail_reasons.extend(_clean_list((primary or {}).get("fail_reasons")))
    if fail_reasons:
        merged["fail_reasons"] = _clean_list(fail_reasons)
    elif "fail_reasons" in merged:
        merged.pop("fail_reasons", None)
    return merged


def build_planning_stage_result(candidate: dict) -> dict:
    quality_label = _clean_text(candidate.get("quality_label"))
    if quality_label is None:
        quality_label = "qualified" if candidate.get("qualified") else "watchlist"
    return _compact_dict(
        {
            "seed_ready": bool(candidate.get("seed_ready")),
            "qualified": bool(candidate.get("qualified")),
            "quality_label": quality_label,
            "fail_reasons": _clean_list(candidate.get("quality_fail_reasons")),
            "candidate_score": candidate.get("candidate_score"),
            "confidence_score": candidate.get("confidence_score"),
        }
    )


def ensure_candidate_lineage(
    candidate: dict,
    *,
    stage_source: str | None = None,
    source_detail: str | None = None,
    default_parent_expression: str | None = None,
    default_parent_alpha_id: str | None = None,
    default_parent_candidate_id: str | None = None,
    default_parent_signature: str | None = None,
    default_parent_source: str | None = None,
    default_hypothesis_id: str | None = None,
    default_hypothesis_label: str | None = None,
    default_family: str | None = None,
    default_family_components: list[str] | None = None,
    default_generation_reason: str | None = None,
    include_planning_stage: bool = True,
) -> dict:
    raw_lineage = candidate.get("lineage")
    lineage = deepcopy(raw_lineage) if isinstance(raw_lineage, dict) else {}

    origin = _clean_text(lineage.get("origin")) or canonical_lineage_origin(
        stage_source or candidate.get("source"),
        seed_source=candidate.get("seed_source"),
    )
    sources = _clean_list(lineage.get("sources"))
    if origin:
        sources.append(origin)
    sources = _clean_list(sources)

    source_details = _clean_list(lineage.get("source_details"))
    for item in (
        source_detail,
        candidate.get("seed_source"),
        candidate.get("source_kind"),
        stage_source,
        candidate.get("source"),
    ):
        text = _clean_text(item)
        if text:
            source_details.append(text)
    source_details = _clean_list(source_details)

    family_components = _clean_list(lineage.get("family_components"))
    if not family_components:
        family_components = _clean_list(default_family_components or infer_family_components(candidate))

    generation_reason = _clean_text(lineage.get("generation_reason")) or _clean_text(default_generation_reason)
    generation_reasons = _clean_list(lineage.get("generation_reasons"))
    if generation_reason:
        generation_reasons.append(generation_reason)
    for item in (candidate.get("selection_reason"), candidate.get("repair_status")):
        text = _clean_text(item)
        if text:
            generation_reasons.append(text)
    generation_reasons = _clean_list(generation_reasons)

    parents = []
    if isinstance(lineage.get("parents"), list):
        parents.extend(item for item in lineage["parents"] if isinstance(item, dict))
    parent = _parent_entry(
        candidate,
        default_parent_expression=default_parent_expression,
        default_parent_alpha_id=default_parent_alpha_id,
        default_parent_candidate_id=default_parent_candidate_id,
        default_parent_signature=default_parent_signature,
        default_parent_source=default_parent_source,
    )
    if parent is not None:
        parents.append(parent)

    stage_results = deepcopy(lineage.get("stage_results")) if isinstance(lineage.get("stage_results"), dict) else {}
    if include_planning_stage:
        stage_results["planning"] = _merge_stage_result(
            build_planning_stage_result(candidate),
            stage_results.get("planning"),
        )

    refreshed = _compact_dict(
        {
            "origin": origin,
            "sources": sources,
            "source_details": source_details,
            "hypothesis_id": _clean_text(lineage.get("hypothesis_id"))
            or _clean_text(default_hypothesis_id)
            or _clean_text(candidate.get("thesis_id"))
            or _clean_text(candidate.get("family_id")),
            "hypothesis_label": _clean_text(lineage.get("hypothesis_label"))
            or _clean_text(default_hypothesis_label)
            or _clean_text(candidate.get("thesis"))
            or _clean_text(candidate.get("family")),
            "family": _clean_text(lineage.get("family"))
            or _clean_text(default_family)
            or _clean_text(candidate.get("family_id"))
            or _clean_text(candidate.get("thesis_id"))
            or (family_components[0] if family_components else None),
            "family_components": family_components,
            "generation_reason": generation_reason or (generation_reasons[0] if generation_reasons else None),
            "generation_reasons": generation_reasons,
            "parents": _unique_dicts(parents),
            "stage_results": stage_results,
        }
    )
    return refreshed


def merge_candidate_lineage(primary: dict | None, secondary: dict | None) -> dict:
    if not isinstance(primary, dict):
        return deepcopy(secondary) if isinstance(secondary, dict) else {}
    if not isinstance(secondary, dict):
        return deepcopy(primary)

    merged = deepcopy(primary)
    merged["sources"] = _clean_list((primary.get("sources") or []) + (secondary.get("sources") or []))
    merged["source_details"] = _clean_list((primary.get("source_details") or []) + (secondary.get("source_details") or []))
    merged["family_components"] = _clean_list((primary.get("family_components") or []) + (secondary.get("family_components") or []))
    merged["generation_reasons"] = _clean_list((primary.get("generation_reasons") or []) + (secondary.get("generation_reasons") or []))
    merged["parents"] = _unique_dicts(list(primary.get("parents") or []) + list(secondary.get("parents") or []))

    for field in ("origin", "hypothesis_id", "hypothesis_label", "family", "generation_reason"):
        if field not in merged or _is_blank(merged.get(field)):
            value = secondary.get(field)
            if not _is_blank(value):
                merged[field] = deepcopy(value)

    primary_stages = primary.get("stage_results") if isinstance(primary.get("stage_results"), dict) else {}
    secondary_stages = secondary.get("stage_results") if isinstance(secondary.get("stage_results"), dict) else {}
    stage_results = {}
    for stage_name in sorted(set(primary_stages) | set(secondary_stages)):
        stage_results[stage_name] = _merge_stage_result(primary_stages.get(stage_name), secondary_stages.get(stage_name))
    if stage_results:
        merged["stage_results"] = stage_results

    return _compact_dict(merged)


def attach_evaluation_stage(
    lineage: dict | None,
    record: dict,
    *,
    submit_ready: bool | None = None,
    qualified: bool | None = None,
    quality_label: str | None = None,
    fail_reasons: list[str] | None = None,
) -> dict:
    refreshed = deepcopy(lineage) if isinstance(lineage, dict) else {}
    stage_results = deepcopy(refreshed.get("stage_results")) if isinstance(refreshed.get("stage_results"), dict) else {}
    metrics = record.get("simulation_result") if isinstance(record.get("simulation_result"), dict) and record.get("simulation_result") else record.get("local_metrics")
    if not isinstance(metrics, dict):
        metrics = {}
    stage_results["evaluation"] = _merge_stage_result(
        {
            "backend": _clean_text(record.get("evaluation_backend")),
            "status": _clean_text(record.get("evaluation_status")),
            "submit_ready": submit_ready if submit_ready is not None else record.get("evaluated_submit_ready"),
            "qualified": qualified if qualified is not None else record.get("qualified"),
            "quality_label": _clean_text(quality_label) or _clean_text(record.get("quality_label")),
            "fail_reasons": _clean_list(
                fail_reasons if fail_reasons is not None else record.get("evaluated_fail_reasons") or record.get("quality_fail_reasons")
            ),
            "verdict": _clean_text(metrics.get("verdict")),
            "alpha_score": metrics.get("alpha_score"),
            "sharpe": metrics.get("sharpe"),
            "fitness": metrics.get("fitness"),
            "turnover": metrics.get("turnover"),
        },
        stage_results.get("evaluation"),
    )
    refreshed["stage_results"] = stage_results
    return _compact_dict(refreshed)


def attach_seed_stage(
    lineage: dict | None,
    candidate: dict,
    *,
    seed_source: str,
    selection_score=None,
    compiled_expression: str | None = None,
) -> dict:
    refreshed = deepcopy(lineage) if isinstance(lineage, dict) else {}
    stage_results = deepcopy(refreshed.get("stage_results")) if isinstance(refreshed.get("stage_results"), dict) else {}
    stage_results["seed"] = _merge_stage_result(
        {
            "selected": True,
            "seed_source": _clean_text(seed_source),
            "selection_score": selection_score,
            "compiled_expression": _clean_text(compiled_expression),
            "thesis_id": _clean_text(candidate.get("thesis_id")),
        },
        stage_results.get("seed"),
    )
    refreshed["stage_results"] = stage_results
    refreshed["sources"] = _clean_list(list(refreshed.get("sources") or []) + ["seed"])
    refreshed["source_details"] = _clean_list(list(refreshed.get("source_details") or []) + [seed_source])
    return _compact_dict(refreshed)
