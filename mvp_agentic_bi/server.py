#!/usr/bin/env python3
import base64
import binascii
import csv
import io
import json
import os
import re
import sys
import textwrap
import urllib.error
import urllib.parse
import urllib.request
import uuid
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePosixPath
from typing import Any
import unicodedata
import xml.etree.ElementTree as ET

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"
TEMPLATES_DIR = APP_DIR / "templates"
UPLOAD_DIR = APP_DIR / "uploads"
MAX_FILE_SIZE = 12 * 1024 * 1024
MAX_JSON_BODY_SIZE = 18 * 1024 * 1024
MAX_ROWS = 10000
HOST = "127.0.0.1"
DEFAULT_PORT = 8000

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "").strip() or "gpt-3.5-turbo"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class DatasetSheet:
    name: str
    headers: list[str]
    rows: list[dict[str, str]]
    profile: dict[str, Any]


@dataclass
class WorkbookState:
    filename: str
    stored_path: str
    sheet_names: list[str]
    active_sheet: str
    sheets: dict[str, DatasetSheet]
    warnings: list[str]
    uploaded_at: str


APP_STATE: dict[str, Any] = {
    "workbook": None,
}


XLSX_MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
XLSX_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"


def normalize_text(value: str) -> str:
    value = value.lower().strip().replace("đ", "d")
    value = "".join(
        ch for ch in unicodedata.normalize("NFD", value) if unicodedata.category(ch) != "Mn"
    )
    value = re.sub(r"[^0-9a-zA-Z]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def request_path(raw_path: str) -> str:
    return urllib.parse.unquote(urllib.parse.urlparse(raw_path).path or "/")


def resolve_static_path(raw_path: str) -> Path | None:
    path = request_path(raw_path)
    if not path.startswith("/static/"):
        return None

    relative_path = PurePosixPath(path.removeprefix("/static/"))
    if not relative_path.parts or relative_path.is_absolute() or ".." in relative_path.parts:
        return None

    candidate = (STATIC_DIR / Path(*relative_path.parts)).resolve()
    try:
        candidate.relative_to(STATIC_DIR.resolve())
    except ValueError:
        return None
    return candidate if candidate.is_file() else None


def safe_json_loads(raw: str) -> dict[str, Any] | None:
    raw = raw.strip()
    if not raw:
        return None
    raw = raw.replace("```json", "").replace("```", "").strip()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return None
        try:
            value = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return value if isinstance(value, dict) else None


def to_number(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    text = text.replace("\u00a0", "").replace(" ", "")
    if re.fullmatch(r"-?\d{1,3}(\.\d{3})+(,\d+)?", text):
        text = text.replace(".", "").replace(",", ".")
    elif re.fullmatch(r"-?\d{1,3}(,\d{3})+(\.\d+)?", text):
        text = text.replace(",", "")
    else:
        text = text.replace(",", "")
    if re.fullmatch(r"-?\d+(\.\d+)?", text):
        return float(text)
    return None


def to_date_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    patterns = [
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%d/%m/%Y",
        "%m/%d/%Y",
        "%d-%m-%Y",
        "%Y-%m-%d %H:%M:%S",
        "%d/%m/%Y %H:%M:%S",
    ]
    for pattern in patterns:
        try:
            return datetime.strptime(text, pattern).isoformat()
        except ValueError:
            continue
    return None


def dedupe_headers(headers: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    clean: list[str] = []
    for idx, header in enumerate(headers, start=1):
        name = (header or "").strip() or f"column_{idx}"
        count = seen.get(name, 0)
        if count:
            unique_name = f"{name}_{count + 1}"
        else:
            unique_name = name
        seen[name] = count + 1
        clean.append(unique_name)
    return clean


def parse_csv_bytes(payload: bytes) -> list[tuple[str, list[dict[str, str]]]]:
    text = payload.decode("utf-8-sig", errors="replace")
    sample = text[:4096]
    dialect = csv.excel
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
    except csv.Error:
        pass

    reader = csv.reader(io.StringIO(text), dialect)
    all_rows = [row for row in reader if any(cell.strip() for cell in row)]
    if not all_rows:
        return [("Sheet1", [])]

    headers = dedupe_headers(all_rows[0])
    records: list[dict[str, str]] = []
    warnings: list[str] = []
    for row in all_rows[1:MAX_ROWS + 1]:
        padded = row + [""] * (len(headers) - len(row))
        record = {headers[i]: padded[i].strip() if i < len(padded) else "" for i in range(len(headers))}
        records.append(record)
    if len(all_rows) - 1 > MAX_ROWS:
        warnings.append(f"Dataset lớn hơn {MAX_ROWS:,} dòng nên MVP chỉ nạp {MAX_ROWS:,} dòng đầu.")
    return [("Sheet1", records)] + [("__warnings__", [{"message": warning} for warning in warnings])]


def column_ref_to_index(ref: str) -> int:
    letters = "".join(ch for ch in ref if ch.isalpha()).upper()
    result = 0
    for ch in letters:
        result = result * 26 + (ord(ch) - ord("A") + 1)
    return max(result - 1, 0)


def parse_shared_strings(zf: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in zf.namelist():
        return []
    root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    ns = {"main": XLSX_MAIN_NS}
    values: list[str] = []
    for item in root.findall("main:si", ns):
        text = "".join(node.text or "" for node in item.findall(".//main:t", ns))
        values.append(text)
    return values


def parse_sheet_xml(xml_bytes: bytes, shared_strings: list[str]) -> list[list[str]]:
    ns = {"main": XLSX_MAIN_NS}
    root = ET.fromstring(xml_bytes)
    rows: list[list[str]] = []
    for row in root.findall(".//main:sheetData/main:row", ns):
        cells: dict[int, str] = {}
        max_index = -1
        for cell in row.findall("main:c", ns):
            ref = cell.attrib.get("r", "")
            idx = column_ref_to_index(ref)
            max_index = max(max_index, idx)
            cell_type = cell.attrib.get("t")
            value = ""
            if cell_type == "inlineStr":
                value = "".join(node.text or "" for node in cell.findall(".//main:t", ns))
            else:
                raw_value = cell.findtext("main:v", default="", namespaces=ns)
                if cell_type == "s":
                    try:
                        value = shared_strings[int(raw_value)]
                    except (ValueError, IndexError):
                        value = raw_value
                elif cell_type == "b":
                    value = "TRUE" if raw_value == "1" else "FALSE"
                else:
                    value = raw_value
            cells[idx] = value
        if max_index < 0:
            continue
        rows.append([cells.get(i, "") for i in range(max_index + 1)])
    return rows


def parse_xlsx_bytes(payload: bytes) -> list[tuple[str, list[dict[str, str]]]]:
    workbook_rows: list[tuple[str, list[dict[str, str]]]] = []
    warnings: list[str] = []
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        shared_strings = parse_shared_strings(zf)
        workbook_root = ET.fromstring(zf.read("xl/workbook.xml"))
        rel_root = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        ns = {
            "main": XLSX_MAIN_NS,
            "rel": XLSX_REL_NS,
            "pkg": PKG_REL_NS,
        }
        rel_map: dict[str, str] = {}
        for rel in rel_root.findall("pkg:Relationship", ns):
            rel_map[rel.attrib["Id"]] = rel.attrib["Target"]

        for sheet in workbook_root.findall(".//main:sheets/main:sheet", ns):
            name = sheet.attrib.get("name", "Sheet")
            rel_id = sheet.attrib.get(f"{{{XLSX_REL_NS}}}id")
            if not rel_id or rel_id not in rel_map:
                continue
            target = rel_map[rel_id].lstrip("/")
            sheet_path = target if target.startswith("xl/") else f"xl/{target}"
            if sheet_path not in zf.namelist():
                continue
            matrix = parse_sheet_xml(zf.read(sheet_path), shared_strings)
            if not matrix:
                workbook_rows.append((name, []))
                continue
            headers = dedupe_headers(matrix[0])
            records: list[dict[str, str]] = []
            for row in matrix[1:MAX_ROWS + 1]:
                padded = row + [""] * (len(headers) - len(row))
                records.append({headers[i]: padded[i].strip() if i < len(padded) else "" for i in range(len(headers))})
            if len(matrix) - 1 > MAX_ROWS:
                warnings.append(
                    f"Sheet '{name}' lớn hơn {MAX_ROWS:,} dòng nên MVP chỉ nạp {MAX_ROWS:,} dòng đầu."
                )
            workbook_rows.append((name, records))
    return workbook_rows + [("__warnings__", [{"message": warning} for warning in warnings])]


def load_tabular_bytes(filename: str, payload: bytes) -> tuple[dict[str, DatasetSheet], list[str]]:
    lower_name = filename.lower()
    parsed: list[tuple[str, list[dict[str, str]]]]
    if lower_name.endswith(".csv"):
        parsed = parse_csv_bytes(payload)
    elif lower_name.endswith(".xlsx"):
        parsed = parse_xlsx_bytes(payload)
    else:
        raise ValueError("Chỉ hỗ trợ file .csv hoặc .xlsx trong MVP này.")

    warnings: list[str] = []
    sheets: dict[str, DatasetSheet] = {}
    for sheet_name, rows in parsed:
        if sheet_name == "__warnings__":
            warnings.extend(row["message"] for row in rows)
            continue
        if rows:
            headers = list(rows[0].keys())
        else:
            headers = []
        profile = build_profile(sheet_name, headers, rows)
        sheets[sheet_name] = DatasetSheet(name=sheet_name, headers=headers, rows=rows, profile=profile)
    if not sheets:
        sheets["Sheet1"] = DatasetSheet(name="Sheet1", headers=[], rows=[], profile=build_profile("Sheet1", [], []))
    return sheets, warnings


def build_profile(sheet_name: str, headers: list[str], rows: list[dict[str, str]]) -> dict[str, Any]:
    columns: list[dict[str, Any]] = []
    numeric_columns: list[str] = []
    categorical_columns: list[str] = []
    date_columns: list[str] = []

    for header in headers:
        values = [row.get(header, "").strip() for row in rows]
        non_empty = [value for value in values if value]
        null_count = len(values) - len(non_empty)
        numeric_values = [number for number in (to_number(value) for value in non_empty) if number is not None]
        date_values = [date for date in (to_date_string(value) for value in non_empty) if date is not None]
        unique_values = list(dict.fromkeys(non_empty))[:5]

        inferred_type = "text"
        if non_empty:
            if len(numeric_values) >= max(1, int(len(non_empty) * 0.8)):
                inferred_type = "number"
                numeric_columns.append(header)
            elif len(date_values) >= max(1, int(len(non_empty) * 0.8)):
                inferred_type = "date"
                date_columns.append(header)
            else:
                inferred_type = "text"
                categorical_columns.append(header)

        column_profile: dict[str, Any] = {
            "name": header,
            "type": inferred_type,
            "null_count": null_count,
            "null_ratio": round((null_count / len(values)) * 100, 2) if values else 0,
            "unique_count": len(set(non_empty)),
            "sample_values": unique_values,
        }
        if inferred_type == "number" and numeric_values:
            sorted_vals = sorted(numeric_values)
            n = len(sorted_vals)
            mid = n // 2
            median = sorted_vals[mid] if n % 2 else (sorted_vals[mid - 1] + sorted_vals[mid]) / 2
            avg = sum(numeric_values) / n
            variance = sum((x - avg) ** 2 for x in numeric_values) / n
            stddev = variance ** 0.5
            column_profile["stats"] = {
                "min": round(min(numeric_values), 4),
                "max": round(max(numeric_values), 4),
                "avg": round(avg, 4),
                "median": round(median, 4),
                "stddev": round(stddev, 4),
                "sum": round(sum(numeric_values), 4),
            }
        columns.append(column_profile)

    insights = generate_auto_insights(rows, columns)
    examples = generate_example_questions(columns)
    return {
        "sheet_name": sheet_name,
        "row_count": len(rows),
        "column_count": len(headers),
        "columns": columns,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "date_columns": date_columns,
        "insights": insights,
        "sample_rows": rows[:10],
        "example_questions": examples,
    }


def generate_auto_insights(rows: list[dict[str, str]], columns: list[dict[str, Any]]) -> list[str]:
    insights: list[str] = []
    if not rows or not columns:
        return ["Dataset đang trống hoặc không đọc được nội dung."]

    total_cells = len(rows) * len(columns)
    null_cells = sum(col["null_count"] for col in columns)
    overall_completeness = round((1 - null_cells / total_cells) * 100, 1) if total_cells else 100
    if overall_completeness < 95:
        insights.append(f"Tổng thể dữ liệu đầy đủ {overall_completeness}% — còn {null_cells:,} ô trống trên {total_cells:,} ô.")
    else:
        insights.append(f"Dữ liệu rất sạch: {overall_completeness}% đầy đủ ({len(rows):,} dòng, {len(columns)} cột).")

    high_nulls = [col for col in columns if col["null_ratio"] >= 30]
    if high_nulls:
        names = ", ".join(f"'{col['name']}' ({col['null_ratio']}%)" for col in high_nulls[:3])
        insights.append(f"Cột thiếu dữ liệu đáng kể: {names}. Nên xử lý trước khi phân tích sâu.")

    numeric_columns = [col for col in columns if col["type"] == "number" and col.get("stats")]
    if numeric_columns:
        top_sum = max(numeric_columns, key=lambda item: item["stats"]["sum"])
        insights.append(
            f"Chỉ số tổng lớn nhất: '{top_sum['name']}' = {top_sum['stats']['sum']:,.2f} "
            f"(trung bình {top_sum['stats']['avg']:,.2f}, median {top_sum['stats']['median']:,.2f})."
        )
        if len(numeric_columns) > 1:
            top_spread = max(numeric_columns, key=lambda c: c["stats"]["stddev"])
            insights.append(
                f"Cột biến động nhất: '{top_spread['name']}' với độ lệch chuẩn {top_spread['stats']['stddev']:,.2f}."
            )

    text_columns = [col for col in columns if col["type"] == "text" and 1 < col["unique_count"] <= 50]
    if text_columns and numeric_columns:
        group_col = text_columns[0]["name"]
        measure_col = numeric_columns[0]["name"]
        ranked = rank_group_by(rows, group_col, measure_col, "sum", 3)
        if ranked:
            leader = ranked[0]
            insights.append(
                f"Theo tổng '{measure_col}', nhóm dẫn đầu ở '{group_col}' là '{leader['group']}' ({leader['value']:,.2f})."
            )

    date_columns = [col for col in columns if col["type"] == "date"]
    if date_columns:
        insights.append(f"Phát hiện {len(date_columns)} cột ngày: {', '.join(c['name'] for c in date_columns[:3])}. Hỏi xu hướng theo thời gian để khám phá thêm.")

    if not insights:
        insights.append("Dataset đã nạp thành công. Bắt đầu với: tổng quan, KPI, top nhóm, xu hướng hoặc chất lượng dữ liệu.")
    return insights


def generate_example_questions(columns: list[dict[str, Any]]) -> list[str]:
    numeric_columns = [col["name"] for col in columns if col["type"] == "number"]
    text_columns = [col["name"] for col in columns if col["type"] == "text"]
    date_columns = [col["name"] for col in columns if col["type"] == "date"]
    examples = ["Cho tôi tổng quan dữ liệu này", "Chất lượng dữ liệu như thế nào?"]
    if numeric_columns:
        examples.append(f"Tổng {numeric_columns[0]} là bao nhiêu?")
        examples.append(f"Trung bình {numeric_columns[0]} là bao nhiêu?")
    if numeric_columns and text_columns:
        examples.append(f"Top 5 {text_columns[0]} theo {numeric_columns[0]}")
        examples.append(f"Tổng {numeric_columns[0]} theo từng {text_columns[0]}")
    if date_columns and numeric_columns:
        examples.append(f"Xu hướng {numeric_columns[0]} theo {date_columns[0]}")
    return examples[:6]


def match_column(question: str, candidates: list[str], preferred_type: str | None = None, profile: dict[str, Any] | None = None) -> str | None:
    if not candidates:
        return None
    normalized_question = normalize_text(question)
    best_name = None
    best_score = 0
    for candidate in candidates:
        norm_candidate = normalize_text(candidate)
        if not norm_candidate:
            continue
        tokens = norm_candidate.split()
        score = sum(2 for token in tokens if token in normalized_question)
        if norm_candidate in normalized_question:
            score += 5
        if preferred_type and profile:
            for column in profile["columns"]:
                if column["name"] == candidate and column["type"] == preferred_type:
                    score += 1
                    break
        if score > best_score:
            best_name = candidate
            best_score = score
    return best_name if best_score > 0 else None


def detect_group_by(question: str, profile: dict[str, Any]) -> str | None:
    normalized_question = normalize_text(question)
    match = re.search(r"(?:theo|by)\s+(.+)$", normalized_question)
    candidate_text = match.group(1).strip() if match else normalized_question
    columns = [col["name"] for col in profile["columns"] if col["type"] in {"text", "date"}]
    return match_column(candidate_text, columns, profile=profile)


def detect_filters(question: str, rows: list[dict[str, str]], profile: dict[str, Any]) -> list[dict[str, str]]:
    normalized_question = normalize_text(question)
    filters: list[dict[str, str]] = []
    for column in profile["columns"]:
        if column["type"] != "text" or column["unique_count"] > 15:
            continue
        values = list(dict.fromkeys(row.get(column["name"], "").strip() for row in rows if row.get(column["name"], "").strip()))
        for value in values:
            if len(value) < 2:
                continue
            if normalize_text(value) and normalize_text(value) in normalized_question:
                filters.append({"column": column["name"], "value": value})
                break
    return filters


def heuristic_plan(question: str, sheet: DatasetSheet) -> dict[str, Any]:
    profile = sheet.profile
    normalized_question = normalize_text(question)
    all_columns = [column["name"] for column in profile["columns"]]
    numeric_columns = profile["numeric_columns"]
    date_columns = profile["date_columns"]

    if any(term in normalized_question for term in [
        "tong quan", "overview", "summary", "tom tat", "mo ta", "gioi thieu", "so luoc"
    ]):
        intent = "overview"
    elif any(term in normalized_question for term in [
        "missing", "null", "thieu", "quality", "chat luong", "schema", "kiem tra", "lo hong", "trong"
    ]):
        intent = "data_quality"
    elif any(term in normalized_question for term in ["bao nhieu dong", "so dong", "row count", "tong so dong"]):
        intent = "count_rows"
    elif any(term in normalized_question for term in [
        "xu huong", "trend", "theo thoi gian", "bien dong", "tang giam", "qua cac", "theo nam", "theo thang",
        "theo quy", "theo ngay", "evolution", "over time"
    ]):
        intent = "trend"
    elif any(term in normalized_question for term in [
        "top", "cao nhat", "lon nhat", "nhieu nhat", "dan dau", "xep hang", "hang dau", "best", "worst", "thap nhat nhat"
    ]):
        intent = "top_n"
    else:
        intent = "aggregate"

    explicit_sum = any(term in normalized_question for term in ["tong", "sum", "total", "cong don"])
    explicit_count = any(
        term in normalized_question
        for term in ["so dong", "row count", "how many rows", "dem", "count", "so luong", "bao nhieu dong", "bao nhieu"]
    )

    aggregation = "sum"
    if any(term in normalized_question for term in ["trung binh", "average", "avg", "mean", "tb"]):
        aggregation = "avg"
    elif any(term in normalized_question for term in ["nho nhat", "min", "thap nhat", "thap nhat", "nho nhat"]):
        aggregation = "min"
    elif any(term in normalized_question for term in ["lon nhat", "max", "cao nhat", "dinh"]):
        aggregation = "max"
    elif explicit_count and not explicit_sum:
        aggregation = "count"

    measure = match_column(question, numeric_columns or all_columns, preferred_type="number", profile=profile)
    group_by = detect_group_by(question, profile)

    if intent == "trend":
        if not group_by and date_columns:
            group_by = date_columns[0]
        elif group_by and group_by not in date_columns:
            intent = "groupby"

    if intent in {"top_n", "aggregate"} and group_by:
        intent = "groupby"

    filters = detect_filters(question, sheet.rows, profile)

    limit_match = re.search(r"top\s+(\d+)", normalized_question)
    limit = int(limit_match.group(1)) if limit_match else 5

    if not measure and aggregation == "count":
        intent = "count_rows" if not group_by else "groupby"

    if not measure and numeric_columns:
        measure = numeric_columns[0]

    if intent == "aggregate" and not measure:
        intent = "overview"

    return {
        "source": "heuristic",
        "intent": intent,
        "measure": measure,
        "group_by": group_by,
        "aggregation": aggregation,
        "limit": limit,
        "filters": filters,
    }


def call_openai_plan(question: str, sheet: DatasetSheet) -> dict[str, Any] | None:
    if not OPENAI_API_KEY:
        return None

    schema_summary = {
        "sheet_name": sheet.name,
        "row_count": sheet.profile["row_count"],
        "columns": [
            {
                "name": column["name"],
                "type": column["type"],
                "null_ratio": column["null_ratio"],
                "sample_values": column["sample_values"],
            }
            for column in sheet.profile["columns"]
        ],
        "example_questions": sheet.profile["example_questions"],
    }
    system_prompt = textwrap.dedent(
        """
        You are the planning agent for a local BI MVP.
        Return JSON only. Do not include markdown fences.
        Supported intents: overview, data_quality, count_rows, aggregate, groupby, top_n, trend.
        Supported aggregations: sum, avg, min, max, count.
        Use intent=trend when the user asks about time-series or evolution over a date column; group_by must be a date column.
        Use chart_type: "bar" for groupby/top_n, "line" for trend, "kpi" for single aggregate, "table" for data_quality/overview.
        You must only use columns that exist in the provided schema.
        If the user asks a question that cannot be mapped safely, return intent=overview.
        """
    ).strip()
    user_prompt = json.dumps(
        {
            "question": question,
            "schema": schema_summary,
            "response_contract": {
                "intent": "overview|data_quality|count_rows|aggregate|groupby|top_n|trend",
                "measure": "column name or null",
                "group_by": "column name or null",
                "aggregation": "sum|avg|min|max|count",
                "chart_type": "bar|line|kpi|table",
                "limit": 5,
                "filters": [{"column": "column", "value": "value"}],
                "reason": "short explanation",
            },
        },
        ensure_ascii=False,
    )

    payload = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    request = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
        print(f"[OpenAI API Error]: Lỗi gọi API Planner - {e}", file=sys.stderr)
        return None

    choices = data.get("choices") or []
    if not choices:
        return None
    raw_content = choices[0].get("message", {}).get("content", "")
    parsed = safe_json_loads(raw_content)
    if not parsed:
        return None
    parsed["source"] = "openai"
    parsed.setdefault("filters", [])
    parsed.setdefault("limit", 5)
    parsed.setdefault("aggregation", "sum")
    return parsed


def apply_filters(rows: list[dict[str, str]], filters: list[dict[str, str]]) -> list[dict[str, str]]:
    if not filters:
        return rows
    filtered = rows
    for filter_item in filters:
        column = filter_item.get("column")
        value = str(filter_item.get("value", "")).strip()
        if not column or not value:
            continue
        filtered = [row for row in filtered if normalize_text(row.get(column, "")) == normalize_text(value)]
    return filtered


def aggregate_numeric(values: list[float], aggregation: str) -> float:
    if not values:
        return 0.0
    if aggregation == "avg":
        return sum(values) / len(values)
    if aggregation == "min":
        return min(values)
    if aggregation == "max":
        return max(values)
    return sum(values)


def rank_group_by(rows: list[dict[str, str]], group_by: str, measure: str | None, aggregation: str, limit: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    counts: Counter[str] = Counter()
    for row in rows:
        group = row.get(group_by, "").strip() or "(Trống)"
        counts[group] += 1
        if measure:
            number = to_number(row.get(measure, ""))
            if number is not None:
                grouped[group].append(number)
    results: list[dict[str, Any]] = []
    for group, count in counts.items():
        if measure and aggregation != "count":
            numbers = grouped.get(group, [])
            value = aggregate_numeric(numbers, aggregation)
        else:
            value = float(count)
        results.append({"group": group, "value": round(value, 4), "row_count": count})
    results.sort(key=lambda item: item["value"], reverse=True)
    return results[:limit]


def sort_trend_results(results: list[dict[str, Any]], group_by: str, profile: dict[str, Any]) -> list[dict[str, Any]]:
    is_date = group_by in profile.get("date_columns", [])
    def sort_key(item: dict[str, Any]) -> Any:
        g = item["group"]
        if is_date:
            dt = to_date_string(g)
            return dt or g
        return g
    return sorted(results, key=sort_key)


def execute_plan(question: str, sheet: DatasetSheet, plan: dict[str, Any]) -> dict[str, Any]:
    rows = apply_filters(sheet.rows, plan.get("filters", []))
    if not rows and sheet.rows:
        return {
            "answer": "Không còn dòng dữ liệu nào sau khi áp dụng bộ lọc suy ra từ câu hỏi. Hãy hỏi lại với điều kiện rõ hơn.",
            "plan": plan,
            "result_columns": [],
            "result_rows": [],
            "chart_type": "table",
            "warnings": ["Bộ lọc suy luận không khớp với dữ liệu hiện có."],
        }

    intent = plan.get("intent", "overview")
    aggregation = plan.get("aggregation", "sum")
    measure = plan.get("measure")
    group_by = plan.get("group_by")
    limit = int(plan.get("limit", 5) or 5)

    if intent == "count_rows":
        return {
            "answer": f"Dataset hiện có {len(rows):,} dòng dữ liệu khả dụng để phân tích.",
            "plan": plan,
            "result_columns": ["metric", "value"],
            "result_rows": [{"metric": "row_count", "value": len(rows)}],
            "chart_type": "kpi",
            "warnings": [],
        }

    if intent == "data_quality":
        quality_rows = []
        for column in sheet.profile["columns"]:
            quality_rows.append(
                {
                    "column": column["name"],
                    "type": column["type"],
                    "null_ratio_percent": column["null_ratio"],
                    "unique_count": column["unique_count"],
                    "sample_values": ", ".join(str(v) for v in column.get("sample_values", [])[:3]),
                }
            )
        quality_rows.sort(key=lambda item: item["null_ratio_percent"], reverse=True)
        worst = quality_rows[0] if quality_rows else None
        answer = "Tôi đã tổng hợp chất lượng dữ liệu theo từng cột."
        if worst:
            if worst["null_ratio_percent"] > 0:
                answer += f" Cột cần chú ý nhất là '{worst['column']}' với tỷ lệ thiếu {worst['null_ratio_percent']}%."
            else:
                answer += " Tất cả các cột đều đầy đủ dữ liệu — không có ô trống nào."
        return {
            "answer": answer,
            "plan": plan,
            "result_columns": ["column", "type", "null_ratio_percent", "unique_count", "sample_values"],
            "result_rows": quality_rows,
            "chart_type": "table",
            "warnings": [],
        }

    if intent == "overview":
        profile = sheet.profile
        answer = (
            f"Sheet '{sheet.name}' có {profile['row_count']:,} dòng và {profile['column_count']} cột. "
            f"Có {len(profile['numeric_columns'])} cột số, {len(profile['categorical_columns'])} cột phân loại "
            f"và {len(profile['date_columns'])} cột ngày."
        )
        return {
            "answer": answer,
            "plan": plan,
            "result_columns": ["insight"],
            "result_rows": [{"insight": item} for item in profile["insights"]],
            "chart_type": "table",
            "warnings": [],
        }

    if intent == "trend" and group_by:
        all_results = rank_group_by(rows, group_by, measure, aggregation, len(rows))
        trend_results = sort_trend_results(all_results, group_by, sheet.profile)
        if not trend_results:
            return {
                "answer": f"Không tính được xu hướng theo '{group_by}'.",
                "plan": plan,
                "result_columns": [],
                "result_rows": [],
                "chart_type": "line",
                "warnings": [],
            }
        metric_name = measure or "số dòng"
        action = {"sum": "tổng", "avg": "trung bình", "count": "số lượng"}.get(aggregation, "tổng")
        answer = (
            f"Xu hướng {action} '{metric_name}' theo '{group_by}' qua {len(trend_results)} mốc thời gian. "
            f"Điểm đầu: {trend_results[0]['value']:,.2f} — Điểm cuối: {trend_results[-1]['value']:,.2f}."
        )
        return {
            "answer": answer,
            "plan": plan,
            "result_columns": ["group", "value", "row_count"],
            "result_rows": trend_results,
            "chart_type": "line",
            "warnings": [],
        }

    if group_by:
        ranked = rank_group_by(rows, group_by, measure, aggregation, limit)
        if not ranked:
            return {
                "answer": f"Không tính được kết quả theo '{group_by}'. Có thể cột đo lường '{measure}' không chứa số hợp lệ.",
                "plan": plan,
                "result_columns": [],
                "result_rows": [],
                "chart_type": "bar",
                "warnings": [],
            }
        leader = ranked[0]
        metric_name = measure or "số dòng"
        action = {
            "sum": "tổng",
            "avg": "trung bình",
            "min": "giá trị nhỏ nhất",
            "max": "giá trị lớn nhất",
            "count": "số lượng",
        }.get(aggregation, "tổng")
        answer = (
            f"Xếp hạng theo '{group_by}': nhóm đứng đầu theo {action} '{metric_name}' "
            f"là '{leader['group']}' với {leader['value']:,.2f}."
        )
        return {
            "answer": answer,
            "plan": plan,
            "result_columns": ["group", "value", "row_count"],
            "result_rows": ranked,
            "chart_type": "bar",
            "warnings": [],
        }

    if measure:
        numeric_values = [number for number in (to_number(row.get(measure, "")) for row in rows) if number is not None]
        if not numeric_values and aggregation != "count":
            return {
                "answer": f"Cột '{measure}' không có đủ giá trị số để tính '{aggregation}'.",
                "plan": plan,
                "result_columns": [],
                "result_rows": [],
                "chart_type": "kpi",
                "warnings": [],
            }
        if aggregation == "count":
            value = float(len(rows))
        else:
            value = aggregate_numeric(numeric_values, aggregation)
        action_label = {"sum": "Tổng", "avg": "Trung bình", "min": "Nhỏ nhất", "max": "Lớn nhất", "count": "Số lượng"}.get(aggregation, aggregation)
        answer = f"{action_label} của '{measure}' là {value:,.2f} (tính trên {len(numeric_values):,} giá trị hợp lệ / {len(rows):,} dòng)."
        return {
            "answer": answer,
            "plan": plan,
            "result_columns": ["metric", "value"],
            "result_rows": [{"metric": f"{aggregation}({measure})", "value": round(value, 4)}],
            "chart_type": "kpi",
            "warnings": [],
        }

    return {
        "answer": "Tôi chưa map được câu hỏi vào một phép phân tích cụ thể. Hãy thử hỏi: tổng quan, KPI, top nhóm, xu hướng hoặc chất lượng dữ liệu.",
        "plan": plan,
        "result_columns": [],
        "result_rows": [],
        "chart_type": "table",
        "warnings": [],
    }


def serialize_workbook_state() -> dict[str, Any]:
    workbook: WorkbookState | None = APP_STATE.get("workbook")
    if workbook is None:
        return {
            "has_dataset": False,
            "mode": "openai" if OPENAI_API_KEY else "heuristic",
            "model": OPENAI_MODEL if OPENAI_API_KEY else None,
        }
    sheet = workbook.sheets[workbook.active_sheet]
    return {
        "has_dataset": True,
        "filename": workbook.filename,
        "stored_path": workbook.stored_path,
        "uploaded_at": workbook.uploaded_at,
        "mode": "openai" if OPENAI_API_KEY else "heuristic",
        "model": OPENAI_MODEL if OPENAI_API_KEY else None,
        "warnings": workbook.warnings,
        "sheet_names": workbook.sheet_names,
        "active_sheet": workbook.active_sheet,
        "profile": sheet.profile,
    }


class AgenticBIHandler(BaseHTTPRequestHandler):
    server_version = "AgenticBIMVP/0.1"

    def log_message(self, format: str, *args: Any) -> None:
        return

    def do_GET(self) -> None:
        path = request_path(self.path)
        if path == "/":
            self._serve_file(TEMPLATES_DIR / "index.html", "text/html; charset=utf-8")
            return
        if path.startswith("/static/"):
            file_path = resolve_static_path(self.path)
            if file_path is None:
                self._send_json({"error": "File not found"}, status=HTTPStatus.NOT_FOUND)
                return
            content_type = "text/plain; charset=utf-8"
            if file_path.suffix == ".css":
                content_type = "text/css; charset=utf-8"
            elif file_path.suffix == ".js":
                content_type = "application/javascript; charset=utf-8"
            self._serve_file(file_path, content_type)
            return
        if path == "/api/state":
            self._send_json(serialize_workbook_state())
            return
        self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        path = request_path(self.path)
        if path == "/api/upload":
            self._handle_upload()
            return
        if path == "/api/select-sheet":
            self._handle_select_sheet()
            return
        if path == "/api/question":
            self._handle_question()
            return
        self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)

    def _read_json(self) -> dict[str, Any]:
        raw_length = str(self.headers.get("Content-Length", "0")).strip()
        try:
            content_length = int(raw_length)
        except ValueError as exc:
            raise ValueError("Content-Length không hợp lệ.") from exc
        if content_length < 0:
            raise ValueError("Content-Length không hợp lệ.")
        if content_length > MAX_JSON_BODY_SIZE:
            raise ValueError(f"Request vượt quá giới hạn {MAX_JSON_BODY_SIZE // (1024 * 1024)}MB.")
        raw = self.rfile.read(content_length)
        if content_length and len(raw) != content_length:
            raise ValueError("Request body không đầy đủ.")
        if not raw:
            return {}
        try:
            decoded = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("Request body phải là UTF-8 hợp lệ.") from exc
        try:
            payload = json.loads(decoded)
        except json.JSONDecodeError as exc:
            raise ValueError("JSON không hợp lệ.") from exc
        if not isinstance(payload, dict):
            raise ValueError("JSON body phải là object.")
        return payload

    def _serve_file(self, path: Path, content_type: str) -> None:
        if not path.exists():
            self._send_json({"error": "File not found"}, status=HTTPStatus.NOT_FOUND)
            return
        body = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _handle_upload(self) -> None:
        try:
            payload = self._read_json()
            filename = str(payload.get("filename", "")).strip()
            encoded = str(payload.get("content_base64", ""))
            if not filename or not encoded:
                raise ValueError("Thiếu filename hoặc content_base64.")
            raw_file = base64.b64decode(encoded, validate=True)
            if len(raw_file) > MAX_FILE_SIZE:
                raise ValueError(f"File vượt quá giới hạn {MAX_FILE_SIZE // (1024 * 1024)}MB của MVP.")
            if not filename.lower().endswith((".csv", ".xlsx")):
                raise ValueError("Chỉ hỗ trợ file .csv hoặc .xlsx.")

            stored_name = f"{uuid.uuid4().hex}_{Path(filename).name}"
            stored_path = UPLOAD_DIR / stored_name
            stored_path.write_bytes(raw_file)

            sheets, warnings = load_tabular_bytes(filename, raw_file)
            sheet_names = list(sheets.keys())
            workbook = WorkbookState(
                filename=filename,
                stored_path=str(stored_path),
                sheet_names=sheet_names,
                active_sheet=sheet_names[0],
                sheets=sheets,
                warnings=warnings,
                uploaded_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            )
            APP_STATE["workbook"] = workbook
            self._send_json({"ok": True, "state": serialize_workbook_state()})
        except (ValueError, binascii.Error, zipfile.BadZipFile) as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)

    def _handle_select_sheet(self) -> None:
        workbook: WorkbookState | None = APP_STATE.get("workbook")
        if workbook is None:
            self._send_json({"error": "Chưa có dataset."}, status=HTTPStatus.BAD_REQUEST)
            return
        try:
            payload = self._read_json()
        except ValueError as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return
        sheet_name = str(payload.get("sheet_name", ""))
        if sheet_name not in workbook.sheets:
            self._send_json({"error": "Sheet không tồn tại."}, status=HTTPStatus.BAD_REQUEST)
            return
        workbook.active_sheet = sheet_name
        self._send_json({"ok": True, "state": serialize_workbook_state()})

    def _handle_question(self) -> None:
        workbook: WorkbookState | None = APP_STATE.get("workbook")
        if workbook is None:
            self._send_json({"error": "Hãy tải file Excel/CSV trước."}, status=HTTPStatus.BAD_REQUEST)
            return
        try:
            payload = self._read_json()
        except ValueError as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return
        question = str(payload.get("question", "")).strip()
        if not question:
            self._send_json({"error": "Câu hỏi không được để trống."}, status=HTTPStatus.BAD_REQUEST)
            return

        sheet = workbook.sheets[workbook.active_sheet]
        plan = call_openai_plan(question, sheet) or heuristic_plan(question, sheet)
        response = execute_plan(question, sheet, plan)
        response["mode"] = "openai" if plan.get("source") == "openai" else "heuristic"
        self._send_json(response)


def run_server() -> None:
    port = int(os.getenv("PORT", str(DEFAULT_PORT)))
    address = (HOST, port)
    try:
        httpd = ThreadingHTTPServer(address, AgenticBIHandler)
    except OSError as exc:
        message = str(exc).lower()
        if exc.errno in {48, 98, 10048} or "address already in use" in message or "only one usage" in message:
            print(
                f"Không mở được cổng {port} vì đang bị dùng. Hãy chạy lại với PORT khác, ví dụ:"
            )
            print(f"PORT={port + 1} python3 server.py")
            sys.exit(1)
        raise

    print(f"Agentic AI-BI MVP đang chạy tại http://{HOST}:{port}")
    print(f"OpenAI planner: {'BẬT' if OPENAI_API_KEY else 'TẮT'}")
    if OPENAI_API_KEY:
        print(f"Model: {OPENAI_MODEL}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nDừng server.")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        print("Chạy: python3 server.py")
        print("Đổi cổng: PORT=8001 python3 server.py")
        sys.exit(0)
    run_server()
