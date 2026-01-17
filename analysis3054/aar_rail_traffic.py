"""AAR weekly rail traffic tables (North America, US, Canada, Mexico)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
import json
import os
import re
import tempfile
import time
from typing import Dict, Iterable, List, Optional

import pandas as pd
import requests

AAR_RAILTRAFFIC_URL_TEMPLATE = (
    "https://www.aar.org/wp-content/uploads/{year}/{folder_month:02d}/"
    "{year}-{file_month:02d}-{day:02d}{suffix}"
)
ALT_AAR_RAILTRAFFIC_URL_TEMPLATE = (
    "https://108.181.11.193/wp-content/uploads/{year}/{folder_month:02d}/"
    "{year}-{file_month:02d}-{day:02d}{suffix}"
)
ALT_AAR_RAILTRAFFIC_CPO = "__cpo=aHR0cHM6Ly93d3cuYWFyLm9yZw"
ALT_AAR_RAILTRAFFIC_MONTH = 6
AAR_RAILTRAFFIC_SUFFIXES = ("-railtraffic.pdf", "-railtraffic-AAR.pdf")

DATA_DIR = Path(__file__).resolve().parent / "data" / "rail_traffic"

AAR_TABLE_FILES = {
    "north_american_rail_traffic": DATA_DIR / "rail_traffic_north_american.csv",
    "us_rail_traffic": DATA_DIR / "rail_traffic_us.csv",
    "canadian_rail_traffic": DATA_DIR / "rail_traffic_canada.csv",
    "mexican_rail_traffic": DATA_DIR / "rail_traffic_mexico.csv",
}

TABLE_NAME_MAP = {
    "northamericanrailtraffic": "north_american_rail_traffic",
    "northamericanrailtraffic1": "north_american_rail_traffic",
    "usrailtraffic": "us_rail_traffic",
    "u.s.railtraffic": "us_rail_traffic",
    "u.srailtraffic": "us_rail_traffic",
    "canadianrailtraffic": "canadian_rail_traffic",
    "mexicanrailtraffic": "mexican_rail_traffic",
}

GEMINI_API_KEY_ENV = "GOOGLE_API_KEY"
ALT_GEMINI_API_KEY_ENV = "GOOGLE_AI_API_KEY"
FALLBACK_GEMINI_API_KEY_ENV = "GEMINI_API_KEY"


class AARRailTrafficError(Exception):
    """Raised when AAR rail traffic extraction fails."""


@dataclass
class RailTable:
    key: str
    week_number: int
    week_ended: date
    rows: List[Dict[str, object]]


def _resolve_gemini_api_key(api_key: Optional[str]) -> str:
    key = (api_key or "").strip()
    if not key:
        key = os.getenv(GEMINI_API_KEY_ENV, "").strip()
    if not key:
        key = os.getenv(ALT_GEMINI_API_KEY_ENV, "").strip()
    if not key:
        key = os.getenv(FALLBACK_GEMINI_API_KEY_ENV, "").strip()
    if not key:
        raise AARRailTrafficError(
            "Google AI API key not set. Set it before running Gemini extraction:\n"
            "1) Create a key in Google AI Studio.\n"
            f"2) In your shell: export {GEMINI_API_KEY_ENV}='YOUR_KEY'\n"
            "3) (Optional) Add the export line to ~/.zshrc or ~/.bashrc and restart the shell.\n"
            f"4) Re-run the command. You can also set {ALT_GEMINI_API_KEY_ENV} or "
            f"{FALLBACK_GEMINI_API_KEY_ENV} instead."
        )
    return key


def _apply_cpo(url: str) -> str:
    if "?" in url:
        return f"{url}&{ALT_AAR_RAILTRAFFIC_CPO}"
    return f"{url}?{ALT_AAR_RAILTRAFFIC_CPO}"


def build_aar_railtraffic_url(
    pub_date: date,
    *,
    template: str = AAR_RAILTRAFFIC_URL_TEMPLATE,
    month_override: Optional[int] = None,
    folder_month_override: Optional[int] = None,
    file_month_override: Optional[int] = None,
    suffix: str = "-railtraffic.pdf",
    include_cpo: bool = False,
) -> str:
    month = month_override if month_override is not None else pub_date.month
    folder_month = folder_month_override if folder_month_override is not None else month
    file_month = file_month_override if file_month_override is not None else month
    url = template.format(
        year=pub_date.year,
        month=month,
        folder_month=folder_month,
        file_month=file_month,
        day=pub_date.day,
        suffix=suffix,
    )
    if include_cpo:
        return _apply_cpo(url)
    return url


def build_aar_railtraffic_url_candidates(pub_date: date) -> List[str]:
    candidates: List[str] = []
    templates = (
        (AAR_RAILTRAFFIC_URL_TEMPLATE, False),
        (ALT_AAR_RAILTRAFFIC_URL_TEMPLATE, True),
    )
    for template, include_cpo in templates:
        for suffix in AAR_RAILTRAFFIC_SUFFIXES:
            candidates.append(
                build_aar_railtraffic_url(
                    pub_date,
                    template=template,
                    include_cpo=include_cpo,
                    suffix=suffix,
                )
            )
            if pub_date.month != ALT_AAR_RAILTRAFFIC_MONTH:
                candidates.append(
                    build_aar_railtraffic_url(
                        pub_date,
                        template=template,
                        folder_month_override=ALT_AAR_RAILTRAFFIC_MONTH,
                        file_month_override=pub_date.month,
                        include_cpo=include_cpo,
                        suffix=suffix,
                    )
                )
    seen: set[str] = set()
    unique = []
    for url in candidates:
        if url in seen:
            continue
        unique.append(url)
        seen.add(url)
    return unique


def most_recent_weekday(anchor: date, weekday: int) -> date:
    """Return most recent weekday <= anchor. Monday=0, Sunday=6."""
    delta = (anchor.weekday() - weekday) % 7
    return anchor - timedelta(days=delta)


def next_weekday(anchor: date, weekday: int) -> date:
    """Return the next weekday >= anchor. Monday=0, Sunday=6."""
    delta = (weekday - anchor.weekday()) % 7
    return anchor + timedelta(days=delta)


def generate_weekly_dates(anchor: date, weeks: int) -> List[date]:
    return [anchor - timedelta(days=7 * i) for i in range(weeks)]


def _expected_publication_dates(
    start_year: int, anchor: date, publication_weekday: int
) -> List[date]:
    dates: List[date] = []
    start = date(start_year, 1, 1)
    current = most_recent_weekday(start, publication_weekday)
    if current < start:
        current += timedelta(days=7)
    while current <= anchor:
        dates.append(current)
        current += timedelta(days=7)
    return dates


def _infer_publication_dates(
    frame: Optional[pd.DataFrame], publication_weekday: int
) -> set[date]:
    if frame is None or frame.empty or "week_ended" not in frame.columns:
        return set()
    week_ended = pd.to_datetime(frame["week_ended"], errors="coerce").dt.date
    week_ended = week_ended.dropna()
    return {next_weekday(value, publication_weekday) for value in week_ended}


def _missing_publication_dates(
    expected: Iterable[date],
    existing: Dict[str, pd.DataFrame],
    publication_weekday: int,
) -> List[date]:
    expected_set = set(expected)
    missing: set[date] = set()
    for key in AAR_TABLE_FILES:
        present = _infer_publication_dates(existing.get(key), publication_weekday)
        missing.update(expected_set - present)
    return sorted(missing)


def _download_pdf(url: str, *, timeout: float = 30.0) -> Optional[bytes]:
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers, timeout=timeout)
        if response.status_code == 404:
            return None
        response.raise_for_status()
    except Exception as exc:
        raise AARRailTrafficError(f"Failed to download {url}: {exc}") from exc
    return response.content


def _normalize_table_key(name: str) -> Optional[str]:
    text = re.sub(r"\s+", "", name.strip().lower())
    text = text.replace("\u00a0", "")
    return TABLE_NAME_MAP.get(text)


def _parse_week_ended(text: str) -> date:
    cleaned = re.sub(r"\s+", " ", text.strip())
    for fmt in ("%B %d, %Y", "%b %d, %Y"):
        try:
            return datetime.strptime(cleaned, fmt).date()
        except ValueError:
            continue
    raise AARRailTrafficError(f"Unable to parse week ended date: {text}")


def _coerce_number(value: object) -> float:
    if value is None:
        return 0.0
    text = str(value).strip()
    if not text or text in {"-", "--", "nan", "NaN"}:
        return 0.0
    negative = text.startswith("(") and text.endswith(")")
    if negative:
        text = text[1:-1]
    text = text.replace(",", "")
    try:
        number = float(text)
    except ValueError:
        return 0.0
    return -number if negative else number


def _extract_tables_gemini(
    pdf_path: Path,
    *,
    model_name: str = "gemini-3-flash-preview",
    api_key: Optional[str] = None,
    timeout_seconds: int = 300,
    retry_on_parse_error: int = 1,
) -> List[RailTable]:
    try:
        import google.genai as genai
        from google.genai import types
    except ImportError as exc:  # pragma: no cover - optional path
        raise AARRailTrafficError(
            "google-genai is required for Gemini extraction. Install with `pip install google-genai`."
        ) from exc

    api_key = _resolve_gemini_api_key(api_key)
    client = genai.Client(api_key=api_key)

    sample_file = client.files.upload(file=str(pdf_path))
    start_time = time.time()
    while sample_file.state == types.FileState.PROCESSING:
        if time.time() - start_time > timeout_seconds:
            raise AARRailTrafficError("Gemini upload timed out.")
        time.sleep(1)
        sample_file = client.files.get(name=sample_file.name)
    if sample_file.state == types.FileState.FAILED:
        raise AARRailTrafficError("Gemini file processing failed.")

    prompt = """
Extract rail traffic tables from this PDF.

Return ONLY valid JSON with this structure:
{
  "week_number": "51",
  "week_ended": "December 20, 2025",
  "tables": [
    {
      "table_name": "North American Rail Traffic",
      "rows": [
        {"commodity": "Chemicals", "cars": 45660}
      ]
    }
  ]
}

Rules:
1) Tables include North American Rail Traffic, U.S. Rail Traffic, Canadian Rail Traffic, Mexican Rail Traffic.
2) Use the "Cars" value from the "This Week" column only.
3) Exclude any rows that are totals (Total Carloads, Total Intermodal Units, Total Traffic).
4) Blank or '-' values should be 0.
5) Numbers in parentheses should be negative.
6) Preserve row labels exactly as shown.
"""

    response_schema = {
        "type": "object",
        "properties": {
            "week_number": {"type": "string"},
            "week_ended": {"type": "string"},
            "tables": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "table_name": {"type": "string"},
                        "rows": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "commodity": {"type": "string"},
                                    "cars": {"type": "number"},
                                },
                                "required": ["commodity", "cars"],
                            },
                        },
                    },
                    "required": ["table_name", "rows"],
                },
            },
        },
        "required": ["week_number", "week_ended", "tables"],
    }

    file_part = types.Part.from_uri(
        file_uri=sample_file.uri,
        mime_type=sample_file.mime_type or "application/pdf",
    )

    def request_payload() -> object:
        response = client.models.generate_content(
            model=model_name,
            contents=[file_part, prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_json_schema=response_schema,
            ),
        )
        if response.parsed is not None:
            return response.parsed
        response_text = response.text or ""
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            match = re.search(r"(\{.*\}|\[.*\])", response_text, re.DOTALL)
            if match:
                return json.loads(match.group(0))
        raise AARRailTrafficError("Gemini response could not be parsed as JSON.")

    payload = None
    attempts = max(0, int(retry_on_parse_error))
    for attempt in range(attempts + 1):
        try:
            payload = request_payload()
            break
        except Exception as exc:
            if attempt >= attempts:
                raise
            time.sleep(1)

    if not isinstance(payload, dict):
        raise AARRailTrafficError("Gemini response is not a JSON object.")

    week_number_raw = str(payload.get("week_number", "")).strip()
    week_number = int(re.sub(r"[^0-9]", "", week_number_raw) or 0)
    week_ended_raw = str(payload.get("week_ended", "")).strip()
    week_ended = _parse_week_ended(week_ended_raw)

    tables = payload.get("tables") or []
    if not isinstance(tables, list):
        raise AARRailTrafficError("Gemini response missing tables list.")

    results: List[RailTable] = []
    for table in tables:
        if not isinstance(table, dict):
            continue
        name = str(table.get("table_name", "")).strip()
        key = _normalize_table_key(name)
        if not key:
            continue
        rows = []
        for row in table.get("rows") or []:
            if not isinstance(row, dict):
                continue
            commodity = str(row.get("commodity", "")).strip()
            if not commodity or re.search(r"total", commodity, re.IGNORECASE):
                continue
            rows.append({"commodity": commodity, "cars": _coerce_number(row.get("cars"))})
        if rows:
            results.append(RailTable(key=key, week_number=week_number, week_ended=week_ended, rows=rows))

    if not results:
        raise AARRailTrafficError("Gemini returned no usable rail traffic tables.")

    return results


def _tables_to_frames(tables: Iterable[RailTable]) -> Dict[str, pd.DataFrame]:
    frames: Dict[str, List[Dict[str, object]]] = {key: [] for key in AAR_TABLE_FILES}
    for table in tables:
        for row in table.rows:
            frames[table.key].append(
                {
                    "week_number": table.week_number,
                    "week_ended": table.week_ended.isoformat(),
                    "commodity": row.get("commodity", ""),
                    "cars": row.get("cars", 0.0),
                }
            )
    return {key: pd.DataFrame(rows) for key, rows in frames.items() if rows}


def extract_aar_railtraffic_pdf(
    pub_date: date,
    *,
    model_name: str = "gemini-3-flash-preview",
    api_key: Optional[str] = None,
    timeout_seconds: int = 300,
    retry_on_parse_error: int = 1,
    retry_attempts: int = 2,
    retry_delay: float = 2.0,
) -> Dict[str, pd.DataFrame]:
    pdf_bytes = None
    saw_missing = False
    last_exc: Optional[Exception] = None
    for url in build_aar_railtraffic_url_candidates(pub_date):
        try:
            pdf_bytes = _download_pdf(url)
        except Exception as exc:
            last_exc = exc
            continue
        if pdf_bytes is None:
            saw_missing = True
            continue
        break
    if pdf_bytes is None:
        if last_exc and not saw_missing:
            raise last_exc
        return {}

    with tempfile.TemporaryDirectory() as temp_dir:
        pdf_path = Path(temp_dir) / f"aar_railtraffic_{pub_date.isoformat()}.pdf"
        pdf_path.write_bytes(pdf_bytes)
        attempt = 0
        while True:
            try:
                tables = _extract_tables_gemini(
                    pdf_path,
                    model_name=model_name,
                    api_key=api_key,
                    timeout_seconds=timeout_seconds,
                    retry_on_parse_error=retry_on_parse_error,
                )
                break
            except Exception as exc:
                attempt += 1
                if attempt > max(0, retry_attempts):
                    raise
                time.sleep(retry_delay)

    return _tables_to_frames(tables)


def _merge_tables(
    existing: Optional[pd.DataFrame],
    new: pd.DataFrame,
) -> pd.DataFrame:
    if existing is None or existing.empty:
        combined = new
    else:
        combined = pd.concat([existing, new], ignore_index=True)
    combined = combined.drop_duplicates(subset=["week_number", "week_ended", "commodity"], keep="last")
    combined = combined.sort_values(["week_ended", "commodity"]).reset_index(drop=True)
    return combined


def load_aar_railtraffic_tables() -> Dict[str, pd.DataFrame]:
    result: Dict[str, pd.DataFrame] = {}
    for key, path in AAR_TABLE_FILES.items():
        if path.exists():
            result[key] = pd.read_csv(path)
        else:
            result[key] = pd.DataFrame()
    return result


def update_aar_railtraffic_tables(
    *,
    weeks: int = 4,
    start_year: int = 2022,
    end_date: Optional[date] = None,
    model_name: str = "gemini-3-flash-preview",
    api_key: Optional[str] = None,
    timeout_seconds: int = 300,
    retry_on_parse_error: int = 1,
    retry_attempts: int = 2,
    retry_delay: float = 2.0,
    publication_weekday: int = 2,
    full_refresh: bool = False,
    only_missing: bool = True,
) -> Dict[str, pd.DataFrame]:
    today = date.today()
    anchor = most_recent_weekday(today, publication_weekday)
    if end_date:
        anchor = min(anchor, most_recent_weekday(end_date, publication_weekday))

    existing = load_aar_railtraffic_tables()
    updated = {key: df.copy() for key, df in existing.items()}
    expected_dates = _expected_publication_dates(start_year, anchor, publication_weekday)

    if full_refresh:
        dates = expected_dates
    elif only_missing:
        dates = _missing_publication_dates(expected_dates, existing, publication_weekday)
    else:
        dates = generate_weekly_dates(anchor, weeks)

    for pub_date in dates:
        try:
            frames = extract_aar_railtraffic_pdf(
                pub_date,
                model_name=model_name,
                api_key=api_key,
                timeout_seconds=timeout_seconds,
                retry_on_parse_error=retry_on_parse_error,
                retry_attempts=retry_attempts,
                retry_delay=retry_delay,
            )
        except Exception as exc:
            print(f"AAR rail traffic extraction failed for {pub_date}: {exc}")
            continue
        for key, new_frame in frames.items():
            updated[key] = _merge_tables(updated.get(key), new_frame)

    for key, path in AAR_TABLE_FILES.items():
        frame = updated.get(key)
        if frame is None or frame.empty:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)

    return updated


__all__ = [
    "AAR_RAILTRAFFIC_URL_TEMPLATE",
    "AAR_TABLE_FILES",
    "build_aar_railtraffic_url",
    "update_aar_railtraffic_tables",
    "extract_aar_railtraffic_pdf",
    "load_aar_railtraffic_tables",
]


if __name__ == "__main__":
    import sys

    # Quick way to run the update directly from this file
    # Ensure project root is in path
    root_dir = Path(__file__).resolve().parent.parent
    if str(root_dir) not in sys.path:
        sys.path.insert(0, str(root_dir))

    print("Running AAR rail traffic update from module...")
    
    # Try to find a key
    key = os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_AI_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not key:
        print("Warning: No API key found in environment. Update may fail if not cached.")
    
    try:
        update_aar_railtraffic_tables(api_key=key, only_missing=True)
        print("Update completed successfully.")
    except Exception as e:
        print(f"Update failed: {e}")
