"""Port of Savannah monthly TEU throughput (GAPorts)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import json
import os
import re
import tempfile
import time
from typing import Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup

PORT_SAVANNAH_PAGE_URL = "https://gaports.com/sales/by-the-numbers/"
PORT_SAVANNAH_LINK_TEXT = "Monthly TEU Throughput"

DATA_DIR = Path(__file__).resolve().parent / "data" / "port_teu"
PORT_SAVANNAH_CACHE_CSV = DATA_DIR / "port_savannah_teu.csv"

MONTHS = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]
MONTH_INDEX = {month: idx for idx, month in enumerate(MONTHS, start=1)}

GEMINI_API_KEY_ENV = "GOOGLE_API_KEY"
ALT_GEMINI_API_KEY_ENV = "GOOGLE_AI_API_KEY"


class PortSavannahError(Exception):
    """Raised when Port of Savannah extraction fails."""


def _resolve_gemini_api_key(api_key: Optional[str]) -> str:
    key = (api_key or "").strip()
    if not key:
        key = os.getenv(GEMINI_API_KEY_ENV, "").strip()
    if not key:
        key = os.getenv(ALT_GEMINI_API_KEY_ENV, "").strip()
    if not key:
        raise PortSavannahError(
            "Google AI API key not set. Set it before running Gemini extraction:\n"
            "1) Create a key in Google AI Studio.\n"
            f"2) In your shell: export {GEMINI_API_KEY_ENV}='YOUR_KEY'\n"
            "3) (Optional) Add the export line to ~/.zshrc or ~/.bashrc and restart the shell.\n"
            f"4) Re-run the command. You can also set {ALT_GEMINI_API_KEY_ENV} instead."
        )
    return key


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


def fetch_port_savannah_pdf_url(
    *,
    page_url: str = PORT_SAVANNAH_PAGE_URL,
    link_text: str = PORT_SAVANNAH_LINK_TEXT,
    timeout: float = 30.0,
) -> str:
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(page_url, headers=headers, timeout=timeout)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    target = None
    for anchor in soup.find_all("a"):
        text = " ".join(anchor.get_text(strip=True).split())
        if link_text.lower() in text.lower():
            target = anchor.get("href")
            break

    if not target:
        raise PortSavannahError("Could not locate the Monthly TEU Throughput PDF link.")

    return requests.compat.urljoin(page_url, target)


def _download_pdf(url: str, *, timeout: float = 60.0) -> bytes:
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()
    return response.content


def _extract_port_savannah_rows_gemini(
    pdf_path: Path,
    *,
    model_name: str = "gemini-3-flash-preview",
    api_key: Optional[str] = None,
    timeout_seconds: int = 300,
    retry_on_parse_error: int = 1,
) -> list[dict]:
    try:
        import google.genai as genai
        from google.genai import types
    except ImportError as exc:  # pragma: no cover - optional path
        raise PortSavannahError(
            "google-genai is required for Gemini extraction. Install with `pip install google-genai`."
        ) from exc

    api_key = _resolve_gemini_api_key(api_key)
    client = genai.Client(api_key=api_key)

    sample_file = client.files.upload(file=str(pdf_path))
    start_time = time.time()
    while sample_file.state == types.FileState.PROCESSING:
        if time.time() - start_time > timeout_seconds:
            raise PortSavannahError("Gemini upload timed out.")
        time.sleep(1)
        sample_file = client.files.get(name=sample_file.name)
    if sample_file.state == types.FileState.FAILED:
        raise PortSavannahError("Gemini file processing failed.")

    prompt = """
Extract the Port of Savannah Monthly TEU Throughput table.

Return ONLY valid JSON with this structure:
{
  "rows": [
    {
      "year": 2024,
      "month": "Jan",
      "import_full": 219079,
      "import_empty": 1042,
      "export_full": 104685,
      "export_empty": 103229
    }
  ]
}

Rules:
1) The table is grouped by year. Extract all years >= 2022.
2) Only capture the following rows for each year: Import Full, Import Empty, Export Full, Export Empty.
3) Ignore Import Total, Export Total, Total Full, Total Empty, Total All, and any percent-change rows.
4) If a month has blank cells, return 0 for that cell.
5) Preserve month abbreviations as shown (Jan, Feb, ...).
"""

    response_schema = {
        "type": "object",
        "properties": {
            "rows": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "year": {"type": "integer"},
                        "month": {"type": "string"},
                        "import_full": {"type": "number"},
                        "import_empty": {"type": "number"},
                        "export_full": {"type": "number"},
                        "export_empty": {"type": "number"},
                    },
                    "required": [
                        "year",
                        "month",
                        "import_full",
                        "import_empty",
                        "export_full",
                        "export_empty",
                    ],
                },
            }
        },
        "required": ["rows"],
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
        raise PortSavannahError("Gemini response could not be parsed as JSON.")

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

    rows = payload.get("rows") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise PortSavannahError("Gemini response missing rows list.")
    return [row for row in rows if isinstance(row, dict)]


def fetch_port_savannah_teu_table(
    *,
    model_name: str = "gemini-3-flash-preview",
    api_key: Optional[str] = None,
    timeout_seconds: int = 300,
    retry_on_parse_error: int = 1,
    page_url: str = PORT_SAVANNAH_PAGE_URL,
) -> pd.DataFrame:
    """Fetch Port of Savannah monthly TEU table from the latest PDF."""
    pdf_url = fetch_port_savannah_pdf_url(page_url=page_url)
    pdf_bytes = _download_pdf(pdf_url)

    with tempfile.TemporaryDirectory() as temp_dir:
        pdf_path = Path(temp_dir) / "port_savannah_teu.pdf"
        pdf_path.write_bytes(pdf_bytes)
        rows = _extract_port_savannah_rows_gemini(
            pdf_path,
            model_name=model_name,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
            retry_on_parse_error=retry_on_parse_error,
        )

    data = []
    for row in rows:
        year = int(row.get("year", 0) or 0)
        if year < 2022:
            continue
        month = str(row.get("month", "")).strip().title()
        month = month[:3]
        if month not in MONTH_INDEX:
            continue
        import_full = _coerce_number(row.get("import_full"))
        import_empty = _coerce_number(row.get("import_empty"))
        export_full = _coerce_number(row.get("export_full"))
        export_empty = _coerce_number(row.get("export_empty"))
        if import_full == 0 and import_empty == 0 and export_full == 0 and export_empty == 0:
            continue
        data.append(
            {
                "Month": month,
                "Year": year,
                "Import Full": import_full,
                "Import Empty": import_empty,
                "Export Full": export_full,
                "Export Empty": export_empty,
            }
        )

    frame = pd.DataFrame(
        data,
        columns=[
            "Month",
            "Year",
            "Import Full",
            "Import Empty",
            "Export Full",
            "Export Empty",
        ],
    )
    if frame.empty:
        raise PortSavannahError("No rows extracted from Port of Savannah PDF.")

    frame["Import Total"] = frame["Import Full"] + frame["Import Empty"]
    frame["Export Total"] = frame["Export Full"] + frame["Export Empty"]
    frame["Total"] = frame["Import Total"] + frame["Export Total"]

    frame["MonthIndex"] = frame["Month"].map(MONTH_INDEX)
    frame = frame.sort_values(["Year", "MonthIndex"]).drop(columns=["MonthIndex"]).reset_index(drop=True)

    return frame


def load_port_savannah_teu_cache(path: Optional[Path] = None) -> pd.DataFrame:
    """Load cached Port of Savannah TEU data if available."""
    csv_path = Path(path) if path else PORT_SAVANNAH_CACHE_CSV
    if not csv_path.exists():
        return pd.DataFrame()
    return pd.read_csv(csv_path)


def update_port_savannah_teu_cache(
    *,
    model_name: str = "gemini-3-flash-preview",
    api_key: Optional[str] = None,
    timeout_seconds: int = 300,
    retry_on_parse_error: int = 1,
    output_csv: Path | str = PORT_SAVANNAH_CACHE_CSV,
) -> pd.DataFrame:
    """Refresh the Port of Savannah TEU CSV with the latest PDF."""
    output_path = Path(output_csv)
    frame = fetch_port_savannah_teu_table(
        model_name=model_name,
        api_key=api_key,
        timeout_seconds=timeout_seconds,
        retry_on_parse_error=retry_on_parse_error,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)

    return frame


__all__ = [
    "fetch_port_savannah_pdf_url",
    "fetch_port_savannah_teu_table",
    "load_port_savannah_teu_cache",
    "update_port_savannah_teu_cache",
]
