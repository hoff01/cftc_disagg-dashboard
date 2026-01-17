"""Port Houston monthly container performance statistics."""

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

PORT_HOUSTON_PAGE_URL = "https://porthouston.com/about/our-port/statistics/"
PORT_HOUSTON_LINK_TEXT = "Detailed Monthly Container Performance Statistics"

DATA_DIR = Path(__file__).resolve().parent / "data" / "port_teu"
PORT_HOUSTON_CACHE_CSV = DATA_DIR / "port_houston_teu.csv"

GEMINI_API_KEY_ENV = "GOOGLE_API_KEY"
ALT_GEMINI_API_KEY_ENV = "GOOGLE_AI_API_KEY"


class PortHoustonError(Exception):
    """Raised when Port Houston extraction fails."""


def _resolve_gemini_api_key(api_key: Optional[str]) -> str:
    key = (api_key or "").strip()
    if not key:
        key = os.getenv(GEMINI_API_KEY_ENV, "").strip()
    if not key:
        key = os.getenv(ALT_GEMINI_API_KEY_ENV, "").strip()
    if not key:
        raise PortHoustonError(
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
    text = text.replace(",", "")
    negative = text.startswith("(") and text.endswith(")")
    if negative:
        text = text[1:-1]
    try:
        number = float(text)
    except ValueError:
        return 0.0
    return -number if negative else number


def fetch_port_houston_pdf_url(
    *,
    page_url: str = PORT_HOUSTON_PAGE_URL,
    link_text: str = PORT_HOUSTON_LINK_TEXT,
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
        raise PortHoustonError("Could not locate the Port Houston PDF link.")

    return requests.compat.urljoin(page_url, target)


def _download_pdf(url: str, *, timeout: float = 60.0) -> bytes:
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()
    return response.content


def _extract_port_houston_rows_gemini(
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
        raise PortHoustonError(
            "google-genai is required for Gemini extraction. Install with `pip install google-genai`."
        ) from exc

    api_key = _resolve_gemini_api_key(api_key)
    client = genai.Client(api_key=api_key)

    sample_file = client.files.upload(file=str(pdf_path))
    start_time = time.time()
    while sample_file.state == types.FileState.PROCESSING:
        if time.time() - start_time > timeout_seconds:
            raise PortHoustonError("Gemini upload timed out.")
        time.sleep(1)
        sample_file = client.files.get(name=sample_file.name)
    if sample_file.state == types.FileState.FAILED:
        raise PortHoustonError("Gemini file processing failed.")

    prompt = """
Extract the Detailed Monthly Container Performance Statistics table from this PDF.

Return ONLY valid JSON with this structure:
{
  "rows": [
    {
      "date": "Jan-01",
      "loaded_imports": 0,
      "loaded_exports": 0,
      "loaded_total": 0,
      "empty_imports": 0,
      "empty_exports": 0,
      "empty_total": 0,
      "loaded_empty_total": 0
    }
  ]
}

Rules:
1) Extract all rows across all pages of the table.
2) Use the exact table values; if a cell is blank or '-', return 0.
3) If a number is in parentheses, return it as negative.
4) Ignore any totals rows, notes, footers, or capacity tables.
5) Preserve the date text as shown (e.g., Jan-01).
"""

    response_schema = {
        "type": "object",
        "properties": {
            "rows": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "date": {"type": "string"},
                        "loaded_imports": {"type": "number"},
                        "loaded_exports": {"type": "number"},
                        "loaded_total": {"type": "number"},
                        "empty_imports": {"type": "number"},
                        "empty_exports": {"type": "number"},
                        "empty_total": {"type": "number"},
                        "loaded_empty_total": {"type": "number"},
                    },
                    "required": [
                        "date",
                        "loaded_imports",
                        "loaded_exports",
                        "loaded_total",
                        "empty_imports",
                        "empty_exports",
                        "empty_total",
                        "loaded_empty_total",
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
        raise PortHoustonError("Gemini response could not be parsed as JSON.")

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
        raise PortHoustonError("Gemini response missing rows list.")
    return [row for row in rows if isinstance(row, dict)]


def fetch_port_houston_teu_table(
    *,
    model_name: str = "gemini-3-flash-preview",
    api_key: Optional[str] = None,
    timeout_seconds: int = 300,
    retry_on_parse_error: int = 1,
    page_url: str = PORT_HOUSTON_PAGE_URL,
) -> pd.DataFrame:
    """Fetch Port Houston TEU table from the latest PDF using Gemini."""
    pdf_url = fetch_port_houston_pdf_url(page_url=page_url)
    pdf_bytes = _download_pdf(pdf_url)

    with tempfile.TemporaryDirectory() as temp_dir:
        pdf_path = Path(temp_dir) / "port_houston_teu.pdf"
        pdf_path.write_bytes(pdf_bytes)
        rows = _extract_port_houston_rows_gemini(
            pdf_path,
            model_name=model_name,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
            retry_on_parse_error=retry_on_parse_error,
        )

    data = []
    for row in rows:
        date = str(row.get("date", "")).strip()
        if not date or re.search(r"total", date, re.IGNORECASE):
            continue
        data.append(
            {
                "Date": date,
                "Loaded Imports": _coerce_number(row.get("loaded_imports")),
                "Loaded Exports": _coerce_number(row.get("loaded_exports")),
                "Loaded Total": _coerce_number(row.get("loaded_total")),
                "Empty Imports": _coerce_number(row.get("empty_imports")),
                "Empty Exports": _coerce_number(row.get("empty_exports")),
                "Empty Total": _coerce_number(row.get("empty_total")),
                "Loaded and Empty Total": _coerce_number(row.get("loaded_empty_total")),
            }
        )

    frame = pd.DataFrame(
        data,
        columns=[
            "Date",
            "Loaded Imports",
            "Loaded Exports",
            "Loaded Total",
            "Empty Imports",
            "Empty Exports",
            "Empty Total",
            "Loaded and Empty Total",
        ],
    )
    if frame.empty:
        raise PortHoustonError("No rows extracted from Port Houston PDF.")

    frame = frame.drop_duplicates().reset_index(drop=True)
    return frame


def load_port_houston_teu_cache(path: Optional[Path] = None) -> pd.DataFrame:
    """Load cached Port Houston TEU data if available."""
    csv_path = Path(path) if path else PORT_HOUSTON_CACHE_CSV
    if not csv_path.exists():
        return pd.DataFrame()
    return pd.read_csv(csv_path)


def update_port_houston_teu_cache(
    *,
    model_name: str = "gemini-3-flash-preview",
    api_key: Optional[str] = None,
    timeout_seconds: int = 300,
    retry_on_parse_error: int = 1,
    output_csv: Path | str = PORT_HOUSTON_CACHE_CSV,
) -> pd.DataFrame:
    """Refresh the Port Houston TEU CSV with the latest PDF."""
    output_path = Path(output_csv)
    frame = fetch_port_houston_teu_table(
        model_name=model_name,
        api_key=api_key,
        timeout_seconds=timeout_seconds,
        retry_on_parse_error=retry_on_parse_error,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)

    return frame


__all__ = [
    "fetch_port_houston_pdf_url",
    "fetch_port_houston_teu_table",
    "load_port_houston_teu_cache",
    "update_port_houston_teu_cache",
]
