"""Gemini-backed extractor for Texas RRC refinery statements."""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urljoin

import httpx
import pandas as pd
from bs4 import BeautifulSoup

BASE_STATEMENT_URL = (
    "https://www.rrc.texas.gov/oil-and-gas/research-and-statistics/refinery-statements/"
)
DEFAULT_START_YEAR = 2021

MONTH_NAME_TO_NUMBER = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}
NUMBER_TO_MONTH_NAME = {value: key for key, value in MONTH_NAME_TO_NUMBER.items()}

TABLE_COLUMNS = [
    "Name of Material",
    "Code",
    "Storage Beginning of Month",
    "Receipts",
    "Input Runs to Stills and/or Blends",
    "Products Manufactured",
    "Fuel Used",
    "Deliveries",
    "Storage End of Month",
]
NUMERIC_COLUMNS = TABLE_COLUMNS[2:]
CODE_NAME_MAP = {
    1: "Propane",
    2: "Butane",
    3: "Butane-Propane",
    4: "Motor Gasoline",
    5: "Kerosene",
    6: "Home Heating Oil",
    7: "Diesel Fuel",
    8: "Other Middle Distillates",
    9: "Aviation Gasoline",
    10: "Kerosene-Type Jet Fuel",
    11: "Naphtha-Type Jet Fuel",
    12: "Fuel Oil #4 For Utility Use",
    13: "Fuel Oils #5, #6 For Utility Use",
    14: "Fuel Oil #4 For Non-Utility Use",
    15: "Fuel Oils #5, #6 For Non-Utility Use",
    16: "Bunker C",
    17: "Navy Special",
    18: "Other Residual Fuels",
    19: "Petrochemical Feedstocks",
    20: "Lubricants",
    21: "Special Naphthas",
    22: "Solvent Products",
    23: "Miscellaneous",
    24: "Crude Oil",
}
REFINERY_NAME_ALIASES = {
    "buckeye texas processing": "Buckeye Texas Processing (Splitter)",
    "buckeye texas processing buckeye texas processing": "Buckeye Texas Processing (Splitter)",
    "western refining el paso refinery": "Marathon El Paso",
    "western refining o g el paso refinery": "Marathon El Paso",
    "western refining company": "Marathon El Paso",
    "el paso refinery": "Marathon El Paso",
    "baytown texas": "Baytown Refinery",
    "exxon mobil corporation baytown texas": "Baytown Refinery",
    "beaumont texas": "Beaumont Refinery",
    "exxon mobil corporation beaumont texas": "Beaumont Refinery",
    "alon usa l p big spring": "Big Spring Refinery",
    "alon usa l p big spring refinery": "Big Spring Refinery",
    "alon usa l p": "Big Spring Refinery",
    "big spring": "Big Spring Refinery",
    "big spring refinery": "Big Spring Refinery",
    "borger refinery": "Borger Refinery",
    "borger refinery ngl process center": "Borger Refinery",
    "phillips 66 borger refinery": "Borger Refinery",
    "phillips 66 borger refinery ngl process center": "Borger Refinery",
    "avery refinery": "Valero Bill Greehey Refinery East",
    "valero bill greehey refinery e": "Valero Bill Greehey Refinery East",
    "valero bill creehey refinery e": "Valero Bill Greehey Refinery East",
    "valero refining texas lp valero bill greehey refinery e": "Valero Bill Greehey Refinery East",
    "valero bill greehey refinery": "Valero Bill Greehey Refinery East",
    "valero bill greehey refinery w": "Valero Bill Greehey Refinery West",
    "valero bill greehey refi erv w": "Valero Bill Greehey Refinery West",
    "valero refining texas lp valero bill greehey refinery w": "Valero Bill Greehey Refinery West",
    "delek refining ltd tyler refinery": "Delek Refining, LTD - Tyler Refinery",
    "tyler": "Delek Refining, LTD - Tyler Refinery",
    "hartree channelview llc": "Hartree Channelview, LLC.",
    "hartree refining": "Hartree Channelview, LLC.",
    "channelview": "Hartree Channelview, LLC.",
    "houston refinery lp": "Lyondell Houston",
    "houston refining lp": "Lyondell Houston",
    "lyondell houston": "Lyondell Houston",
    "galveston bay refinery": "Marathon Galveston Bay",
    "blanchard refining galveston bay refinery": "Marathon Galveston Bay",
    "blanchard refining": "Marathon Galveston Bay",
    "blanchard refinery": "Marathon Galveston Bay",
    "citgo refining and chemical company lp": "Citgo Corpus Christi",
    "citgo refining and chemical company l p corpus christi refinery": "Citgo Corpus Christi",
    "citgo refining chemical company lp": "Citgo Corpus Christi",
    "flint hills resources": "Flint Hills Corpus Christi",
    "flint hills resources flint hills resources": "Flint Hills Corpus Christi",
    "motiva enterprises llc": "Motiva Port Arthur",
    "motiva enterprises llc port arthur refinery": "Motiva Port Arthur",
    "motiva enterprises llc - port arthur refinery": "Motiva Port Arthur",
    "motiva enterprises, llc - port arthur refinery": "Motiva Port Arthur",
    "valero port arthur": "Valero Port Arthur",
    "valero port arthur refinery": "Valero Port Arthur",
    "kinder morgan crude condensate llc": "Galena Park Splitter",
    "kinder morgan crude condensate llc galena park splitter": "Galena Park Splitter",
    "galena park splitter": "Galena Park Splitter",
    "diamond shamrock refining company l p three rivers": "Valero Three Rivers",
    "three rivers": "Valero Three Rivers",
    "diamond shamrock refining company mckee": "Valero McKee",
    "diamond shamrock refining company l p mckee": "Valero McKee",
    "mckee": "Valero McKee",
    "texas international refinery": "Texas International Refinery",
    "texas international refining llc texas international refinery": "Texas International Refinery",
    "lazarus energy holdings llc": "Nixon Refinery",
    "lazarus energy holdings llc nixon refinery": "Nixon Refinery",
    "nixon refinery": "Nixon Refinery",
    "shell oil company deer park manufacturing complex": "Deer Park",
    "deer park manufacturing complex": "Deer Park",
    "deer park refining limited partnership deer park manufacturing complex": "Deer Park",
    "deer park refining limited partnership deer park": "Deer Park",
    "deer park refining": "Deer Park",
    "the premcor refining group inc premcor port arthur refinery": "Premcor Port Arthur Refinery",
    "premcor port arthur refinery": "Premcor Port Arthur Refinery",
    "phillips 66 co sweeny refinery": "Sweeny Refinery",
    "sweeny refinery": "Sweeny Refinery",
}
FACILITY_ID_CANONICAL = {
    "8-169": "8-286",
    "8-285": "8-286",
    "8-289": "8-286",
    "4-1300": "4-1365",
}
FACILITY_ID_BY_NAME = {
    "big spring refinery": "8-286",
    "galveston bay refinery": "3-52",
}
FACILITY_ID_NAME_OVERRIDES = {
    "3-12": "Baytown Refinery",
    "3-138": "Beaumont Refinery",
    "3-182": "Valero Houston Refinery",
    "3-31": "Valero Texas City",
    "4-3350": "Valero Bill Greehey Refinery West",
    "4-1365": "Valero Bill Greehey Refinery East",
    "3-52": "Marathon Galveston Bay",
    "3-203": "Motiva Port Arthur",
    "2-57": "Valero Three Rivers",
    "10-26": "Valero McKee",
    "3-14": "Deer Park",
    "4-161": "Citgo Corpus Christi",
    "4-3718": "Flint Hills Corpus Christi",
    "3-24": "Valero Port Arthur",
    "10-13": "P66 Borger",
    "3-228": "P66 Sweeny Refinery",
    "3-3695": "GCC Bunkers Refinery Galveston",
    "3-78": "TotalEnergies Port Arthur",
    "8-5": "Marathon El Paso",
    "8-2869": "Delek Big Spring Refinery",
    "6-111": "Delek Tyler Refinery",
    "3-1375": "Hartree Refinery",
    "1-692": "San Antonio Refinery",
    "3-8": "Chevron Pasadena Refinery",
    "3-1358": "Galena Park Splitter",
    "1-763": "Nixon Refinery",
    "4-4269": "Buckeye Texas Processing (Splitter)",
    "3-6": "Lyondell Houston",
}

GEMINI_API_KEY_ENV = "GOOGLE_API_KEY"
ALT_GEMINI_API_KEY_ENV = "GOOGLE_AI_API_KEY"

logger = logging.getLogger("TX_Refineries_Scraper")


class TXRefineryError(Exception):
    """Raised when the refinery pipeline fails."""


class TableParseError(TXRefineryError):
    """Raised when the Gemini response cannot be parsed."""


@dataclass(frozen=True)
class RefineryStatementLink:
    facility_number: str
    operator_name: str
    pdf_url: str
    statement_year: int
    statement_month: int


def _build_statement_url(year: int, month: int) -> str:
    return (
        f"{BASE_STATEMENT_URL}refineries-statements-{year}/"
        f"refinery-statements-{month}-{year}/"
    )


def _iter_statement_months(
    start_year: int,
    end_year: int,
    end_month: int,
    start_month: int = 1,
) -> Iterable[Tuple[int, int]]:
    for year in range(start_year, end_year + 1):
        month_start = start_month if year == start_year else 1
        month_end = end_month if year == end_year else 12
        for month in range(month_start, month_end + 1):
            yield year, month


def _parse_statement_period_from_name(name: str) -> Optional[Tuple[int, int]]:
    basename = Path(name).name.lower()
    month_match = re.search(
        r"(january|february|march|april|may|june|july|august|september|october|november|december)",
        basename,
    )
    year_match = re.search(r"(20\d{2})", basename)
    if not month_match or not year_match:
        return None
    month = MONTH_NAME_TO_NUMBER.get(month_match.group(1))
    year = int(year_match.group(1))
    if month is None:
        return None
    return year, month


def _normalize_facility_number(value: str) -> str:
    text = str(value or "").strip()
    match = re.search(r"(\d{1,2})-(\d{1,4})", text)
    if match:
        return f"{int(match.group(1)):02d}-{int(match.group(2)):04d}"
    return text


def _normalize_facility_id(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if text.lower() in {"nan", "none", "null", "na", "n/a"}:
        return ""
    text = text.replace(" ", "")
    match = re.search(r"(\d{1,2})\s*-\s*(\d{1,4})", text)
    if match:
        return f"{int(match.group(1))}-{int(match.group(2))}"
    digits = re.sub(r"\D", "", text)
    if not digits:
        return ""
    if len(digits) <= 4:
        return str(int(digits))
    left = digits[:-4]
    right = digits[-4:]
    return f"{int(left)}-{int(right)}"


def _resolve_facility_id(
    raw_facility_id: object,
    sequence_id: Optional[str],
    override_facility_id: Optional[str],
) -> str:
    if override_facility_id:
        return _normalize_facility_id(override_facility_id)
    normalized_sequence = _normalize_facility_id(sequence_id) if sequence_id else ""
    normalized_raw = _normalize_facility_id(raw_facility_id)
    if normalized_sequence:
        if normalized_raw and normalized_raw != normalized_sequence:
            logger.warning(
                "Facility ID mismatch (raw=%s sequence=%s); using sequence ID.",
                normalized_raw,
                normalized_sequence,
            )
        return normalized_sequence
    return normalized_raw


def _extract_statement_links(
    html: str,
    base_url: str,
    statement_year: int,
    statement_month: int,
) -> List[RefineryStatementLink]:
    soup = BeautifulSoup(html, "html.parser")
    links: List[RefineryStatementLink] = []
    for row in soup.select("table tr"):
        cells = row.find_all("td")
        if not cells:
            continue
        facility_raw = cells[0].get_text(strip=True) if cells else ""
        facility_number = _normalize_facility_number(facility_raw)
        link_tag = row.find("a", href=True)
        if not link_tag:
            continue
        href = link_tag.get("href", "")
        if not href.lower().endswith(".pdf"):
            continue
        pdf_url = urljoin(base_url, href)
        operator_name = link_tag.get_text(strip=True)
        links.append(
            RefineryStatementLink(
                facility_number=facility_number,
                operator_name=operator_name,
                pdf_url=pdf_url,
                statement_year=statement_year,
                statement_month=statement_month,
            )
        )
    return links


def _fetch_statement_links_for_month(
    client: httpx.Client,
    year: int,
    month: int,
    timeout: float,
) -> List[RefineryStatementLink]:
    url = _build_statement_url(year, month)
    response = client.get(url, timeout=timeout)
    if response.status_code == 404:
        logger.info("No statement page for %s-%s", month, year)
        return []
    response.raise_for_status()
    return _extract_statement_links(response.text, url, year, month)


def _fetch_statement_links_sync(
    start_year: int,
    start_month: int,
    end_year: int,
    end_month: int,
    timeout: float,
) -> List[RefineryStatementLink]:
    headers = {"User-Agent": "analysis3054/tx_refineries"}
    links: List[RefineryStatementLink] = []
    with httpx.Client(headers=headers, follow_redirects=True, timeout=timeout) as client:
        for year, month in _iter_statement_months(start_year, end_year, end_month, start_month):
            links.extend(_fetch_statement_links_for_month(client, year, month, timeout))
    return links


def _is_header_keyword(text: str) -> bool:
    cleaned = re.sub(r"[^a-z]", "", str(text).lower())
    return cleaned in {
        "name",
        "material",
        "code",
        "storage",
        "beginning",
        "receipts",
        "input",
        "runs",
        "products",
        "manufactured",
        "fuel",
        "used",
        "deliveries",
        "end",
        "month",
    }


def _rotate_words(
    words: Sequence[Dict[str, float]],
    width: float,
    height: float,
    angle: int,
) -> List[Dict[str, float]]:
    if angle not in {0, 90, 180, 270}:
        raise ValueError(f"Unsupported rotation angle: {angle}")
    rotated: List[Dict[str, float]] = []
    for word in words:
        x0 = float(word.get("x0", 0))
        x1 = float(word.get("x1", x0))
        top = float(word.get("top", 0))
        bottom = float(word.get("bottom", top))

        if angle == 0:
            new_x0, new_x1 = x0, x1
            new_top, new_bottom = top, bottom
        elif angle == 90:
            new_x0, new_x1 = top, bottom
            new_top, new_bottom = width - x1, width - x0
        elif angle == 180:
            new_x0, new_x1 = width - x1, width - x0
            new_top, new_bottom = height - bottom, height - top
        else:  # 270
            new_x0, new_x1 = height - bottom, height - top
            new_top, new_bottom = x0, x1

        rotated.append(
            {
                **word,
                "x0": new_x0,
                "x1": new_x1,
                "top": new_top,
                "bottom": new_bottom,
            }
        )
    return rotated


def _detect_page_rotation_from_bytes(pdf_bytes: bytes) -> int:
    try:
        import pdfplumber
    except ImportError as exc:  # pragma: no cover - optional path
        raise TableParseError("pdfplumber is required for orientation detection.") from exc

    try:
        with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
            if not pdf.pages:
                return 0
            page = pdf.pages[0]
            words = page.extract_words() or []
            if not words:
                return 0
            width = page.width or 0
            height = page.height or 0
            if not width or not height:
                return 0

            scores: Dict[int, int] = {}
            for angle in (0, 90, 180, 270):
                rotated_words = _rotate_words(words, width, height, angle)
                score = sum(1 for w in rotated_words if _is_header_keyword(w.get("text", "")))
                scores[angle] = score

            best_angle = max(scores, key=scores.get)
            return best_angle if scores[best_angle] >= 3 else 0
    except Exception as exc:
        logger.warning("Rotation detection failed; defaulting to 0: %s", exc)
        return 0


def _rotate_pypdf_page(page, angle: int):
    if not angle:
        return page
    if hasattr(page, "rotate"):
        return page.rotate(angle)
    if hasattr(page, "rotate_clockwise"):
        return page.rotate_clockwise(angle)
    if hasattr(page, "rotateClockwise"):
        return page.rotateClockwise(angle)
    return page


def build_tx_refineries_first_pages_pdf(
    start_year: int = DEFAULT_START_YEAR,
    start_month: int = 1,
    end_year: Optional[int] = None,
    end_month: Optional[int] = None,
    output_path: str | Path = "tx_refineries_first_pages.pdf",
    *,
    rotate_pages: bool = True,
    timeout: float = 60.0,
    progress_every: int = 25,
) -> Path:
    """Download refinery statement PDFs and append their first pages into a single PDF."""
    now = datetime.utcnow()
    resolved_end_year = end_year or now.year
    resolved_end_month = end_month or now.month
    if start_year == resolved_end_year and start_month > resolved_end_month:
        raise TXRefineryError("Start date is after end date.")

    links = _fetch_statement_links_sync(
        start_year,
        start_month,
        resolved_end_year,
        resolved_end_month,
        timeout=timeout,
    )
    if not links:
        raise TXRefineryError("No refinery statement links found for the requested period.")

    unique_links: Dict[Tuple[str, int, int], RefineryStatementLink] = {}
    for link in links:
        key = (link.facility_number, link.statement_year, link.statement_month)
        if key not in unique_links:
            unique_links[key] = link

    ordered_links = list(unique_links.values())
    output_path = Path(output_path)

    try:
        from pypdf import PdfReader, PdfWriter
    except ImportError as exc:  # pragma: no cover - optional path
        raise TableParseError(
            "pypdf is required to build combined PDFs. Install with `pip install pypdf`."
        ) from exc

    writer = PdfWriter()
    headers = {"User-Agent": "analysis3054/tx_refineries"}
    with httpx.Client(timeout=timeout, headers=headers, follow_redirects=True) as client:
        for idx, link in enumerate(ordered_links, start=1):
            response = client.get(link.pdf_url)
            response.raise_for_status()
            pdf_bytes = response.content
            rotation = _detect_page_rotation_from_bytes(pdf_bytes) if rotate_pages else 0
            try:
                reader = PdfReader(BytesIO(pdf_bytes), strict=False)
            except Exception as exc:
                logger.warning("Failed to read PDF %s: %s", link.pdf_url, exc)
                continue
            if not reader.pages:
                continue
            page = reader.pages[0]
            if rotation:
                page = _rotate_pypdf_page(page, rotation)
            writer.add_page(page)
            if progress_every and idx % progress_every == 0:
                logger.info("Added %s/%s statement pages", idx, len(ordered_links))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        writer.write(handle)
    logger.info("Wrote combined PDF: %s (%s pages)", output_path, len(writer.pages))
    return output_path


def build_tx_refineries_full_pages_pdf(
    start_year: int = DEFAULT_START_YEAR,
    start_month: int = 1,
    end_year: Optional[int] = None,
    end_month: Optional[int] = None,
    output_path: str | Path = "tx_refineries_full_pages.pdf",
    *,
    rotate_pages: bool = True,
    failed_pdf_dir: Optional[Path] = None,
    timeout: float = 60.0,
    progress_every: int = 25,
) -> Path:
    """Download refinery statement PDFs and append all pages into a single PDF."""
    now = datetime.utcnow()
    resolved_end_year = end_year or now.year
    resolved_end_month = end_month or now.month
    if start_year == resolved_end_year and start_month > resolved_end_month:
        raise TXRefineryError("Start date is after end date.")

    links = _fetch_statement_links_sync(
        start_year,
        start_month,
        resolved_end_year,
        resolved_end_month,
        timeout=timeout,
    )
    if not links:
        raise TXRefineryError("No refinery statement links found for the requested period.")

    unique_links: Dict[Tuple[str, int, int], RefineryStatementLink] = {}
    for link in links:
        key = (link.facility_number, link.statement_year, link.statement_month)
        if key not in unique_links:
            unique_links[key] = link

    ordered_links = list(unique_links.values())
    output_path = Path(output_path)
    failed_dir = Path(failed_pdf_dir) if failed_pdf_dir is not None else None
    if failed_dir is not None:
        failed_dir.mkdir(parents=True, exist_ok=True)

    try:
        from pypdf import PdfReader, PdfWriter
    except ImportError as exc:  # pragma: no cover - optional path
        raise TableParseError(
            "pypdf is required to build combined PDFs. Install with `pip install pypdf`."
        ) from exc

    writer = PdfWriter()
    headers = {"User-Agent": "analysis3054/tx_refineries"}
    with httpx.Client(timeout=timeout, headers=headers, follow_redirects=True) as client:
        for idx, link in enumerate(ordered_links, start=1):
            response = client.get(link.pdf_url)
            response.raise_for_status()
            pdf_bytes = response.content
            rotation = _detect_page_rotation_from_bytes(pdf_bytes) if rotate_pages else 0
            try:
                reader = PdfReader(BytesIO(pdf_bytes), strict=False)
            except Exception as exc:
                logger.warning("Failed to read PDF %s: %s", link.pdf_url, exc)
                if failed_dir is not None:
                    failed_path = failed_dir / (
                        f"failed_{link.facility_number}_{link.statement_year}_{link.statement_month:02d}.pdf"
                    )
                    failed_path.write_bytes(pdf_bytes)
                continue
            for page in reader.pages:
                if rotation:
                    page = _rotate_pypdf_page(page, rotation)
                writer.add_page(page)
            if progress_every and idx % progress_every == 0:
                logger.info("Merged %s/%s refinery PDFs", idx, len(ordered_links))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        writer.write(handle)
    logger.info("Wrote combined PDF: %s (%s pages)", output_path, len(writer.pages))
    return output_path


def _resolve_gemini_api_key(api_key: Optional[str]) -> str:
    key = (api_key or "").strip()
    if not key:
        key = os.getenv(GEMINI_API_KEY_ENV, "").strip()
    if not key:
        key = os.getenv(ALT_GEMINI_API_KEY_ENV, "").strip()
    if not key:
        key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key:
        raise TXRefineryError(
            "Google AI API key not set. Set it before running Gemini extraction:\n"
            "1) Create a key in Google AI Studio.\n"
            f"2) In your shell: export {GEMINI_API_KEY_ENV}='YOUR_KEY'\n"
            "3) (Optional) Add the export line to ~/.zshrc or ~/.bashrc and restart the shell.\n"
            f"4) Re-run the command. You can also set {ALT_GEMINI_API_KEY_ENV} instead."
        )
    return key


def _parse_report_month(value: str) -> Optional[int]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    text = re.sub(r"[^a-z0-9]", "", text)
    if text.isdigit():
        month = int(text)
        return month if 1 <= month <= 12 else None
    for name, number in MONTH_NAME_TO_NUMBER.items():
        if text.startswith(name[:3]) or text == name:
            return number
    return None


def _parse_report_year(value: str) -> Optional[int]:
    if value is None:
        return None
    text = re.sub(r"[^0-9]", "", str(value))
    if not text:
        return None
    if len(text) == 2:
        return 2000 + int(text)
    if len(text) == 4:
        return int(text)
    return None


def _coerce_number(value: object) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if text in {"", "-", "—"}:
        return 0.0
    negative = False
    if text.startswith("(") and text.endswith(")"):
        negative = True
        text = text[1:-1]
    text = text.replace(",", "").replace(" ", "")
    try:
        number = float(text)
    except ValueError:
        return 0.0
    return -number if negative else number


def _normalize_report_rows(table_df: pd.DataFrame) -> pd.DataFrame:
    if table_df.empty:
        return table_df
    working = table_df.copy()
    working["Code"] = pd.to_numeric(working["Code"], errors="coerce").fillna(0).astype(int)
    working["Name of Material"] = (
        working["Name of Material"].fillna("").astype(str).str.strip().str.lower()
    )
    numeric_cols = [col for col in TABLE_COLUMNS[2:] if col in working.columns]
    working["_abs_total"] = working[numeric_cols].abs().sum(axis=1)
    working = working[working["Code"].between(1, 24)]
    if working.empty:
        base = pd.DataFrame({"Code": list(range(1, 25))})
        base["Name of Material"] = base["Code"].map(CODE_NAME_MAP)
        for col in numeric_cols:
            base[col] = 0.0
        return base[TABLE_COLUMNS]

    working = (
        working.sort_values("_abs_total", ascending=False)
        .drop_duplicates(subset=["Code"], keep="first")
        .drop(columns="_abs_total")
    )

    base = pd.DataFrame({"Code": list(range(1, 25))})
    base["Name of Material"] = base["Code"].map(CODE_NAME_MAP)
    merged = base.merge(working, on="Code", how="left", suffixes=("", "_parsed"))

    if "Name of Material_parsed" in merged.columns:
        merged = merged.drop(columns=["Name of Material_parsed"])
    merged["Name of Material"] = merged["Code"].map(CODE_NAME_MAP)
    merged["Name of Material"] = merged["Name of Material"].astype(str).str.strip().str.lower()

    for col in numeric_cols:
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0.0)

    for col in TABLE_COLUMNS:
        if col not in merged.columns:
            merged[col] = 0.0 if col in numeric_cols else ""

    return merged[TABLE_COLUMNS]


def _normalize_month_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    group_cols = ["facility_id", "refinery_name", "statement_year", "statement_month"]
    frames: List[pd.DataFrame] = []
    for _, group in df.groupby(group_cols, dropna=False):
        normalized = _normalize_report_rows(group[TABLE_COLUMNS])
        sample = group.iloc[0]
        normalized["statement_year"] = sample.get("statement_year", 0)
        normalized["statement_month"] = sample.get("statement_month", 0)
        normalized["statement_month_name"] = sample.get("statement_month_name", "")
        normalized["statement_date"] = sample.get("statement_date", pd.NaT)
        normalized["facility_id"] = sample.get("facility_id", "")
        normalized["refinery_name"] = sample.get("refinery_name", "")
        frames.append(normalized)
    combined = pd.concat(frames, ignore_index=True)
    combined = _drop_missing_facility_duplicates(combined)
    return combined


def _drop_missing_facility_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "facility_id" not in df.columns:
        return df
    working = df.copy()
    working["facility_id"] = working["facility_id"].fillna("").astype(str).str.strip()

    def _canonical_refinery_key(value: object) -> str:
        key = _normalize_refinery_name_key(value)
        alias = REFINERY_NAME_ALIASES.get(key)
        if alias:
            key = _normalize_refinery_name_key(alias)
        return key

    working["refinery_name_key"] = working.get("refinery_name", "").map(_canonical_refinery_key)
    working["Code"] = pd.to_numeric(working.get("Code"), errors="coerce").fillna(0).astype(int)
    key_cols = ["refinery_name_key", "statement_year", "statement_month", "Code"]
    missing = working["facility_id"].str.len() == 0
    if not missing.any():
        return working
    present_keys = working.loc[~missing, key_cols].drop_duplicates()
    if present_keys.empty:
        return working
    present_keys["__present"] = True
    merged = working.merge(present_keys, on=key_cols, how="left")
    drop_mask = missing & merged["__present"].fillna(False)
    cleaned = merged.loc[~drop_mask].drop(columns=["__present", "refinery_name_key"])
    return cleaned


def _normalize_party_name(value: object) -> str:
    text = str(value or "").replace("\n", " ").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _clean_transfer_rows(df: pd.DataFrame, party_column: str) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned[party_column] = cleaned[party_column].map(_normalize_party_name)
    cleaned = cleaned[cleaned[party_column].str.len() > 0]
    cleaned = cleaned[~cleaned[party_column].str.contains(r"\btotal", case=False, regex=True)]
    cleaned = cleaned[~cleaned[party_column].str.contains(r"\bcapacity", case=False, regex=True)]
    cleaned["code"] = pd.to_numeric(cleaned["code"], errors="coerce").fillna(0).astype(int)
    cleaned["barrels"] = pd.to_numeric(cleaned["barrels"], errors="coerce").fillna(0.0)
    cleaned = cleaned[~((cleaned["code"] == 0) & (cleaned["barrels"] == 0))]
    return cleaned


def _apply_latest_refinery_name(
    df: pd.DataFrame,
    facility_col: str = "facility_id",
    name_col: str = "refinery_name",
) -> pd.DataFrame:
    if df.empty or facility_col not in df.columns or name_col not in df.columns:
        return df
    if "statement_date" not in df.columns:
        return df
    working = df.copy()
    working["statement_date"] = pd.to_datetime(working["statement_date"], errors="coerce")
    working[name_col] = working[name_col].fillna("").astype(str).str.strip()
    valid = working[working[facility_col].astype(str).str.len() > 0].copy()
    valid = valid[valid["statement_date"].notna()].sort_values("statement_date")
    if valid.empty:
        return working
    latest = valid.groupby(facility_col)[name_col].agg(lambda x: x.iloc[-1]).to_dict()
    working[name_col] = working[facility_col].map(latest).fillna(working[name_col])
    return working


def _apply_operator_name_map(
    df: pd.DataFrame,
    links: Iterable[RefineryStatementLink],
    facility_col: str = "facility_id",
    name_col: str = "refinery_name",
) -> pd.DataFrame:
    if df.empty or facility_col not in df.columns:
        return df
    name_map = {
        _normalize_facility_id(link.facility_number): str(link.operator_name).strip()
        for link in links
        if _normalize_facility_id(link.facility_number)
    }
    if not name_map:
        return df
    working = df.copy()
    working[facility_col] = working[facility_col].astype(str).map(_normalize_facility_id)
    working[name_col] = working[facility_col].map(name_map).fillna(working.get(name_col, ""))
    return working


def _normalize_refinery_name_key(value: object) -> str:
    text = str(value or "")
    text = re.sub(r"\([^)]*\)", "", text)
    text = re.sub(r"[^A-Za-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def _apply_refinery_name_aliases(df: pd.DataFrame, name_col: str = "refinery_name") -> pd.DataFrame:
    if df.empty or name_col not in df.columns:
        return df
    working = df.copy()
    keys = working[name_col].map(_normalize_refinery_name_key)
    working[name_col] = keys.map(lambda k: REFINERY_NAME_ALIASES.get(k, "")).where(
        keys.str.len() > 0,
        working[name_col],
    )
    # Fill any non-aliased names back in.
    working[name_col] = working[name_col].where(working[name_col].astype(str).str.len() > 0, df[name_col])
    return working


def _apply_facility_id_overrides(
    df: pd.DataFrame,
    facility_col: str = "facility_id",
    name_col: str = "refinery_name",
) -> pd.DataFrame:
    if df.empty or facility_col not in df.columns:
        return df
    working = df.copy()
    working[facility_col] = working[facility_col].astype(str).map(_normalize_facility_id)
    working[facility_col] = working[facility_col].map(
        lambda value: FACILITY_ID_CANONICAL.get(value, value)
    )
    if name_col in working.columns:
        override = working[facility_col].map(FACILITY_ID_NAME_OVERRIDES).fillna("")
        if override.str.len().gt(0).any():
            working[name_col] = override.where(override.str.len() > 0, working[name_col])
        name_key = working[name_col].map(_normalize_refinery_name_key)
        needs_name = working[facility_col].astype(str).str.len() == 0
        mapped = name_key.map(FACILITY_ID_BY_NAME)
        working.loc[needs_name & mapped.notna(), facility_col] = mapped[needs_name & mapped.notna()]
    return working


def TX_refineries_gemini(
    input_path: str | Path,
    *,
    model_name: str = "gemini-3-flash-preview",
    api_key: Optional[str] = None,
    default_year: Optional[int] = None,
    default_month: Optional[int] = None,
    override_statement_year: Optional[int] = None,
    override_statement_month: Optional[int] = None,
    override_facility_id: Optional[str] = None,
    facility_id_sequence: Optional[Sequence[str]] = None,
    normalize_rows: bool = True,
    output_path: Optional[str | Path] = None,
    timeout_seconds: int = 300,
    retry_on_parse_error: int = 1,
) -> pd.DataFrame:
    """Extract refinery tables from a combined PDF using Gemini."""
    try:
        import google.genai as genai
        from google.genai import types
    except ImportError as exc:  # pragma: no cover - optional path
        raise TableParseError(
            "google-genai is required for Gemini extraction. Install with `pip install google-genai`."
        ) from exc

    api_key = _resolve_gemini_api_key(api_key)
    client = genai.Client(api_key=api_key)

    input_path = Path(input_path)
    if not input_path.exists():
        raise TXRefineryError(f"Input file not found: {input_path}")

    sample_file = client.files.upload(file=str(input_path))
    start_time = time.time()
    while sample_file.state == types.FileState.PROCESSING:
        if time.time() - start_time > timeout_seconds:
            raise TXRefineryError("Gemini upload timed out.")
        time.sleep(1)
        sample_file = client.files.get(name=sample_file.name)
    if sample_file.state == types.FileState.FAILED:
        raise TXRefineryError("Gemini file processing failed.")

    prompt = """
You are extracting Monthly Report and Operations Statement data.

Return ONLY valid JSON with this structure:
{
  "reports": [
    {
      "facility_id": "...",
      "refinery_name": "...",
      "report_month": "...",
      "report_year": "...",
      "rows": [
        {
          "name_of_material": "...",
          "code": "...",
          "storage_beginning": 0,
          "receipts": 0,
          "input_runs": 0,
          "products_mfg": 0,
          "fuel_used": 0,
          "deliveries": 0,
          "storage_end": 0
        }
      ]
    }
  ]
}

Rules:
1) Extract facility ID from the Registration No. (e.g., 01-0692). Return it as facility_id.
2) Extract refinery name from Operator/Refinery Name.
3) Extract Month/Year from the header. Convert 2-digit years to 4-digit (e.g., 25 -> 2025).
4) Always return all 24 rows (codes 1-24), even if values are blank. Use 0 for blanks/dashes/NA.
5) Negative numbers in parentheses should be returned as negative.
6) If a far-right column mirrors the Code column, ignore it and use the real Storage End column.
"""

    response_schema = {
        "type": "object",
        "properties": {
            "reports": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "facility_id": {"type": "string"},
                        "refinery_name": {"type": "string"},
                        "report_month": {"type": "string"},
                        "report_year": {"type": "string"},
                        "rows": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name_of_material": {"type": "string"},
                                    "code": {"type": "string"},
                                    "storage_beginning": {"type": "number"},
                                    "receipts": {"type": "number"},
                                    "input_runs": {"type": "number"},
                                    "products_mfg": {"type": "number"},
                                    "fuel_used": {"type": "number"},
                                    "deliveries": {"type": "number"},
                                    "storage_end": {"type": "number"},
                                },
                                "required": [
                                    "name_of_material",
                                    "code",
                                    "storage_beginning",
                                    "receipts",
                                    "input_runs",
                                    "products_mfg",
                                    "fuel_used",
                                    "deliveries",
                                    "storage_end",
                                ],
                            },
                        },
                    },
                    "required": ["refinery_name", "report_month", "report_year", "rows"],
                },
            }
        },
        "required": ["reports"],
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
        raise TableParseError("Gemini response could not be parsed as JSON.")

    payload = None
    attempts = max(0, int(retry_on_parse_error))
    for attempt in range(attempts + 1):
        try:
            payload = request_payload()
            break
        except Exception as exc:
            if attempt >= attempts:
                raise
            logger.warning("Gemini JSON parse failed; retrying (%s/%s): %s", attempt + 1, attempts, exc)

    reports = payload.get("reports") if isinstance(payload, dict) else payload
    if not isinstance(reports, list):
        raise TableParseError("Gemini response missing 'reports' list.")

    frames: List[pd.DataFrame] = []
    for idx, report in enumerate(reports):
        if not isinstance(report, dict):
            continue
        refinery_name = str(report.get("refinery_name", "")).strip()
        sequence_id = None
        if facility_id_sequence is not None and idx < len(facility_id_sequence):
            sequence_id = facility_id_sequence[idx]
        raw_facility_id = report.get("facility_id", "")
        facility_id = _resolve_facility_id(raw_facility_id, sequence_id, override_facility_id)
        report_month_raw = report.get("report_month", "")
        report_year_raw = report.get("report_year", "")
        parsed_month = _parse_report_month(report_month_raw)
        parsed_year = _parse_report_year(report_year_raw)
        report_month = override_statement_month or parsed_month or default_month
        report_year = override_statement_year or parsed_year or default_year
        month_name = NUMBER_TO_MONTH_NAME.get(report_month, "")
        if not month_name and report_month_raw:
            month_name = str(report_month_raw).strip().lower()

        rows = report.get("rows") or []
        table_rows: List[Dict[str, object]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            table_rows.append(
                {
                    "Name of Material": row.get("name_of_material", ""),
                    "Code": row.get("code", ""),
                    "Storage Beginning of Month": _coerce_number(row.get("storage_beginning")),
                    "Receipts": _coerce_number(row.get("receipts")),
                    "Input Runs to Stills and/or Blends": _coerce_number(row.get("input_runs")),
                    "Products Manufactured": _coerce_number(row.get("products_mfg")),
                    "Fuel Used": _coerce_number(row.get("fuel_used")),
                    "Deliveries": _coerce_number(row.get("deliveries")),
                    "Storage End of Month": _coerce_number(row.get("storage_end")),
                }
            )
        if not table_rows:
            continue
        table_df = pd.DataFrame(table_rows)
        if normalize_rows:
            table_df = _normalize_report_rows(table_df)
        else:
            table_df["Code"] = pd.to_numeric(table_df["Code"], errors="coerce").fillna(0.0)
        table_df["statement_year"] = report_year or 0
        table_df["statement_month"] = report_month or 0
        table_df["statement_month_name"] = month_name
        if report_year and report_month:
            table_df["statement_date"] = pd.Timestamp(report_year, report_month, 1)
        else:
            table_df["statement_date"] = pd.NaT
        table_df["facility_id"] = facility_id
        table_df["refinery_name"] = refinery_name
        frames.append(table_df)

    if not frames:
        raise TableParseError("Gemini did not return any statement rows.")

    combined = pd.concat(frames, ignore_index=True)
    if not combined.empty:
        combined = combined.drop_duplicates().reset_index(drop=True)
    combined = combined[TABLE_COLUMNS + [
        "statement_year",
        "statement_month",
        "statement_month_name",
        "statement_date",
        "facility_id",
        "refinery_name",
    ]]
    combined = _apply_latest_refinery_name(combined)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(output_path, index=False)

    return combined


def TX_refineries_receipts_deliveries_gemini(
    input_path: str | Path,
    *,
    model_name: str = "gemini-3-flash-preview",
    api_key: Optional[str] = None,
    default_year: Optional[int] = None,
    default_month: Optional[int] = None,
    override_statement_year: Optional[int] = None,
    override_statement_month: Optional[int] = None,
    override_facility_id: Optional[str] = None,
    facility_id_sequence: Optional[Sequence[str]] = None,
    output_received_path: Optional[str | Path] = None,
    output_delivered_path: Optional[str | Path] = None,
    timeout_seconds: int = 300,
    retry_on_parse_error: int = 1,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Extract Actual Receipts/Deliveries tables from a combined PDF using Gemini."""
    try:
        import google.genai as genai
        from google.genai import types
    except ImportError as exc:  # pragma: no cover - optional path
        raise TableParseError(
            "google-genai is required for Gemini extraction. Install with `pip install google-genai`."
        ) from exc

    api_key = _resolve_gemini_api_key(api_key)
    client = genai.Client(api_key=api_key)

    input_path = Path(input_path)
    if not input_path.exists():
        raise TXRefineryError(f"Input file not found: {input_path}")

    sample_file = client.files.upload(file=str(input_path))
    start_time = time.time()
    while sample_file.state == types.FileState.PROCESSING:
        if time.time() - start_time > timeout_seconds:
            raise TXRefineryError("Gemini upload timed out.")
        time.sleep(1)
        sample_file = client.files.get(name=sample_file.name)
    if sample_file.state == types.FileState.FAILED:
        raise TXRefineryError("Gemini file processing failed.")

    prompt = """
You are extracting the "ACTUAL RECEIPTS" and "ACTUAL DELIVERIES" tables from
Texas Railroad Commission refinery statements.

Return ONLY valid JSON with this structure:
{
  "reports": [
    {
      "refinery_name": "...",
      "report_month": "...",
      "report_year": "...",
      "received_rows": [
        {"received_from": "...", "code": "...", "barrels": 0}
      ],
      "delivered_rows": [
        {"delivered_to": "...", "code": "...", "barrels": 0}
      ]
    }
  ]
}

Rules:
1) Extract facility ID from the Registration No. (e.g., 01-0692). Return it as facility_id.
2) Extract refinery name from Operator/Refinery Name on page 1.
3) Extract Month/Year from the header. Convert 2-digit years to 4-digit (e.g., 25 -> 2025).
4) Only use the "ACTUAL RECEIPTS" (Received From) and "ACTUAL DELIVERIES" (Delivered To) tables.
5) Ignore any table with "CAPACITY" in the header.
6) Each row may contain multiple Code/Barrels pairs; output one row per pair.
7) Ignore totals rows and totals columns.
8) Negative numbers in parentheses should be returned as negative.
9) Blanks/dashes/NA should be returned as 0.
"""

    response_schema = {
        "type": "object",
        "properties": {
            "reports": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "facility_id": {"type": "string"},
                        "refinery_name": {"type": "string"},
                        "report_month": {"type": "string"},
                        "report_year": {"type": "string"},
                        "received_rows": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "received_from": {"type": "string"},
                                    "code": {"type": "string"},
                                    "barrels": {"type": "number"},
                                },
                                "required": ["received_from", "code", "barrels"],
                            },
                        },
                        "delivered_rows": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "delivered_to": {"type": "string"},
                                    "code": {"type": "string"},
                                    "barrels": {"type": "number"},
                                },
                                "required": ["delivered_to", "code", "barrels"],
                            },
                        },
                    },
                    "required": [
                        "refinery_name",
                        "report_month",
                        "report_year",
                        "received_rows",
                        "delivered_rows",
                    ],
                },
            }
        },
        "required": ["reports"],
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
        raise TableParseError("Gemini response could not be parsed as JSON.")

    payload = None
    attempts = max(0, int(retry_on_parse_error))
    for attempt in range(attempts + 1):
        try:
            payload = request_payload()
            break
        except Exception as exc:
            if attempt >= attempts:
                raise
            logger.warning("Gemini JSON parse failed; retrying (%s/%s): %s", attempt + 1, attempts, exc)

    reports = payload.get("reports") if isinstance(payload, dict) else payload
    if not isinstance(reports, list):
        raise TableParseError("Gemini response missing 'reports' list.")

    received_frames: List[pd.DataFrame] = []
    delivered_frames: List[pd.DataFrame] = []

    for idx, report in enumerate(reports):
        if not isinstance(report, dict):
            continue
        refinery_name = str(report.get("refinery_name", "")).strip()
        sequence_id = None
        if facility_id_sequence is not None and idx < len(facility_id_sequence):
            sequence_id = facility_id_sequence[idx]
        raw_facility_id = report.get("facility_id", "")
        facility_id = _resolve_facility_id(raw_facility_id, sequence_id, override_facility_id)
        report_month_raw = report.get("report_month", "")
        report_year_raw = report.get("report_year", "")
        parsed_month = _parse_report_month(report_month_raw)
        parsed_year = _parse_report_year(report_year_raw)
        report_month = override_statement_month or parsed_month or default_month
        report_year = override_statement_year or parsed_year or default_year
        month_name = NUMBER_TO_MONTH_NAME.get(report_month, "")
        if not month_name and report_month_raw:
            month_name = str(report_month_raw).strip().lower()

        received_rows = report.get("received_rows") or []
        received_payload: List[Dict[str, object]] = []
        for row in received_rows:
            if not isinstance(row, dict):
                continue
            received_payload.append(
                {
                    "refinery_name": refinery_name,
                    "received_from": row.get("received_from", ""),
                    "code": row.get("code", ""),
                    "barrels": _coerce_number(row.get("barrels")),
                    "facility_id": facility_id,
                    "statement_year": report_year or 0,
                    "statement_month": report_month or 0,
                    "statement_month_name": month_name,
                    "statement_date": pd.Timestamp(report_year, report_month, 1)
                    if report_year and report_month
                    else pd.NaT,
                }
            )

        delivered_rows = report.get("delivered_rows") or []
        delivered_payload: List[Dict[str, object]] = []
        for row in delivered_rows:
            if not isinstance(row, dict):
                continue
            delivered_payload.append(
                {
                    "refinery_name": refinery_name,
                    "delivered_to": row.get("delivered_to", ""),
                    "code": row.get("code", ""),
                    "barrels": _coerce_number(row.get("barrels")),
                    "facility_id": facility_id,
                    "statement_year": report_year or 0,
                    "statement_month": report_month or 0,
                    "statement_month_name": month_name,
                    "statement_date": pd.Timestamp(report_year, report_month, 1)
                    if report_year and report_month
                    else pd.NaT,
                }
            )

        if received_payload:
            received_df = _clean_transfer_rows(pd.DataFrame(received_payload), "received_from")
            received_frames.append(received_df)
        if delivered_payload:
            delivered_df = _clean_transfer_rows(pd.DataFrame(delivered_payload), "delivered_to")
            delivered_frames.append(delivered_df)

    if not received_frames and not delivered_frames:
        raise TableParseError("Gemini did not return any receipts/deliveries rows.")

    received_df = (
        pd.concat(received_frames, ignore_index=True)
        if received_frames
        else pd.DataFrame(
            columns=[
                "refinery_name",
                "received_from",
                "code",
                "barrels",
                "facility_id",
                "statement_year",
                "statement_month",
                "statement_month_name",
                "statement_date",
            ]
        )
    )
    delivered_df = (
        pd.concat(delivered_frames, ignore_index=True)
        if delivered_frames
        else pd.DataFrame(
            columns=[
                "refinery_name",
                "delivered_to",
                "code",
                "barrels",
                "facility_id",
                "statement_year",
                "statement_month",
                "statement_month_name",
                "statement_date",
            ]
        )
    )

    received_df = _apply_latest_refinery_name(received_df)
    delivered_df = _apply_latest_refinery_name(delivered_df)

    if output_received_path is not None:
        output_received_path = Path(output_received_path)
        output_received_path.parent.mkdir(parents=True, exist_ok=True)
        received_df.to_csv(output_received_path, index=False)

    if output_delivered_path is not None:
        output_delivered_path = Path(output_delivered_path)
        output_delivered_path.parent.mkdir(parents=True, exist_ok=True)
        delivered_df.to_csv(output_delivered_path, index=False)

    return received_df, delivered_df


def TX_refineries_receipts_deliveries(
    start_year: int = DEFAULT_START_YEAR,
    start_month: int = 1,
    end_year: Optional[int] = None,
    end_month: Optional[int] = None,
    *,
    model_name: str = "gemini-3-flash-preview",
    api_key: Optional[str] = None,
    output_received_csv: Optional[str | Path] = None,
    output_delivered_csv: Optional[str | Path] = None,
    existing_received_path: Optional[str | Path] = None,
    existing_delivered_path: Optional[str | Path] = None,
    load_existing: bool = True,
    refresh_months: int = 5,
    timeout: float = 60.0,
    rotate_pages: bool = True,
    retry_attempts: int = 1,
    sleep_seconds: float = 0.5,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Extract Actual Receipts/Deliveries for each month and return two DataFrames."""
    if not (1 <= start_month <= 12):
        raise ValueError("start_month must be between 1 and 12.")
    if end_month is not None and not (1 <= end_month <= 12):
        raise ValueError("end_month must be between 1 and 12.")

    now = datetime.utcnow()
    resolved_end_year = end_year or now.year
    resolved_end_month = end_month or now.month
    if start_year == resolved_end_year and start_month > resolved_end_month:
        return pd.DataFrame(), pd.DataFrame()

    resolved_existing_received = Path(existing_received_path) if existing_received_path else None
    if resolved_existing_received is None and output_received_csv:
        candidate = Path(output_received_csv)
        if candidate.exists():
            resolved_existing_received = candidate

    resolved_existing_delivered = Path(existing_delivered_path) if existing_delivered_path else None
    if resolved_existing_delivered is None and output_delivered_csv:
        candidate = Path(output_delivered_csv)
        if candidate.exists():
            resolved_existing_delivered = candidate

    existing_received = pd.DataFrame()
    existing_delivered = pd.DataFrame()
    if load_existing:
        if resolved_existing_received is not None:
            existing_received = _load_existing_transfer_data(resolved_existing_received)
        if resolved_existing_delivered is not None:
            existing_delivered = _load_existing_transfer_data(resolved_existing_delivered)

    if refresh_months > 0:
        if not existing_received.empty:
            year_series = existing_received["statement_year"].astype(int)
            month_series = existing_received["statement_month"].astype(int)
            valid = (year_series > 0) & (month_series > 0)
            if valid.any():
                month_index = _month_index_series(year_series, month_series)
                max_index = month_index[valid].max()
                cutoff = max_index - (refresh_months - 1)
                existing_received = existing_received[month_index < cutoff]
        if not existing_delivered.empty:
            year_series = existing_delivered["statement_year"].astype(int)
            month_series = existing_delivered["statement_month"].astype(int)
            valid = (year_series > 0) & (month_series > 0)
            if valid.any():
                month_index = _month_index_series(year_series, month_series)
                max_index = month_index[valid].max()
                cutoff = max_index - (refresh_months - 1)
                existing_delivered = existing_delivered[month_index < cutoff]

    completed_months: set[tuple[int, int]] = set()
    if not existing_received.empty and not existing_delivered.empty:
        received_months = {
            (int(y), int(m))
            for y, m in existing_received[["statement_year", "statement_month"]]
            .drop_duplicates()
            .itertuples(index=False)
        }
        delivered_months = {
            (int(y), int(m))
            for y, m in existing_delivered[["statement_year", "statement_month"]]
            .drop_duplicates()
            .itertuples(index=False)
        }
        completed_months = received_months & delivered_months

    received_frames: List[pd.DataFrame] = [existing_received] if not existing_received.empty else []
    delivered_frames: List[pd.DataFrame] = [existing_delivered] if not existing_delivered.empty else []

    for year, month in _iter_statement_months(
        start_year,
        resolved_end_year,
        resolved_end_month,
        start_month=start_month,
    ):
        if (year, month) in completed_months:
            continue
        links = _fetch_statement_links_sync(
            year,
            month,
            year,
            month,
            timeout=timeout,
        )
        ordered_links: List[RefineryStatementLink] = []
        seen_links: set[Tuple[str, int, int]] = set()
        for link in links:
            key = (link.facility_number, link.statement_year, link.statement_month)
            if key in seen_links:
                continue
            seen_links.add(key)
            ordered_links.append(link)

        facility_sequence = [
            _normalize_facility_id(link.facility_number)
            for link in ordered_links
            if _normalize_facility_id(link.facility_number)
        ]
        pdf_path = Path(f"tx_refineries_{year}_{month:02d}_full_pages.pdf")
        failed_dir = Path(f".tx_refineries_failed/{year}_{month:02d}")
        try:
            build_tx_refineries_full_pages_pdf(
                start_year=year,
                start_month=month,
                end_year=year,
                end_month=month,
                output_path=pdf_path,
                rotate_pages=rotate_pages,
                failed_pdf_dir=failed_dir,
                timeout=timeout,
                progress_every=50,
            )
        except TXRefineryError as exc:
            if "No refinery statement links" in str(exc):
                continue
            raise

        received_df = None
        delivered_df = None
        for attempt in range(retry_attempts + 1):
            try:
                received_df, delivered_df = TX_refineries_receipts_deliveries_gemini(
                    pdf_path,
                    model_name=model_name,
                    api_key=api_key,
                    override_statement_year=year,
                    override_statement_month=month,
                    facility_id_sequence=facility_sequence,
                    retry_on_parse_error=1,
                )
                break
            except Exception as exc:
                message = str(exc).lower()
                if "invalid_argument" in message or "invalid argument" in message:
                    logger.warning(
                        "Gemini rejected combined PDF; splitting %s into smaller chunks", pdf_path
                    )
                    chunk_paths = _split_pdf_for_gemini(pdf_path, max_pages=50)
                    if not chunk_paths:
                        raise
                    try:
                        received_parts: List[pd.DataFrame] = []
                        delivered_parts: List[pd.DataFrame] = []
                        for chunk in chunk_paths:
                            rec_part, del_part = TX_refineries_receipts_deliveries_gemini(
                                chunk,
                                model_name=model_name,
                                api_key=api_key,
                                override_statement_year=year,
                                override_statement_month=month,
                                facility_id_sequence=facility_sequence,
                                retry_on_parse_error=1,
                            )
                            if rec_part is not None and not rec_part.empty:
                                received_parts.append(rec_part)
                            if del_part is not None and not del_part.empty:
                                delivered_parts.append(del_part)
                        received_df = (
                            pd.concat(received_parts, ignore_index=True)
                            if received_parts
                            else pd.DataFrame()
                        )
                        delivered_df = (
                            pd.concat(delivered_parts, ignore_index=True)
                            if delivered_parts
                            else pd.DataFrame()
                        )
                        break
                    finally:
                        for chunk in chunk_paths:
                            if chunk.exists():
                                chunk.unlink()
                if attempt >= retry_attempts:
                    raise TableParseError(
                        f"{year}-{month:02d}: Gemini extraction failed."
                    ) from exc
                logger.warning(
                    "%s-%02d Gemini extraction failed; retrying (%s/%s): %s",
                    year,
                    month,
                    attempt + 1,
                    retry_attempts,
                    exc,
                )

        if failed_dir.exists():
            failed_received: List[pd.DataFrame] = []
            failed_delivered: List[pd.DataFrame] = []
            for failed_pdf in sorted(failed_dir.glob("*.pdf")):
                rec_fail, del_fail = TX_refineries_receipts_deliveries_gemini(
                    failed_pdf,
                    model_name=model_name,
                    api_key=api_key,
                    override_statement_year=year,
                    override_statement_month=month,
                    retry_on_parse_error=1,
                )
                if rec_fail is not None and not rec_fail.empty:
                    failed_received.append(rec_fail)
                if del_fail is not None and not del_fail.empty:
                    failed_delivered.append(del_fail)
            if failed_received:
                received_df = (
                    pd.concat([received_df] + failed_received, ignore_index=True)
                    if received_df is not None and not received_df.empty
                    else pd.concat(failed_received, ignore_index=True)
                )
            if failed_delivered:
                delivered_df = (
                    pd.concat([delivered_df] + failed_delivered, ignore_index=True)
                    if delivered_df is not None and not delivered_df.empty
                    else pd.concat(failed_delivered, ignore_index=True)
                )
            for failed_pdf in failed_dir.glob("*.pdf"):
                failed_pdf.unlink()
            failed_dir.rmdir()

        if received_df is not None and not received_df.empty:
            received_df = _apply_operator_name_map(received_df, ordered_links)
            received_frames.append(received_df)
        if delivered_df is not None and not delivered_df.empty:
            delivered_df = _apply_operator_name_map(delivered_df, ordered_links)
            delivered_frames.append(delivered_df)

        if pdf_path.exists():
            pdf_path.unlink()

        if output_received_csv or output_delivered_csv:
            current_received = (
                pd.concat(received_frames, ignore_index=True)
                if received_frames
                else pd.DataFrame()
            )
            current_delivered = (
                pd.concat(delivered_frames, ignore_index=True)
                if delivered_frames
                else pd.DataFrame()
            )
            if not current_received.empty:
                current_received = current_received.drop_duplicates().reset_index(drop=True)
            if not current_delivered.empty:
                current_delivered = current_delivered.drop_duplicates().reset_index(drop=True)
            if output_received_csv is not None:
                output_received_csv = Path(output_received_csv)
                output_received_csv.parent.mkdir(parents=True, exist_ok=True)
                current_received.to_csv(output_received_csv, index=False)
            if output_delivered_csv is not None:
                output_delivered_csv = Path(output_delivered_csv)
                output_delivered_csv.parent.mkdir(parents=True, exist_ok=True)
                current_delivered.to_csv(output_delivered_csv, index=False)

        if sleep_seconds:
            time.sleep(sleep_seconds)

    received_combined = (
        pd.concat(received_frames, ignore_index=True) if received_frames else pd.DataFrame()
    )
    delivered_combined = (
        pd.concat(delivered_frames, ignore_index=True) if delivered_frames else pd.DataFrame()
    )
    if not received_combined.empty:
        received_combined = received_combined.drop_duplicates().reset_index(drop=True)
    if not delivered_combined.empty:
        delivered_combined = delivered_combined.drop_duplicates().reset_index(drop=True)

    received_combined = _apply_latest_refinery_name(received_combined)
    delivered_combined = _apply_latest_refinery_name(delivered_combined)
    received_combined = _apply_refinery_name_aliases(received_combined)
    delivered_combined = _apply_refinery_name_aliases(delivered_combined)
    received_combined = _apply_facility_id_overrides(received_combined)
    delivered_combined = _apply_facility_id_overrides(delivered_combined)

    if output_received_csv is not None:
        output_received_csv = Path(output_received_csv)
        output_received_csv.parent.mkdir(parents=True, exist_ok=True)
        received_combined.to_csv(output_received_csv, index=False)

    if output_delivered_csv is not None:
        output_delivered_csv = Path(output_delivered_csv)
        output_delivered_csv.parent.mkdir(parents=True, exist_ok=True)
        delivered_combined.to_csv(output_delivered_csv, index=False)

    return received_combined, delivered_combined


def _month_index(year: int, month: int) -> int:
    return (year * 12) + (month - 1)


def _month_index_series(year: pd.Series, month: pd.Series) -> pd.Series:
    return (year.astype(int) * 12) + (month.astype(int) - 1)


def _load_existing_data(existing_path: Optional[Path]) -> pd.DataFrame:
    if existing_path is None or not existing_path.exists():
        return pd.DataFrame()
    if existing_path.suffix.lower() == ".pkl":
        df = pd.read_pickle(existing_path)
    else:
        df = pd.read_csv(existing_path)
    if df.empty:
        return df
    for col in TABLE_COLUMNS:
        if col not in df.columns:
            df[col] = 0
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype("float64")
    if "facility_id" not in df.columns:
        df["facility_id"] = ""
    df["facility_id"] = df["facility_id"].astype(str).map(_normalize_facility_id)
    if "statement_year" not in df.columns or "statement_month" not in df.columns:
        if "statement_date" in df.columns:
            df["statement_date"] = pd.to_datetime(df["statement_date"], errors="coerce")
            df["statement_year"] = df["statement_date"].dt.year
            df["statement_month"] = df["statement_date"].dt.month
    return df


def _load_existing_transfer_data(existing_path: Optional[Path]) -> pd.DataFrame:
    if existing_path is None or not existing_path.exists():
        return pd.DataFrame()
    if existing_path.suffix.lower() == ".pkl":
        df = pd.read_pickle(existing_path)
    else:
        df = pd.read_csv(existing_path)
    if df.empty:
        return df
    if "code" not in df.columns:
        df["code"] = 0
    if "barrels" not in df.columns:
        df["barrels"] = 0
    if "facility_id" not in df.columns:
        df["facility_id"] = ""
    df["facility_id"] = df["facility_id"].astype(str).map(_normalize_facility_id)
    df["code"] = pd.to_numeric(df["code"], errors="coerce").fillna(0).astype(int)
    df["barrels"] = pd.to_numeric(df["barrels"], errors="coerce").fillna(0.0)
    if "statement_year" not in df.columns or "statement_month" not in df.columns:
        if "statement_date" in df.columns:
            df["statement_date"] = pd.to_datetime(df["statement_date"], errors="coerce")
            df["statement_year"] = df["statement_date"].dt.year
            df["statement_month"] = df["statement_date"].dt.month
    return df


def _split_pdf_for_gemini(pdf_path: Path, max_pages: int = 50) -> List[Path]:
    try:
        from pypdf import PdfReader, PdfWriter
    except ImportError as exc:  # pragma: no cover - optional path
        raise TableParseError(
            "pypdf is required to split PDFs. Install with `pip install pypdf`."
        ) from exc

    reader = PdfReader(str(pdf_path), strict=False)
    if not reader.pages:
        return []

    chunk_paths: List[Path] = []
    total_pages = len(reader.pages)
    for idx in range(0, total_pages, max_pages):
        writer = PdfWriter()
        for page in reader.pages[idx : idx + max_pages]:
            writer.add_page(page)
        chunk_path = pdf_path.with_name(f"{pdf_path.stem}_part{idx // max_pages + 1}.pdf")
        with chunk_path.open("wb") as handle:
            writer.write(handle)
        chunk_paths.append(chunk_path)
    return chunk_paths


def _validate_month(df: pd.DataFrame, year: int, month: int) -> None:
    if df.empty:
        raise TXRefineryError(f"{year}-{month:02d}: empty dataframe")
    mismatch = df[~((df["statement_year"] == year) & (df["statement_month"] == month))]
    if not mismatch.empty:
        raise TXRefineryError(
            f"{year}-{month:02d}: rows with mismatched period {mismatch[['statement_year','statement_month']].drop_duplicates().to_dict('records')}"
        )
    if "facility_id" in df.columns:
        facility_series = df["facility_id"].astype(str)
        use_facility = facility_series.str.len().gt(0).all()
    else:
        use_facility = False
    group_key = "facility_id" if use_facility else "refinery_name"
    for refinery, group in df.groupby(group_key):
        if len(group) != 24:
            raise TXRefineryError(f"{year}-{month:02d}: {refinery} has {len(group)} rows")
        codes = set(pd.to_numeric(group["Code"], errors="coerce").fillna(0).astype(int).tolist())
        expected = set(range(1, 25))
        if codes != expected:
            missing = expected - codes
            extra = codes - expected
            raise TXRefineryError(
                f"{year}-{month:02d}: {refinery} code mismatch missing={sorted(missing)} extra={sorted(extra)}"
            )


def _missing_facilities(df: pd.DataFrame, expected_ids: set[str]) -> dict[str, int]:
    if df.empty or not expected_ids:
        return {}
    counts = (
        df.groupby("facility_id")
        .size()
        .to_dict()
        if "facility_id" in df.columns
        else {}
    )
    missing = {fid: counts.get(fid, 0) for fid in expected_ids if counts.get(fid, 0) != 24}
    return missing


def _retry_missing_facilities(
    df: pd.DataFrame,
    *,
    year: int,
    month: int,
    link_map: dict[str, RefineryStatementLink],
    model_name: str,
    api_key: Optional[str],
    timeout: float,
) -> pd.DataFrame:
    expected_ids = set(link_map.keys())
    if df.empty or not expected_ids:
        return df
    df = df.copy()
    df["facility_id"] = df.get("facility_id", "").astype(str).map(_normalize_facility_id)
    missing = _missing_facilities(df, expected_ids)
    if not missing:
        return df

    headers = {"User-Agent": "analysis3054/tx_refineries"}
    retry_dir = Path(".tx_refineries_retry")
    retry_dir.mkdir(parents=True, exist_ok=True)
    try:
        from pypdf import PdfReader, PdfWriter
    except ImportError as exc:  # pragma: no cover
        logger.warning("pypdf not available for retry rotation: %s", exc)
        PdfReader = PdfWriter = None  # type: ignore

    def _write_first_page(pdf_bytes: bytes, angle: int, dest: Path) -> bool:
        if PdfReader is None or PdfWriter is None:
            dest.write_bytes(pdf_bytes)
            return True
        try:
            reader = PdfReader(BytesIO(pdf_bytes), strict=False)
        except Exception as exc:  # pragma: no cover - fallback for malformed PDFs
            logger.warning("Retry PDF parse failed; using original bytes: %s", exc)
            dest.write_bytes(pdf_bytes)
            return True
        if not reader.pages:
            return False
        page = reader.pages[0]
        if angle:
            page = _rotate_pypdf_page(page, angle)
        writer = PdfWriter()
        writer.add_page(page)
        with dest.open("wb") as handle:
            writer.write(handle)
        return True

    with httpx.Client(timeout=timeout, headers=headers, follow_redirects=True) as client:
        for facility_id in sorted(missing):
            link = link_map.get(facility_id)
            if not link:
                continue
            response = client.get(link.pdf_url)
            response.raise_for_status()
            pdf_bytes = response.content
            detected = _detect_page_rotation_from_bytes(pdf_bytes)
            angles = [detected] + [angle for angle in (0, 90, 180, 270) if angle != detected]
            df_part = pd.DataFrame()
            for angle in angles:
                pdf_path = retry_dir / f"{year}_{month:02d}_{facility_id}_{angle}.pdf"
                if not _write_first_page(pdf_bytes, angle, pdf_path):
                    continue
                try:
                    df_part = TX_refineries_gemini(
                        pdf_path,
                        model_name=model_name,
                        api_key=api_key,
                        override_statement_year=year,
                        override_statement_month=month,
                        override_facility_id=facility_id,
                        normalize_rows=False,
                        retry_on_parse_error=1,
                    )
                finally:
                    if pdf_path.exists():
                        pdf_path.unlink()
                if df_part is not None and len(df_part) == 24:
                    break
            if df_part is None or df_part.empty:
                logger.warning("%s-%02d facility %s retry returned no rows", year, month, facility_id)
                continue
            df_part["facility_id"] = facility_id
            df = df[df["facility_id"] != facility_id]
            df = pd.concat([df, df_part], ignore_index=True)

    if retry_dir.exists() and not any(retry_dir.iterdir()):
        retry_dir.rmdir()
    return df


def TX_refineries(
    start_year: int = DEFAULT_START_YEAR,
    start_month: int = 1,
    end_year: Optional[int] = None,
    end_month: Optional[int] = None,
    *,
    model_name: str = "gemini-3-flash-preview",
    api_key: Optional[str] = None,
    output_csv: Optional[str | Path] = "tx_refineries_2021_present.csv",
    output_pkl: Optional[str | Path] = "tx_refineries_2021_present.pkl",
    checkpoint_path: Optional[str | Path] = None,
    existing_path: Optional[str | Path] = None,
    load_existing: bool = True,
    refresh_months: int = 3,
    retry_attempts: int = 1,
    sleep_seconds: float = 0.5,
    timeout: float = 60.0,
) -> pd.DataFrame:
    """Run the Gemini extraction month-by-month and return a combined DataFrame."""
    if not (1 <= start_month <= 12):
        raise ValueError("start_month must be between 1 and 12.")
    if end_month is not None and not (1 <= end_month <= 12):
        raise ValueError("end_month must be between 1 and 12.")

    now = datetime.utcnow()
    resolved_end_year = end_year or now.year
    resolved_end_month = end_month or now.month
    if start_year == resolved_end_year and start_month > resolved_end_month:
        return pd.DataFrame()

    resolved_existing = Path(existing_path) if existing_path else None
    if resolved_existing is None and output_csv:
        candidate = Path(output_csv)
        if candidate.exists():
            resolved_existing = candidate

    existing_df = pd.DataFrame()
    if load_existing and resolved_existing is not None:
        existing_df = _load_existing_data(resolved_existing)
        if refresh_months > 0 and not existing_df.empty:
            year_series = existing_df["statement_year"].astype(int)
            month_series = existing_df["statement_month"].astype(int)
            valid = (year_series > 0) & (month_series > 0)
            if valid.any():
                month_index = _month_index(year_series, month_series)
                max_index = month_index[valid].max()
                cutoff = max_index - (refresh_months - 1)
                existing_df = existing_df[month_index < cutoff]

    completed_months = set()
    if not existing_df.empty:
        completed_months = {
            (int(y), int(m))
            for y, m in existing_df[["statement_year", "statement_month"]]
            .drop_duplicates()
            .itertuples(index=False)
        }

    frames = [existing_df] if not existing_df.empty else []

    for year, month in _iter_statement_months(
        start_year,
        resolved_end_year,
        resolved_end_month,
        start_month=start_month,
    ):
        if (year, month) in completed_months:
            continue

        links = _fetch_statement_links_sync(
            year,
            month,
            year,
            month,
            timeout=timeout,
        )
        ordered_links: List[RefineryStatementLink] = []
        seen_links: set[Tuple[str, int, int]] = set()
        for link in links:
            key = (link.facility_number, link.statement_year, link.statement_month)
            if key in seen_links:
                continue
            seen_links.add(key)
            ordered_links.append(link)

        facility_sequence = [
            _normalize_facility_id(link.facility_number)
            for link in ordered_links
            if _normalize_facility_id(link.facility_number)
        ]
        link_map = {
            _normalize_facility_id(link.facility_number): link
            for link in ordered_links
            if _normalize_facility_id(link.facility_number)
        }

        pdf_path = Path(f"tx_refineries_{year}_{month:02d}_first_pages.pdf")
        try:
            build_tx_refineries_first_pages_pdf(
                start_year=year,
                start_month=month,
                end_year=year,
                end_month=month,
                output_path=pdf_path,
                rotate_pages=True,
                timeout=timeout,
                progress_every=50,
            )
        except TXRefineryError as exc:
            if "No refinery statement links" in str(exc):
                continue
            raise

        df_month = None
        for attempt in range(retry_attempts + 1):
            df_month = TX_refineries_gemini(
                pdf_path,
                model_name=model_name,
                api_key=api_key,
                override_statement_year=year,
                override_statement_month=month,
                facility_id_sequence=facility_sequence,
                normalize_rows=False,
                retry_on_parse_error=1,
            )
            try:
                df_month = _retry_missing_facilities(
                    df_month,
                    year=year,
                    month=month,
                    link_map=link_map,
                    model_name=model_name,
                    api_key=api_key,
                    timeout=timeout,
                )
                remaining = _missing_facilities(df_month, set(link_map.keys()))
                if remaining:
                    logger.warning(
                        "%s-%02d: padding missing rows for facility IDs %s",
                        year,
                        month,
                        sorted(remaining.keys()),
                    )
                df_month = _apply_operator_name_map(df_month, ordered_links)
                df_month = _apply_refinery_name_aliases(df_month)
                df_month = _apply_facility_id_overrides(df_month)
                df_month = _normalize_month_df(df_month)
                _validate_month(df_month, year, month)
                break
            except TXRefineryError:
                if attempt >= retry_attempts:
                    raise

        if df_month is not None:
            frames.append(df_month)

        if pdf_path.exists():
            pdf_path.unlink()

        if checkpoint_path:
            checkpoint_path = Path(checkpoint_path)
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            pd.concat(frames, ignore_index=True).to_csv(checkpoint_path, index=False)

        if sleep_seconds:
            time.sleep(sleep_seconds)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)

    if output_csv:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(output_csv, index=False)
    if output_pkl:
        output_pkl = Path(output_pkl)
        output_pkl.parent.mkdir(parents=True, exist_ok=True)
        combined.to_pickle(output_pkl)

    return combined


__all__ = [
    "TX_refineries",
    "TX_refineries_gemini",
    "TX_refineries_receipts_deliveries",
    "TX_refineries_receipts_deliveries_gemini",
    "TXRefineryError",
    "TableParseError",
    "build_tx_refineries_first_pages_pdf",
    "build_tx_refineries_full_pages_pdf",
]
