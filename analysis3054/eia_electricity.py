"""EIA annual electricity data extracts (EIA-860 + EIA-923)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import io
import re
import zipfile
from typing import Iterable, Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup

EIA860_PAGE_URL = "https://www.eia.gov/electricity/data/eia860/"
EIA923_PAGE_URL = "https://www.eia.gov/electricity/data/eia923/"

DATA_DIR = Path(__file__).resolve().parent / "data"


class EIAElectricityError(Exception):
    """Raised when EIA electricity data extraction fails."""


@dataclass(frozen=True)
class EIA860Files:
    generator: str
    multifuel: str


@dataclass(frozen=True)
class EIA923Files:
    workbook: str


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _abs_url(page_url: str, href: str) -> str:
    return requests.compat.urljoin(page_url, href)


def _download_bytes(url: str, *, timeout: float = 60.0) -> bytes:
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, timeout=timeout, headers=headers)
    response.raise_for_status()
    return response.content


def _is_zip_bytes(content: bytes) -> bool:
    return len(content) >= 4 and content[:2] == b"PK"


def _probe_url(url: str, *, timeout: float = 15.0) -> bool:
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, timeout=timeout, headers=headers, stream=True)
        if response.status_code != 200:
            response.close()
            return False
        first_bytes = response.raw.read(4)
        response.close()
        return first_bytes[:2] == b"PK"
    except Exception:
        return False


def _probe_latest_year(
    *,
    page_url: str,
    filename_template: str,
    archive_template: Optional[str] = None,
    start_year: Optional[int] = None,
    end_year: int = 2000,
) -> int:
    current_year = datetime.now().year
    start = start_year or current_year
    for year in range(start, end_year - 1, -1):
        for template in filter(None, [filename_template, archive_template]):
            url = _abs_url(page_url, template.format(year=year))
            if _probe_url(url):
                return year
    raise EIAElectricityError("Unable to locate an available EIA zip by probing.")


def _extract_zip_entries(content: bytes) -> zipfile.ZipFile:
    return zipfile.ZipFile(io.BytesIO(content))


def _find_sheet(sheet_names: Iterable[str], *, required: Iterable[str]) -> Optional[str]:
    tokens = [_normalize(token) for token in required]
    for name in sheet_names:
        normalized = _normalize(name)
        if all(token in normalized for token in tokens):
            return name
    return None


def _find_eia860_zip_links(page_url: str = EIA860_PAGE_URL) -> dict[int, str]:
    headers = {"User-Agent": "Mozilla/5.0"}
    html = requests.get(page_url, timeout=30, headers=headers).text
    soup = BeautifulSoup(html, "html.parser")
    links = [a.get("href") for a in soup.find_all("a", href=True)]
    pattern = re.compile(r"eia860(\\d{4})\\.zip", re.IGNORECASE)
    candidates: dict[int, str] = {}
    for link in links:
        match = pattern.search(link)
        if not match:
            continue
        year = int(match.group(1))
        candidates[year] = _abs_url(page_url, link)
    if not candidates:
        raise EIAElectricityError("No EIA-860 zip links found.")
    return candidates


def resolve_eia860_year(*, year: Optional[int] = None, page_url: str = EIA860_PAGE_URL) -> int:
    if year is not None:
        return year
    try:
        candidates = _find_eia860_zip_links(page_url=page_url)
        return max(candidates)
    except EIAElectricityError:
        return _probe_latest_year(
            page_url=page_url,
            filename_template="xls/eia860{year}.zip",
            archive_template="archive/xls/eia860{year}.zip",
            start_year=datetime.now().year,
        )


def fetch_eia860_zip_url(*, year: Optional[int] = None, page_url: str = EIA860_PAGE_URL) -> str:
    if year is not None:
        base = page_url.rstrip("/")
        return _abs_url(base + "/", f"xls/eia860{year}.zip")
    candidates = _find_eia860_zip_links(page_url=page_url)
    target_year = resolve_eia860_year(year=year, page_url=page_url)
    return candidates[target_year]


def _locate_eia860_files(zip_obj: zipfile.ZipFile) -> EIA860Files:
    generator = None
    multifuel = None
    for name in zip_obj.namelist():
        normalized = name.lower()
        if "3_1_generator" in normalized and normalized.endswith(".xlsx"):
            generator = name
        if "3_5_multifuel" in normalized and normalized.endswith(".xlsx"):
            multifuel = name
    if not generator or not multifuel:
        raise EIAElectricityError("Missing expected 3_1 or 3_5 workbooks in EIA-860 zip.")
    return EIA860Files(generator=generator, multifuel=multifuel)


def _load_sheet(zip_obj: zipfile.ZipFile, workbook: str, sheet: str) -> pd.DataFrame:
    with zip_obj.open(workbook) as handle:
        return pd.read_excel(handle, sheet_name=sheet, engine="openpyxl")


def _extract_eia860_workbook(
    zip_obj: zipfile.ZipFile,
    workbook: str,
) -> dict[str, pd.DataFrame]:
    with zip_obj.open(workbook) as handle:
        excel = pd.ExcelFile(handle, engine="openpyxl")
        sheet_names = excel.sheet_names

    sheets: dict[str, pd.DataFrame] = {}
    operable = _find_sheet(sheet_names, required=["operable"])
    proposed = _find_sheet(sheet_names, required=["proposed"])

    retired_canceled = _find_sheet(sheet_names, required=["retired", "cancel"])
    retired_only = _find_sheet(sheet_names, required=["retired"]) if not retired_canceled else None
    canceled_only = _find_sheet(sheet_names, required=["cancel"]) if not retired_canceled else None

    if operable:
        sheets["operable"] = _load_sheet(zip_obj, workbook, operable)
    if proposed:
        sheets["proposed"] = _load_sheet(zip_obj, workbook, proposed)
    if retired_only:
        sheets["retired"] = _load_sheet(zip_obj, workbook, retired_only)
    if canceled_only:
        sheets["canceled"] = _load_sheet(zip_obj, workbook, canceled_only)
    if retired_canceled:
        sheets["retired_canceled"] = _load_sheet(zip_obj, workbook, retired_canceled)

    if not sheets:
        raise EIAElectricityError(f"No expected sheets found in workbook {workbook}.")
    return sheets


def update_eia860_raw_csvs(
    *,
    year: Optional[int] = None,
    output_dir: Path | str = DATA_DIR,
    page_url: str = EIA860_PAGE_URL,
    timeout: float = 60.0,
) -> dict[str, Path]:
    """Download EIA-860 zip and emit raw CSVs for generator + multifuel tabs."""
    target_year = resolve_eia860_year(year=year, page_url=page_url)
    zip_bytes = None
    zip_url = fetch_eia860_zip_url(year=target_year, page_url=page_url)
    try:
        zip_bytes = _download_bytes(zip_url, timeout=timeout)
        if not _is_zip_bytes(zip_bytes):
            raise EIAElectricityError("Primary EIA-860 download did not return a zip file.")
    except Exception:
        fallback_url = _abs_url(page_url, f"archive/xls/eia860{target_year}.zip")
        zip_bytes = _download_bytes(fallback_url, timeout=timeout)
        if not _is_zip_bytes(zip_bytes):
            raise EIAElectricityError("Fallback EIA-860 download did not return a zip file.")
    zip_obj = _extract_zip_entries(zip_bytes)
    files = _locate_eia860_files(zip_obj)

    output_base = Path(output_dir)
    output_base.mkdir(parents=True, exist_ok=True)

    generator_sheets = _extract_eia860_workbook(zip_obj, files.generator)
    multifuel_sheets = _extract_eia860_workbook(zip_obj, files.multifuel)

    outputs: dict[str, Path] = {}
    for key, df in generator_sheets.items():
        name = f"eia860_{target_year}_generator_{key}.csv"
        path = output_base / name
        df.to_csv(path, index=False)
        outputs[name] = path

    for key, df in multifuel_sheets.items():
        name = f"eia860_{target_year}_multifuel_{key}.csv"
        path = output_base / name
        df.to_csv(path, index=False)
        outputs[name] = path

    return outputs


def _find_eia923_zip_links(page_url: str = EIA923_PAGE_URL) -> dict[int, str]:
    headers = {"User-Agent": "Mozilla/5.0"}
    html = requests.get(page_url, timeout=30, headers=headers).text
    soup = BeautifulSoup(html, "html.parser")
    links = [a.get("href") for a in soup.find_all("a", href=True)]
    pattern = re.compile(r"f923_(\\d{4})\\.zip", re.IGNORECASE)
    candidates: dict[int, str] = {}
    for link in links:
        match = pattern.search(link)
        if not match:
            continue
        year = int(match.group(1))
        candidates[year] = _abs_url(page_url, link)
    if not candidates:
        raise EIAElectricityError("No EIA-923 zip links found.")
    return candidates


def _locate_eia923_workbook(zip_obj: zipfile.ZipFile, year: int) -> EIA923Files:
    for name in zip_obj.namelist():
        if name.lower().startswith("eia923_schedules_2_3_4_5") and name.lower().endswith(".xlsx"):
            return EIA923Files(workbook=name)
    raise EIAElectricityError(f"EIA-923 schedule workbook not found in {year} zip.")


def _select_eia923_year(
    *,
    year: Optional[int],
    page_url: str = EIA923_PAGE_URL,
    timeout: float = 60.0,
) -> tuple[int, bytes, EIA923Files]:
    if year is not None:
        base = page_url.rstrip("/")
        zip_url = _abs_url(base + "/", f"xls/f923_{year}.zip")
        zip_bytes = _download_bytes(zip_url, timeout=timeout)
        if not _is_zip_bytes(zip_bytes):
            raise EIAElectricityError("EIA-923 download did not return a zip file.")
        zip_obj = _extract_zip_entries(zip_bytes)
        files = _locate_eia923_workbook(zip_obj, year)
        return year, zip_bytes, files

    try:
        candidates = _find_eia923_zip_links(page_url=page_url)
    except EIAElectricityError:
        candidates = {}

    if candidates:
        for candidate_year in sorted(candidates.keys(), reverse=True):
            zip_url = candidates[candidate_year]
            zip_bytes = _download_bytes(zip_url, timeout=timeout)
            if not _is_zip_bytes(zip_bytes):
                continue
            zip_obj = _extract_zip_entries(zip_bytes)
            try:
                files = _locate_eia923_workbook(zip_obj, candidate_year)
            except EIAElectricityError:
                continue
            if "M_12" in files.workbook:
                return candidate_year, zip_bytes, files
            if candidate_year == max(candidates):
                fallback = (candidate_year, zip_bytes, files)
    if "fallback" in locals():
        return fallback
    try:
        probed_year = _probe_latest_year(
            page_url=page_url,
            filename_template="xls/f923_{year}.zip",
            archive_template="archive/xls/f923_{year}.zip",
            start_year=datetime.now().year,
        )
        zip_url = _abs_url(page_url, f"xls/f923_{probed_year}.zip")
        zip_bytes = _download_bytes(zip_url, timeout=timeout)
        if not _is_zip_bytes(zip_bytes):
            raise EIAElectricityError("EIA-923 probed download did not return a zip file.")
        zip_obj = _extract_zip_entries(zip_bytes)
        files = _locate_eia923_workbook(zip_obj, probed_year)
        return probed_year, zip_bytes, files
    except Exception as exc:
        raise EIAElectricityError("Could not determine a usable EIA-923 workbook.") from exc


def _extract_eia923_sheets(zip_obj: zipfile.ZipFile, workbook: str) -> dict[str, pd.DataFrame]:
    with zip_obj.open(workbook) as handle:
        excel = pd.ExcelFile(handle, engine="openpyxl")
        sheet_names = excel.sheet_names

    def find_sheet(matchers: Iterable[str]) -> Optional[str]:
        for name in sheet_names:
            normalized = _normalize(name)
            if all(token in normalized for token in matchers):
                return name
        return None

    sheet_map = {
        "page1_generation_fuel": find_sheet(["page1", "generation"]),
        "page3_boiler_fuel": find_sheet(["page3", "boiler", "fuel"]),
        "page4_generator_data": find_sheet(["page4", "generator"]),
        "page5_fuel_receipts_costs": find_sheet(["page5", "fuel", "receipts"]),
    }

    missing = [key for key, name in sheet_map.items() if not name]
    if missing:
        raise EIAElectricityError(f"Missing EIA-923 sheets: {missing}")

    outputs: dict[str, pd.DataFrame] = {}
    for key, sheet in sheet_map.items():
        outputs[key] = _load_sheet(zip_obj, workbook, sheet)
    return outputs


def update_eia923_raw_csvs(
    *,
    year: Optional[int] = None,
    output_dir: Path | str = DATA_DIR,
    page_url: str = EIA923_PAGE_URL,
    timeout: float = 60.0,
) -> dict[str, Path]:
    """Download EIA-923 zip and emit raw CSVs for required pages."""
    target_year, zip_bytes, files = _select_eia923_year(year=year, page_url=page_url, timeout=timeout)
    zip_obj = _extract_zip_entries(zip_bytes)
    sheets = _extract_eia923_sheets(zip_obj, files.workbook)

    output_base = Path(output_dir)
    output_base.mkdir(parents=True, exist_ok=True)

    outputs: dict[str, Path] = {}
    for key, df in sheets.items():
        name = f"eia923_{target_year}_{key}.csv"
        path = output_base / name
        df.to_csv(path, index=False)
        outputs[name] = path

    return outputs


__all__ = [
    "fetch_eia860_zip_url",
    "resolve_eia860_year",
    "update_eia860_raw_csvs",
    "update_eia923_raw_csvs",
]
