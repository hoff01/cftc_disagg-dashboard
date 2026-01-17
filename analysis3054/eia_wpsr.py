"""Helpers for EIA Weekly Petroleum Status Report (WPSR) tables and snapshots."""

from __future__ import annotations

from dataclasses import dataclass
import csv
import asyncio
import io
import math
import os
import platform
from pathlib import Path
import subprocess
import threading
from typing import Mapping, Optional, Sequence

import httpx
import pandas as pd


WPSR_TABLE_URLS = {
    "table1": "https://ir.eia.gov/wpsr/table1.csv",
    "table2": "https://ir.eia.gov/wpsr/table2.csv",
    "table3": "https://ir.eia.gov/wpsr/table3.csv",
    "table4": "https://ir.eia.gov/wpsr/table4.csv",
    "table5": "https://ir.eia.gov/wpsr/table5.csv",
    "table5a": "https://ir.eia.gov/wpsr/table5a.csv",
    "table6": "https://ir.eia.gov/wpsr/table6.csv",
    "table7": "https://ir.eia.gov/wpsr/table7.csv",
    "table9": "https://ir.eia.gov/wpsr/table9.csv",
}

DEFAULT_WPSR_TABLES = (
    "table1",
    "table2",
    "table3",
    "table4",
    "table5",
    "table5a",
    "table6",
    "table7",
    "table9",
)


class WPSRDataError(Exception):
    """Raised when WPSR tables cannot be retrieved or parsed."""


@dataclass(frozen=True)
class WPSRSnapshotRow:
    label: str
    delta: Optional[float]
    stocks: Optional[float]


@dataclass(frozen=True)
class WPSRSnapshotTable:
    title: str
    rows: list[WPSRSnapshotRow]


def _normalize_table_name(name: object) -> str:
    text = str(name).strip().lower()
    if not text.startswith("table"):
        text = f"table{text}"
    if text not in WPSR_TABLE_URLS:
        raise KeyError(f"Unknown WPSR table '{name}'.")
    return text


async def fetch_wpsr_tables_async(
    tables: Optional[Sequence[object]] = None,
    *,
    timeout: float = 8.0,
    max_workers: int = 8,
    retries: int = 2,
) -> dict[str, pd.DataFrame]:
    """Fetch WPSR CSV tables concurrently."""
    requested = tables or DEFAULT_WPSR_TABLES
    ordered: list[str] = []
    seen = set()
    for name in requested:
        key = _normalize_table_name(name)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(key)

    limits = httpx.Limits(max_connections=max_workers, max_keepalive_connections=max_workers)
    try:
        import h2.config  # type: ignore

        http2_enabled = True
    except Exception:
        http2_enabled = False
    transport = httpx.AsyncHTTPTransport(http2=http2_enabled, limits=limits)
    headers = {
        "Accept": "text/csv, */*;q=0.8",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
    }

    def _read_ragged_csv(payload: bytes, encoding: Optional[str] = None) -> pd.DataFrame:
        text = payload.decode(encoding or "utf-8", errors="replace")
        rows = list(csv.reader(io.StringIO(text)))
        if not rows:
            return pd.DataFrame()
        width = max(len(row) for row in rows)
        padded = [row + [""] * (width - len(row)) for row in rows]
        columns = [f"col_{i}" for i in range(width)]
        return pd.DataFrame(padded, columns=columns)

    def _read_csv_bytes(payload: bytes) -> pd.DataFrame:
        try:
            return pd.read_csv(io.BytesIO(payload))
        except UnicodeDecodeError:
            try:
                return pd.read_csv(io.BytesIO(payload), encoding="latin-1")
            except pd.errors.ParserError:
                return _read_ragged_csv(payload, encoding="latin-1")
        except pd.errors.ParserError:
            return _read_ragged_csv(payload)

    async with httpx.AsyncClient(
        transport=transport,
        headers=headers,
        timeout=httpx.Timeout(timeout),
        follow_redirects=True,
    ) as client:
        semaphore = asyncio.Semaphore(max_workers)

        async def fetch_one(name: str, url: str) -> tuple[str, pd.DataFrame]:
            last_exc: Optional[BaseException] = None
            for attempt in range(retries + 1):
                try:
                    async with semaphore:
                        response = await client.get(url)
                    response.raise_for_status()
                    df = _read_csv_bytes(response.content)
                    return name, df
                except (httpx.HTTPError, pd.errors.ParserError, OSError) as exc:
                    last_exc = exc
                    if attempt < retries:
                        await asyncio.sleep(0.1 * (2**attempt))
                        continue
                    raise WPSRDataError(f"Failed to fetch {name} from {url}.") from exc
            raise WPSRDataError(f"Failed to fetch {name} from {url}.") from last_exc

        tasks = [fetch_one(name, WPSR_TABLE_URLS[name]) for name in ordered]
        results = await asyncio.gather(*tasks)

    return dict(results)


def _run_coro_in_thread(coro: asyncio.Future) -> dict[str, pd.DataFrame]:
    payload: dict[str, pd.DataFrame] = {}
    errors: list[BaseException] = []

    def runner() -> None:
        try:
            payload.update(asyncio.run(coro))
        except BaseException as exc:  # pragma: no cover - passthrough
            errors.append(exc)

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    thread.join()

    if errors:
        raise errors[0]
    return payload


def fetch_wpsr_tables(
    tables: Optional[Sequence[object]] = None,
    *,
    timeout: float = 8.0,
    max_workers: int = 8,
    retries: int = 2,
) -> dict[str, pd.DataFrame]:
    """Fetch WPSR CSV tables with a synchronous wrapper."""
    coro = fetch_wpsr_tables_async(
        tables=tables,
        timeout=timeout,
        max_workers=max_workers,
        retries=retries,
    )
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop is None:
        return asyncio.run(coro)
    return _run_coro_in_thread(coro)


def _clean_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip()


def _coerce_float(value: object) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isnan(value):
            return None
        return float(value)
    text = str(value).strip().replace(",", "")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _value_column(df: pd.DataFrame) -> str:
    stub_cols = 0
    if "STUB_1" in df.columns:
        stub_cols += 1
    if "STUB_2" in df.columns:
        stub_cols += 1
    if stub_cols >= len(df.columns):
        raise WPSRDataError("WPSR table is missing data columns.")
    return str(df.columns[stub_cols])


def _section_by_stub2(
    df: pd.DataFrame,
    *,
    stub1_label: str,
    start_label: str,
    end_label: Optional[str] = None,
) -> pd.DataFrame:
    if "STUB_1" not in df.columns or "STUB_2" not in df.columns:
        raise WPSRDataError("Expected STUB_1/STUB_2 columns missing.")
    stub1 = _clean_series(df["STUB_1"])
    subset = df.loc[stub1 == stub1_label].copy()
    stub2 = _clean_series(subset["STUB_2"])
    if (stub2 == start_label).sum() == 0:
        raise WPSRDataError(f"Could not locate '{start_label}' in WPSR table.")
    start_idx = subset.index[stub2 == start_label][0]
    if end_label:
        end_candidates = subset.index[(stub2 == end_label) & (subset.index > start_idx)]
        end_idx = int(end_candidates.min()) if len(end_candidates) else None
    else:
        end_idx = None
    if end_idx is None:
        return subset.loc[start_idx:]
    return subset.loc[start_idx : end_idx - 1]


def _section_by_stub1(
    df: pd.DataFrame,
    *,
    start_label: str,
    end_label: Optional[str] = None,
) -> pd.DataFrame:
    if "STUB_1" not in df.columns:
        raise WPSRDataError("Expected STUB_1 column missing.")
    stub1 = _clean_series(df["STUB_1"])
    if (stub1 == start_label).sum() == 0:
        raise WPSRDataError(f"Could not locate '{start_label}' in WPSR table.")
    start_idx = df.index[stub1 == start_label][0]
    if end_label:
        end_candidates = df.index[(stub1 == end_label) & (df.index > start_idx)]
        end_idx = int(end_candidates.min()) if len(end_candidates) else None
    else:
        end_idx = None
    if end_idx is None:
        return df.loc[start_idx:]
    return df.loc[start_idx : end_idx - 1]


def _extract_rows(
    section: pd.DataFrame,
    *,
    stub_col: str,
    mapping: Sequence[tuple[str, str]],
    value_col: str,
    diff_col: str,
) -> list[WPSRSnapshotRow]:
    if stub_col not in section.columns:
        raise WPSRDataError(f"Expected {stub_col} column missing.")
    if diff_col not in section.columns:
        raise WPSRDataError("Expected Difference column missing.")
    stub = _clean_series(section[stub_col])
    rows: list[WPSRSnapshotRow] = []
    for label, stub_label in mapping:
        match = section.loc[stub == stub_label]
        if match.empty:
            raise WPSRDataError(f"Could not locate '{stub_label}' in WPSR table.")
        row = match.iloc[0]
        rows.append(
            WPSRSnapshotRow(
                label=label,
                delta=_coerce_float(row.get(diff_col)),
                stocks=_coerce_float(row.get(value_col)),
            )
        )
    return rows


def build_wpsr_snapshot_tables(tables: Mapping[str, pd.DataFrame]) -> list[WPSRSnapshotTable]:
    """Extract snapshot rows for gasoline, distillates, jet, and crude."""
    if "table5a" not in tables:
        raise WPSRDataError("table5a is required for gasoline PADD data.")
    if "table6" not in tables:
        raise WPSRDataError("table6 is required for distillate and jet data.")
    if "table4" not in tables:
        raise WPSRDataError("table4 is required for crude data.")

    gas_df = tables["table5a"]
    gas_section = _section_by_stub2(
        gas_df,
        stub1_label="Motor Gasoline",
        start_label="Total Motor Gasoline",
        end_label="Finished Motor Gasoline",
    )
    gas_rows = _extract_rows(
        gas_section,
        stub_col="STUB_2",
        mapping=(
            ("Total US", "Total Motor Gasoline"),
            ("Padd 1", "East Coast (PADD 1)"),
            ("1a", "New England (PADD 1A)"),
            ("1b", "Central Atlantic (PADD 1B)"),
            ("1c", "Lower Atlantic (PADD 1C)"),
            ("Padd 2", "Midwest (PADD 2)"),
            ("Padd 3", "Gulf Coast (PADD 3)"),
            ("Padd 4", "Rocky Mountain (PADD 4)"),
            ("Padd 5", "West Coast (PADD 5)"),
        ),
        value_col=_value_column(gas_df),
        diff_col="Difference",
    )

    dist_df = tables["table6"]
    dist_section = _section_by_stub1(
        dist_df,
        start_label="Distillate Fuel Oil",
        end_label="15 ppm sulfur and Under",
    )
    dist_rows = _extract_rows(
        dist_section,
        stub_col="STUB_1",
        mapping=(
            ("Total US", "Distillate Fuel Oil"),
            ("Padd 1", "East Coast (PADD 1)"),
            ("1a", "New England (PADD 1A)"),
            ("1b", "Central Atlantic (PADD 1B)"),
            ("1c", "Lower Atlantic (PADD 1C)"),
            ("Padd 2", "Midwest (PADD 2)"),
            ("Padd 3", "Gulf Coast (PADD 3)"),
            ("Padd 4", "Rocky Mountain (PADD 4)"),
            ("Padd 5", "West Coast (PADD 5)"),
        ),
        value_col=_value_column(dist_df),
        diff_col="Difference",
    )

    jet_section = _section_by_stub1(
        dist_df,
        start_label="Kerosene-Type Jet Fuel",
        end_label="Residual Fuel Oil",
    )
    jet_rows = _extract_rows(
        jet_section,
        stub_col="STUB_1",
        mapping=(
            ("Total US", "Kerosene-Type Jet Fuel"),
            ("Padd 1", "East Coast (PADD 1)"),
            ("Padd 2", "Midwest (PADD 2)"),
            ("Padd 3", "Gulf Coast (PADD 3)"),
            ("Padd 4", "Rocky Mountain (PADD 4)"),
            ("Padd 5", "West Coast (PADD 5)"),
        ),
        value_col=_value_column(dist_df),
        diff_col="Difference",
    )

    crude_df = tables["table4"]
    crude_rows = _extract_rows(
        crude_df,
        stub_col="STUB_1",
        mapping=(
            ("Total US", "Commercial (Excluding SPR)"),
            ("Padd 1", "East Coast (PADD 1)"),
            ("Padd 2", "Midwest (PADD 2)"),
            ("Padd 3", "Gulf Coast (PADD 3)"),
            ("Padd 4", "Rocky Mountain (PADD 4)"),
            ("Padd 5", "West Coast (PADD 5)"),
            ("Cushing", "Cushing"),
            ("SPR", "SPR"),
        ),
        value_col=_value_column(crude_df),
        diff_col="Difference",
    )

    return [
        WPSRSnapshotTable(title="Gasoline", rows=gas_rows),
        WPSRSnapshotTable(title="Distillates", rows=dist_rows),
        WPSRSnapshotTable(title="Jet", rows=jet_rows),
        WPSRSnapshotTable(title="Crude", rows=crude_rows),
    ]


def _format_delta(value: Optional[float]) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    if value < 0:
        return f"({abs(value):.3f})"
    return f"{value:.3f}"


def _format_stock(value: Optional[float]) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    if value < 0:
        return f"({abs(value):.1f})"
    return f"{value:.1f}"


def _load_font(candidates: Sequence[str], size: int) -> "ImageFont.FreeTypeFont":
    from PIL import ImageFont

    for path in candidates:
        if not path:
            continue
        if not os.path.exists(path):
            continue
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _font_candidates(style: str) -> list[str]:
    system = platform.system()
    if system == "Windows":
        if style == "bold":
            return [
                r"C:\Windows\Fonts\arialbd.ttf",
                r"C:\Windows\Fonts\segoeuib.ttf",
                r"C:\Windows\Fonts\calibrib.ttf",
            ]
        return [
            r"C:\Windows\Fonts\arial.ttf",
            r"C:\Windows\Fonts\segoeui.ttf",
            r"C:\Windows\Fonts\calibri.ttf",
        ]
    if system == "Darwin":
        if style == "bold":
            return [
                "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
                "/Library/Fonts/Arial Bold.ttf",
                "/System/Library/Fonts/Helvetica.ttc",
            ]
        return [
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/Library/Fonts/Arial.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
        ]
    if style == "bold":
        return [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        ]
    return [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    ]


def render_wpsr_snapshot(
    tables: Sequence[WPSRSnapshotTable],
    *,
    output_path: Path | str = "wpsr_snapshot.png",
) -> Path:
    """Render the snapshot tables to a PNG."""
    from PIL import Image, ImageDraw

    output_path = Path(output_path)

    title_font = _load_font(_font_candidates("bold"), size=16)
    header_font = _load_font(_font_candidates("bold"), size=12)
    body_font = _load_font(_font_candidates("regular"), size=12)

    header_text_delta = "Δ w/w"
    header_text_stock = "Stocks"

    dummy = Image.new("RGB", (10, 10))
    draw = ImageDraw.Draw(dummy)

    def text_width(text: str, font) -> int:
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0]

    def text_height(text: str, font) -> int:
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[3] - bbox[1]

    labels = [f"{row.label}:" for table in tables for row in table.rows]
    delta_values = [_format_delta(row.delta) for table in tables for row in table.rows]
    stock_values = [_format_stock(row.stocks) for table in tables for row in table.rows]

    label_width = max([text_width(label, body_font) for label in labels] + [60])
    value_width = max(
        [text_width(header_text_delta, header_font), text_width(header_text_stock, header_font)]
        + [text_width(value, body_font) for value in delta_values + stock_values]
        + [60]
    )

    cell_pad = 5
    label_width += cell_pad * 2
    value_width += cell_pad * 2

    row_height = max(text_height("Ag", body_font), text_height("Ag", header_font), 14) + cell_pad * 2
    header_height = row_height
    title_height = text_height("Ag", title_font)

    table_width = label_width + value_width * 2
    table_gap = 20
    margin_x = 14
    margin_y = 12
    title_gap = 4

    max_rows = max(len(table.rows) for table in tables)
    height = margin_y * 2 + title_height + title_gap + header_height + row_height * max_rows
    width = margin_x * 2 + len(tables) * table_width + (len(tables) - 1) * table_gap

    image = Image.new("RGB", (int(width), int(height)), "white")
    draw = ImageDraw.Draw(image)

    line_color = "#000000"
    header_fill = "#d9e2f3"
    total_fill = "#cfcfcf"
    negative_color = "#c00000"
    text_color = "#000000"

    for index, table in enumerate(tables):
        x0 = margin_x + index * (table_width + table_gap)
        y_title = margin_y
        title_text = table.title
        title_w = text_width(title_text, title_font)
        draw.text((x0 + (table_width - title_w) / 2, y_title), title_text, fill=text_color, font=title_font)

        y0 = y_title + title_height + title_gap
        y1 = y0 + header_height
        y_end = y1 + row_height * len(table.rows)
        x1 = x0 + label_width
        x2 = x1 + value_width
        x3 = x2 + value_width

        draw.rectangle([x1, y0, x2, y1], fill=header_fill, outline=None)
        draw.rectangle([x2, y0, x3, y1], fill=header_fill, outline=None)

        for row_index, row in enumerate(table.rows):
            if row.label == "Total US":
                row_top = y1 + row_index * row_height
                draw.rectangle([x0, row_top, x3, row_top + row_height], fill=total_fill, outline=None)

        draw.rectangle([x0, y0, x3, y_end], outline=line_color, width=1)
        draw.line([x1, y0, x1, y_end], fill=line_color, width=1)
        draw.line([x2, y0, x2, y_end], fill=line_color, width=1)
        draw.line([x0, y1, x3, y1], fill=line_color, width=1)

        for i in range(1, len(table.rows)):
            y = y1 + i * row_height
            draw.line([x0, y, x3, y], fill=line_color, width=1)

        header_delta_w = text_width(header_text_delta, header_font)
        header_stock_w = text_width(header_text_stock, header_font)
        header_y = y0 + (header_height - text_height("Ag", header_font)) / 2
        draw.text(
            (x1 + (value_width - header_delta_w) / 2, header_y),
            header_text_delta,
            fill=text_color,
            font=header_font,
        )
        draw.text(
            (x2 + (value_width - header_stock_w) / 2, header_y),
            header_text_stock,
            fill=text_color,
            font=header_font,
        )

        for row_index, row in enumerate(table.rows):
            row_top = y1 + row_index * row_height
            row_center = row_top + row_height / 2
            label_text = f"{row.label}:"
            delta_text = _format_delta(row.delta)
            stock_text = _format_stock(row.stocks)

            is_subpadd = row.label in {"1a", "1b", "1c"}
            label_font = header_font if row.label in {"Total US", "Padd 1", "Padd 2", "Padd 3", "Padd 4", "Padd 5", "Cushing", "SPR"} else body_font
            label_y = row_center - text_height("Ag", label_font) / 2
            label_x = x0 + cell_pad + (10 if is_subpadd else 0)
            draw.text((label_x, label_y), label_text, fill=text_color, font=label_font)

            delta_color = negative_color if row.delta is not None and row.delta < 0 else text_color
            value_font = header_font if row.label == "Total US" else body_font
            delta_w = text_width(delta_text, value_font)
            delta_x = x1 + (value_width - delta_w) / 2
            value_y = row_center - text_height("Ag", value_font) / 2
            draw.text((delta_x, value_y), delta_text, fill=delta_color, font=value_font)

            stock_w = text_width(stock_text, value_font)
            stock_x = x2 + (value_width - stock_w) / 2
            draw.text((stock_x, value_y), stock_text, fill=text_color, font=value_font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="PNG")
    return output_path


def _copy_png_to_clipboard(path: Path) -> None:
    system = platform.system()
    if system == "Darwin":
        escaped = str(path).replace('"', '\\"')
        left = chr(171)
        right = chr(187)
        script = f'set the clipboard to (read (POSIX file "{escaped}") as {left}class PNGf{right})'
        subprocess.run(["osascript", "-e", script], check=False)
        return
    if system == "Windows":
        ps_path = str(path).replace("'", "''")
        command = (
            "Add-Type -AssemblyName System.Windows.Forms;"
            "Add-Type -AssemblyName System.Drawing;"
            f"$img = [System.Drawing.Image]::FromFile('{ps_path}');"
            "[System.Windows.Forms.Clipboard]::SetImage($img);"
        )
        for exe in ("powershell", "pwsh"):
            try:
                subprocess.run([exe, "-sta", "-command", command], check=False)
                break
            except FileNotFoundError:
                continue


def _open_file(path: Path) -> None:
    system = platform.system()
    if system == "Darwin":
        subprocess.run(["open", str(path)], check=False)
        return
    if system == "Windows":
        os.startfile(str(path))  # type: ignore[attr-defined]
        return
    subprocess.run(["xdg-open", str(path)], check=False)


def create_wpsr_snapshot(
    *,
    output_path: Path | str = "wpsr_snapshot.png",
    tables: Optional[Mapping[str, pd.DataFrame]] = None,
    timeout: float = 8.0,
    max_workers: int = 8,
    retries: int = 2,
    open_image: bool = True,
    copy_clipboard: bool = True,
) -> Path:
    """Fetch WPSR tables and render the release snapshot PNG."""
    tables = dict(tables) if tables is not None else fetch_wpsr_tables(
        timeout=timeout,
        max_workers=max_workers,
        retries=retries,
    )
    snapshot_tables = build_wpsr_snapshot_tables(tables)
    path = render_wpsr_snapshot(snapshot_tables, output_path=output_path)
    if copy_clipboard:
        _copy_png_to_clipboard(path)
    if open_image:
        _open_file(path)
    return path


__all__ = [
    "WPSR_TABLE_URLS",
    "DEFAULT_WPSR_TABLES",
    "WPSRDataError",
    "WPSRSnapshotRow",
    "WPSRSnapshotTable",
    "fetch_wpsr_tables_async",
    "fetch_wpsr_tables",
    "build_wpsr_snapshot_tables",
    "render_wpsr_snapshot",
    "create_wpsr_snapshot",
]
