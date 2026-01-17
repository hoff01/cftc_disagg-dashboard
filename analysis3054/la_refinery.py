"""Utilities for downloading the Louisiana refinery activity report."""

import asyncio
import logging
import subprocess
import tempfile
import time
from pathlib import Path
from typing import List, Optional, Sequence

import pandas as pd
from playwright.async_api import Page, TimeoutError as PlaywrightTimeout, async_playwright

from .refinery_name_maps import apply_la_refinery_name_map_to_df

# Configure logging for production visibility
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("LA_Refinery_Scraper")

# Track whether we have already attempted to install Playwright browsers in this process
_PLAYWRIGHT_INSTALL_ATTEMPTED = False
_INPUT_POLL_INTERVAL = 0.2

DATA_DIR = Path(__file__).resolve().parent / "data" / "refinery"
LA_REFINERY_CACHE_CSV = DATA_DIR / "la_refinery_latest.csv"
LA_REFINERY_CACHE_PKL = DATA_DIR / "la_refinery_latest.pkl"
LA_REFINERY_URL = (
    "https://sonlite.dnr.state.la.us/ords/r/sonris_pub/sonris_data_portal/"
    "r3-activity-report-product-listing?clear=2466"
)


class ScraperError(Exception):
    """Custom exception for scraping failures."""


class RecaptchaRequired(ScraperError):
    """Raised when a recaptcha requires manual completion."""


def load_la_refinery_cache(
    csv_path: Optional[str | Path] = None,
    pkl_path: Optional[str | Path] = None,
) -> Optional[pd.DataFrame]:
    """Load the cached LA refinery dataset if available."""
    csv_path = Path(csv_path) if csv_path else LA_REFINERY_CACHE_CSV
    pkl_path = Path(pkl_path) if pkl_path else LA_REFINERY_CACHE_PKL

    if csv_path.exists():
        df = pd.read_csv(csv_path)
        return apply_la_refinery_name_map_to_df(df)
    if pkl_path.exists():
        df = pd.read_pickle(pkl_path)
        return apply_la_refinery_name_map_to_df(df)
    return None


def save_la_refinery_cache(
    df: pd.DataFrame,
    csv_path: Optional[str | Path] = None,
    pkl_path: Optional[str | Path] = None,
) -> None:
    """Persist the LA refinery dataset to CSV and PKL for distribution."""
    csv_path = Path(csv_path) if csv_path else LA_REFINERY_CACHE_CSV
    pkl_path = Path(pkl_path) if pkl_path else LA_REFINERY_CACHE_PKL

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df = apply_la_refinery_name_map_to_df(df)
    df.to_csv(csv_path, index=False)
    df.to_pickle(pkl_path)


async def safe_click(page: Page, selector_strategies: List[str], element_name: str) -> bool:
    """Attempts to click an element using a list of fallback selectors."""

    for selector in selector_strategies:
        try:
            element = page.locator(selector).first
            await element.wait_for(state="visible", timeout=5000)
            await element.scroll_into_view_if_needed()
            await element.click()
            logger.debug("Clicked '%s' using selector: %s", element_name, selector)
            return True
        except Exception:
            logger.debug("Strategy failed for '%s': %s", element_name, selector)
            continue

    logger.error("Failed to click '%s' after trying all strategies.", element_name)
    return False


async def safe_fill(
    page: Page,
    selector_strategies: List[str],
    value: str,
    element_name: str,
    *,
    timeout_ms: int = 5000,
) -> bool:
    """Attempts to fill an input field using a list of fallback selectors."""

    for selector in selector_strategies:
        try:
            element = page.locator(selector).first
            await element.wait_for(state="visible", timeout=timeout_ms)
            await element.fill(value)
            await element.press("Tab")  # Trigger 'change' events
            logger.debug("Filled '%s' using selector: %s", element_name, selector)
            return True
        except Exception:
            continue

    logger.error("Failed to fill '%s' after trying all strategies.", element_name)
    return False


def _selenium_safe_click(driver, selectors: Sequence[tuple], element_name: str, *, timeout: int = 10) -> bool:
    try:
        from selenium.webdriver.support import expected_conditions as EC
        from selenium.webdriver.support.ui import WebDriverWait
    except ImportError as exc:  # pragma: no cover - optional path
        raise ScraperError(
            "selenium (and websocket-client) are required for Stealthenium-based scraping."
        ) from exc

    for by, selector in selectors:
        try:
            element = WebDriverWait(driver, timeout).until(EC.element_to_be_clickable((by, selector)))
            if element is None:
                continue
            try:
                driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", element)
            except Exception:
                pass
            try:
                element.click()
            except Exception:
                driver.execute_script("arguments[0].click();", element)
            return True
        except Exception:
            continue
    logger.error("Failed to click '%s' after trying all strategies.", element_name)
    return False


def _selenium_safe_fill(
    driver,
    selectors: Sequence[tuple],
    value: str,
    element_name: str,
    *,
    timeout: int = 10,
) -> bool:
    try:
        from selenium.webdriver.support import expected_conditions as EC
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.common.keys import Keys
    except ImportError as exc:  # pragma: no cover - optional path
        raise ScraperError(
            "selenium (and websocket-client) are required for Stealthenium-based scraping."
        ) from exc

    for by, selector in selectors:
        try:
            element = WebDriverWait(driver, timeout).until(EC.visibility_of_element_located((by, selector)))
            if element is None:
                continue
            try:
                driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", element)
            except Exception:
                pass
            element.clear()
            element.send_keys(value)
            element.send_keys(Keys.TAB)
            return True
        except Exception:
            try:
                driver.execute_script(
                    "arguments[0].value = arguments[1];"
                    "arguments[0].dispatchEvent(new Event('change', {bubbles: true}));",
                    element,
                    value,
                )
                return True
            except Exception:
                continue
    logger.error("Failed to fill '%s' after trying all strategies.", element_name)
    return False


def _selenium_detect_recaptcha(driver) -> bool:
    try:
        frames = driver.find_elements("css selector", "iframe[src*='recaptcha']")
        if frames:
            return True
        if driver.find_elements("css selector", "input[id*='CALL_CAPTCHA']"):
            return True
        widgets = driver.find_elements(
            "xpath",
            "//*[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'recaptcha')]",
        )
        return len(widgets) > 0
    except Exception:
        return False


def _selenium_wait_for_recaptcha_clear(
    driver,
    *,
    timeout_seconds: float,
    poll_interval_seconds: float,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if not _selenium_detect_recaptcha(driver):
            return
        time.sleep(poll_interval_seconds)
    raise ScraperError("Recaptcha still present after waiting for manual completion.")


def _selenium_wait_for_apex_processing(driver, *, timeout_seconds: float = 30.0) -> None:
    try:
        from selenium.webdriver.support import expected_conditions as EC
        from selenium.webdriver.support.ui import WebDriverWait
    except ImportError as exc:  # pragma: no cover - optional path
        raise ScraperError(
            "selenium (and websocket-client) are required for Stealthenium-based scraping."
        ) from exc

    try:
        WebDriverWait(driver, 2).until(EC.visibility_of_element_located(("css selector", ".u-Processing")))
    except Exception:
        pass
    try:
        WebDriverWait(driver, timeout_seconds).until(EC.invisibility_of_element_located(("css selector", ".u-Processing")))
    except Exception:
        pass


def _selenium_wait_for_results(driver, *, timeout_seconds: float = 30.0) -> bool:
    start = time.monotonic()
    row_selectors = [
        "table.a-IRR-table tbody tr",
        "table[role='grid'] tbody tr",
        "#report_data tbody tr",
    ]
    no_data_selectors = [
        ".a-IRR-noDataMsg",
    ]
    while time.monotonic() - start < timeout_seconds:
        for selector in row_selectors:
            if driver.find_elements("css selector", selector):
                return True
        for selector in no_data_selectors:
            elements = driver.find_elements("css selector", selector)
            if elements and elements[0].is_displayed():
                return True
        time.sleep(0.5)
    return False


def _selenium_wait_for_inputs_or_recaptcha(
    driver,
    selectors: Sequence[tuple],
    *,
    timeout_seconds: float,
    poll_interval_seconds: float = 0.5,
) -> str:
    """Wait for input fields to appear or recaptcha to block the page."""
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if _selenium_detect_recaptcha(driver):
            return "recaptcha"
        for by, selector in selectors:
            if driver.find_elements(by, selector):
                return "ready"
        time.sleep(poll_interval_seconds)
    return "timeout"


def _selenium_expand_report_parameters(driver) -> bool:
    try:
        from selenium.webdriver.common.by import By
    except ImportError:  # pragma: no cover - optional path
        return False

    selectors = [
        (By.XPATH, "//button[contains(., 'Report Parameters')]"),
        (By.XPATH, "//button[.//span[contains(., 'Report Parameters')]]"),
        (By.XPATH, "//div[contains(@class, 't-Region-header')]//*[contains(., 'Report Parameters')]"),
        (By.XPATH, "//span[contains(., 'Report Parameters')]"),
        (By.CSS_SELECTOR, "button[aria-controls*='REPORT']"),
    ]
    if _selenium_safe_click(driver, selectors, "Report Parameters", timeout=5):
        return True
    try:
        clicked = driver.execute_script(
            """
            const nodes = Array.from(document.querySelectorAll('button,a,span,div'));
            const target = nodes.find(node => node.textContent && node.textContent.trim().includes('Report Parameters'));
            if (!target) return false;
            const clickable = target.closest('button,a') || target;
            clickable.click();
            return true;
            """
        )
        return bool(clicked)
    except Exception:
        return False


def _selenium_wait_for_download_dialog(driver, *, timeout_seconds: float = 10.0) -> bool:
    selectors = [
        "div[role='dialog']",
        ".ui-dialog",
        ".a-IRR-dialog",
    ]
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        for selector in selectors:
            if driver.find_elements("css selector", selector):
                return True
        time.sleep(0.5)
    return False


def _selenium_select_csv_format(driver) -> bool:
    try:
        from selenium.webdriver.common.by import By
    except ImportError:  # pragma: no cover - optional path
        return False

    csv_selectors = [
        (By.XPATH, "//label[contains(., 'CSV')]"),
        (By.CSS_SELECTOR, "input[type='radio'][value*='CSV']"),
        (By.CSS_SELECTOR, "input[type='radio'][value*='csv']"),
    ]
    if _selenium_safe_click(driver, csv_selectors, "CSV Radio", timeout=5):
        return True

    select_candidates = driver.find_elements(By.CSS_SELECTOR, "select")
    for select in select_candidates:
        try:
            if "format" not in (select.get_attribute("id") or "").lower() and "format" not in (
                select.get_attribute("name") or ""
            ).lower():
                continue
            for option in select.find_elements(By.TAG_NAME, "option"):
                if "csv" in option.text.lower():
                    option.click()
                    return True
        except Exception:
            continue

    return False


def _wait_for_download(directory: Path, *, timeout_seconds: float) -> Path:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        csv_files = list(directory.glob("*.csv"))
        if csv_files:
            if not any(directory.glob("*.crdownload")):
                return csv_files[0]
        time.sleep(0.5)
    raise ScraperError("CSV download did not complete in time.")


def _launch_selenium_driver(download_dir: Path, *, headless: bool, use_stealth: bool):
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options as ChromeOptions
    except ImportError as exc:  # pragma: no cover - optional path
        raise ScraperError(
            "selenium (and websocket-client) are required for Stealthenium-based scraping."
        ) from exc

    options = ChromeOptions()
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    if headless:
        options.add_argument("--headless=new")

    prefs = {
        "download.default_directory": str(download_dir),
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "safebrowsing.enabled": True,
    }
    options.add_experimental_option("prefs", prefs)

    driver = webdriver.Chrome(options=options)

    if use_stealth:
        try:
            from stealthenium import stealth

            stealth(
                driver,
                languages=["en-US", "en"],
                vendor="Google Inc.",
                platform="Win32",
                webgl_vendor="Intel Inc.",
                renderer="Intel Iris OpenGL Engine",
            )
            logger.info("Stealthenium applied to Selenium session.")
        except Exception as exc:
            logger.warning("Stealthenium unavailable or failed: %s", exc)

    return driver


def fetch_la_refinery_data_selenium(
    start_date: str = "01-JAN-2018",
    end_date: Optional[str] = None,
    *,
    headless: bool = True,
    wait_for_recaptcha: bool = True,
    recaptcha_timeout: float = 25.0,
    recaptcha_poll_interval: float = 5.0,
    download_timeout: float = 10.0,
    use_stealth: bool = True,
    page_ready_timeout: float = 5.0,
) -> pd.DataFrame:
    """Scrapes the LA DNR SONRIS Data Portal using Selenium + Stealthenium."""
    try:
        from selenium.webdriver.common.by import By
    except ImportError as exc:  # pragma: no cover - optional path
        raise ScraperError(
            "selenium (and websocket-client) are required for Stealthenium-based scraping."
        ) from exc

    with tempfile.TemporaryDirectory() as temp_dir:
        download_dir = Path(temp_dir)
        driver = _launch_selenium_driver(download_dir, headless=headless, use_stealth=use_stealth)
        try:
            begin_date_selectors = [
                (By.CSS_SELECTOR, "input[aria-label*='Begin Report Date']"),
                (By.CSS_SELECTOR, "input[id*='REPORT_DATE_BEGIN']"),
                (By.CSS_SELECTOR, "input[name*='REPORT_DATE_BEGIN']"),
                (By.CSS_SELECTOR, "input[id*='BEGIN_REPORT_DATE']"),
                (By.CSS_SELECTOR, "input[name*='BEGIN_REPORT_DATE']"),
                (By.XPATH, "//label[contains(., 'Begin Report Date')]/following::input[1]"),
                (By.CSS_SELECTOR, "input[placeholder*='DD-MON-YYYY']"),
            ]
            driver.get(LA_REFINERY_URL)

            page_state = _selenium_wait_for_inputs_or_recaptcha(
                driver,
                begin_date_selectors,
                timeout_seconds=page_ready_timeout,
            )
            if page_state == "recaptcha":
                if headless:
                    raise RecaptchaRequired("Recaptcha detected while running headless.")
                _selenium_wait_for_recaptcha_clear(
                    driver,
                    timeout_seconds=recaptcha_timeout,
                    poll_interval_seconds=recaptcha_poll_interval,
                )
            elif page_state == "timeout":
                if wait_for_recaptcha and _selenium_detect_recaptcha(driver):
                    logger.warning("Recaptcha detected; waiting for manual completion.")
                    if headless:
                        raise RecaptchaRequired("Recaptcha detected while running headless.")
                    _selenium_wait_for_recaptcha_clear(
                        driver,
                        timeout_seconds=recaptcha_timeout,
                        poll_interval_seconds=recaptcha_poll_interval,
                    )
                logger.warning("Inputs not visible yet; attempting to expand Report Parameters.")

            _selenium_expand_report_parameters(driver)

            page_state = _selenium_wait_for_inputs_or_recaptcha(
                driver,
                begin_date_selectors,
                timeout_seconds=page_ready_timeout,
            )
            if page_state == "recaptcha":
                if headless:
                    raise RecaptchaRequired("Recaptcha detected while running headless.")
                _selenium_wait_for_recaptcha_clear(
                    driver,
                    timeout_seconds=recaptcha_timeout,
                    poll_interval_seconds=recaptcha_poll_interval,
                )
            if not _selenium_safe_fill(driver, begin_date_selectors, start_date, "Begin Date", timeout=10):
                _selenium_expand_report_parameters(driver)
                if not _selenium_safe_fill(driver, begin_date_selectors, start_date, "Begin Date", timeout=10):
                    raise ScraperError("Could not locate Begin Date input field.")

            if end_date:
                end_date_selectors = [
                    (By.CSS_SELECTOR, "input[aria-label*='End Report Date']"),
                    (By.CSS_SELECTOR, "input[id*='REPORT_DATE_END']"),
                    (By.CSS_SELECTOR, "input[name*='REPORT_DATE_END']"),
                    (By.CSS_SELECTOR, "input[id*='END_REPORT_DATE']"),
                    (By.CSS_SELECTOR, "input[name*='END_REPORT_DATE']"),
                    (By.XPATH, "//label[contains(., 'End Report Date')]/following::input[1]"),
                ]
                if not _selenium_safe_fill(driver, end_date_selectors, end_date, "End Date", timeout=10):
                    _selenium_expand_report_parameters(driver)
                    if not _selenium_safe_fill(driver, end_date_selectors, end_date, "End Date", timeout=10):
                        raise ScraperError("Could not locate End Date input field.")

            execute_selectors = [
                (By.XPATH, "//button[contains(., 'Execute')]"),
            ]
            if not _selenium_safe_click(driver, execute_selectors, "Execute", timeout=10):
                raise ScraperError("Could not click Execute button.")

            _selenium_wait_for_apex_processing(driver)
            if wait_for_recaptcha and _selenium_detect_recaptcha(driver):
                if headless:
                    raise RecaptchaRequired("Recaptcha detected after execution while headless.")
                _selenium_wait_for_recaptcha_clear(
                    driver,
                    timeout_seconds=recaptcha_timeout,
                    poll_interval_seconds=recaptcha_poll_interval,
                )

            _selenium_wait_for_results(driver, timeout_seconds=30.0)

            actions_selectors = [
                (By.XPATH, "//button[contains(., 'Actions')]"),
            ]
            if not _selenium_safe_click(driver, actions_selectors, "Actions", timeout=10):
                raise ScraperError("Could not open Actions menu.")

            download_menu_selectors = [
                (By.XPATH, "//li[.//span[contains(., 'Download')]]"),
                (By.XPATH, "//a[contains(., 'Download')]"),
            ]
            if not _selenium_safe_click(driver, download_menu_selectors, "Download Menu", timeout=10):
                raise ScraperError("Could not open Download dialog.")
            _selenium_wait_for_download_dialog(driver, timeout_seconds=10.0)
            if not _selenium_select_csv_format(driver):
                logger.warning("CSV format option not found; using default selection.")

            download_button_selectors = [
                (By.XPATH, "//button[contains(., 'Download')]"),
                (By.CSS_SELECTOR, "button[id*='DOWNLOAD']"),
                (By.CSS_SELECTOR, "button.t-Button--hot"),
            ]
            if not _selenium_safe_click(driver, download_button_selectors, "Download Button", timeout=10):
                raise ScraperError("Could not trigger CSV download.")

            csv_path = _wait_for_download(download_dir, timeout_seconds=download_timeout)
            df = pd.read_csv(csv_path)
            logger.info("Successfully loaded DataFrame with %s rows.", len(df))
            return df
        finally:
            driver.quit()


def fetch_la_refinery_data_seleniumbase(
    start_date: str = "01-JAN-2018",
    end_date: Optional[str] = None,
    *,
    headless: bool = True,
    wait_for_recaptcha: bool = True,
    recaptcha_timeout: float = 25.0,
    recaptcha_poll_interval: float = 5.0,
    download_timeout: float = 10.0,
    page_ready_timeout: float = 5.0,
) -> pd.DataFrame:
    """Scrapes the LA DNR SONRIS Data Portal using SeleniumBase (uc=True)."""
    try:
        from seleniumbase import SB
        from selenium.webdriver.common.by import By
    except ImportError as exc:  # pragma: no cover - optional path
        raise ScraperError(
            "seleniumbase (and its dependencies) are required for uc-mode scraping."
        ) from exc

    df: Optional[pd.DataFrame] = None
    try:
        with SB(uc=True, test=True, headless=headless) as sb:
            sb.uc_open_with_reconnect(LA_REFINERY_URL, reconnect_time=2)
            driver = sb.driver

            begin_date_selectors = [
                (By.CSS_SELECTOR, "input[aria-label*='Begin Report Date']"),
                (By.CSS_SELECTOR, "input[id*='REPORT_DATE_BEGIN']"),
                (By.CSS_SELECTOR, "input[name*='REPORT_DATE_BEGIN']"),
                (By.CSS_SELECTOR, "input[id*='BEGIN_REPORT_DATE']"),
                (By.CSS_SELECTOR, "input[name*='BEGIN_REPORT_DATE']"),
                (By.XPATH, "//label[contains(., 'Begin Report Date')]/following::input[1]"),
                (By.CSS_SELECTOR, "input[placeholder*='DD-MON-YYYY']"),
            ]

            page_state = _selenium_wait_for_inputs_or_recaptcha(
                driver,
                begin_date_selectors,
                timeout_seconds=page_ready_timeout,
            )
            if page_state == "recaptcha":
                if headless:
                    raise RecaptchaRequired("Recaptcha detected while running headless.")
                if wait_for_recaptcha:
                    try:
                        sb.uc_gui_handle_captcha()
                    except Exception:
                        pass
                    _selenium_wait_for_recaptcha_clear(
                        driver,
                        timeout_seconds=recaptcha_timeout,
                        poll_interval_seconds=recaptcha_poll_interval,
                    )

            _selenium_expand_report_parameters(driver)
            if not _selenium_safe_fill(driver, begin_date_selectors, start_date, "Begin Date", timeout=10):
                _selenium_expand_report_parameters(driver)
                if not _selenium_safe_fill(driver, begin_date_selectors, start_date, "Begin Date", timeout=10):
                    raise ScraperError("Could not locate Begin Date input field.")

            if end_date:
                end_date_selectors = [
                    (By.CSS_SELECTOR, "input[aria-label*='End Report Date']"),
                    (By.CSS_SELECTOR, "input[id*='REPORT_DATE_END']"),
                    (By.CSS_SELECTOR, "input[name*='REPORT_DATE_END']"),
                    (By.CSS_SELECTOR, "input[id*='END_REPORT_DATE']"),
                    (By.CSS_SELECTOR, "input[name*='END_REPORT_DATE']"),
                    (By.XPATH, "//label[contains(., 'End Report Date')]/following::input[1]"),
                ]
                if not _selenium_safe_fill(driver, end_date_selectors, end_date, "End Date", timeout=10):
                    _selenium_expand_report_parameters(driver)
                    if not _selenium_safe_fill(driver, end_date_selectors, end_date, "End Date", timeout=10):
                        raise ScraperError("Could not locate End Date input field.")

            execute_selectors = [
                (By.XPATH, "//button[contains(., 'Execute')]"),
            ]
            if not _selenium_safe_click(driver, execute_selectors, "Execute", timeout=10):
                raise ScraperError("Could not click Execute button.")

            _selenium_wait_for_apex_processing(driver)
            if wait_for_recaptcha and _selenium_detect_recaptcha(driver):
                if headless:
                    raise RecaptchaRequired("Recaptcha detected after execution while headless.")
                _selenium_wait_for_recaptcha_clear(
                    driver,
                    timeout_seconds=recaptcha_timeout,
                    poll_interval_seconds=recaptcha_poll_interval,
                )

            _selenium_wait_for_results(driver, timeout_seconds=30.0)

            actions_selectors = [
                (By.XPATH, "//button[contains(., 'Actions')]"),
            ]
            if not _selenium_safe_click(driver, actions_selectors, "Actions", timeout=10):
                raise ScraperError("Could not open Actions menu.")

            download_menu_selectors = [
                (By.XPATH, "//li[.//span[contains(., 'Download')]]"),
                (By.XPATH, "//a[contains(., 'Download')]"),
            ]
            if not _selenium_safe_click(driver, download_menu_selectors, "Download Menu", timeout=10):
                raise ScraperError("Could not open Download dialog.")

            _selenium_wait_for_download_dialog(driver, timeout_seconds=10.0)
            if not _selenium_select_csv_format(driver):
                logger.warning("CSV format option not found; using default selection.")

            download_button_selectors = [
                (By.XPATH, "//button[contains(., 'Download')]"),
                (By.CSS_SELECTOR, "button[id*='DOWNLOAD']"),
                (By.CSS_SELECTOR, "button.t-Button--hot"),
            ]
            if not _selenium_safe_click(driver, download_button_selectors, "Download Button", timeout=10):
                raise ScraperError("Could not trigger CSV download.")

            downloads_dir = Path(sb.get_downloads_folder())
            for csv_file in downloads_dir.glob("*.csv"):
                try:
                    csv_file.unlink()
                except Exception:
                    continue

            csv_path = _wait_for_download(downloads_dir, timeout_seconds=download_timeout)
            df = pd.read_csv(csv_path)
            try:
                csv_path.unlink()
            except Exception:
                pass
            logger.info("Successfully loaded DataFrame with %s rows.", len(df))
            return df
    except Exception as exc:
        raise ScraperError(str(exc)) from exc

    if df is None:
        raise ScraperError("SeleniumBase did not return a dataset.")
    return df


async def wait_for_apex_processing(page: Page):
    """Specific handler for Oracle APEX 'Processing' overlays."""

    try:
        try:
            await page.wait_for_selector(".u-Processing", state="visible", timeout=2000)
        except PlaywrightTimeout:
            pass  # Spinner might not have appeared, which is fine

        await page.wait_for_selector(".u-Processing", state="detached", timeout=30000)
    except Exception:
        # Fallback to generic network idle
        await page.wait_for_load_state("networkidle")


def _iter_browser_launchers(playwright, *, headless: bool):
    """Yield browser launch callables in preferred order."""

    def chromium_channel(channel_name):
        return lambda: playwright.chromium.launch(headless=headless, channel=channel_name)

    yield chromium_channel("chrome")
    yield chromium_channel("msedge")
    yield lambda: playwright.webkit.launch(headless=headless)
    # Fallbacks if preferred channels are unavailable
    yield lambda: playwright.chromium.launch(headless=headless)
    yield lambda: playwright.webkit.launch(headless=headless)


async def _detect_recaptcha(page: Page) -> bool:
    """Best-effort detection of blocking recaptcha challenges."""

    try:
        for frame in page.frames:
            if "recaptcha/api2/bframe" in frame.url:
                return True
    except Exception:
        pass

    try:
        for frame in page.frames:
            if "recaptcha/api2/anchor" not in frame.url:
                continue
            try:
                anchor = frame.locator("#recaptcha-anchor")
                if await anchor.count() == 0:
                    return True
                state = await anchor.get_attribute("aria-checked")
                if state == "true":
                    return False
                return True
            except Exception:
                return True
    except Exception:
        pass

    selectors = [
        "div[role='dialog'] iframe[src*='recaptcha']",
        "text=/i'm not a robot/i",
        "text=/recaptcha/i",
    ]
    for selector in selectors:
        try:
            locator = page.locator(selector)
            if await locator.count() > 0 and await locator.first.is_visible():
                return True
        except Exception:
            continue
    return False


async def _try_expand_report_parameters(page: Page) -> None:
    selectors = [
        "button:has-text('Report Parameters')",
        ".t-Region-header:has-text('Report Parameters') button",
        ".t-Region-title:has-text('Report Parameters') button",
        ".t-Region-title:has-text('Report Parameters')",
        "[aria-controls*='REPORT_PARAMETERS']",
        "[aria-controls*='report_parameters']",
    ]
    for selector in selectors:
        try:
            locator = page.locator(selector).first
            if await locator.count() == 0:
                continue
            expanded = await locator.get_attribute("aria-expanded")
            if expanded == "false":
                await locator.click()
            return
        except Exception:
            continue


async def _wait_for_input_ready(
    page: Page,
    selectors: List[str],
    *,
    timeout_seconds: float,
    poll_interval_seconds: float,
) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        for selector in selectors:
            try:
                locator = page.locator(selector).first
                if await locator.count() > 0 and await locator.is_visible():
                    return True
            except Exception:
                continue
        await page.wait_for_timeout(int(poll_interval_seconds * 1000))
    return False


async def _wait_for_recaptcha_clear(
    page: Page,
    *,
    timeout_seconds: float,
    poll_interval_seconds: float,
    bring_to_front: bool = False,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if bring_to_front:
            try:
                await page.bring_to_front()
            except Exception:
                pass
        if not await _detect_recaptcha(page):
            return
        await page.wait_for_timeout(int(poll_interval_seconds * 1000))
    raise ScraperError("Recaptcha still present after waiting for manual completion.")


async def _wait_for_results_ready(
    page: Page,
    *,
    timeout_seconds: float,
    poll_interval_seconds: float,
) -> bool:
    """Wait for results rows (or no-data message) to render after execution."""

    row_selectors = [
        ".a-IRR-table tbody tr",
        "table.a-IRR-table tbody tr",
        "table[role='grid'] tbody tr",
        "#report_data tbody tr",
    ]
    no_data_selectors = [
        ".a-IRR-noDataMsg",
        "text=/no data found/i",
    ]
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        for selector in row_selectors:
            try:
                locator = page.locator(selector)
                if await locator.count() > 0:
                    return True
            except Exception:
                continue
        for selector in no_data_selectors:
            try:
                locator = page.locator(selector)
                if await locator.count() > 0 and await locator.first.is_visible():
                    return True
            except Exception:
                continue
        await page.wait_for_timeout(int(poll_interval_seconds * 1000))
    return False


async def _handle_recaptcha(
    page: Page,
    *,
    headless: bool,
    wait_for_recaptcha: bool,
    recaptcha_timeout: float,
    recaptcha_poll_interval: float,
) -> None:
    if not wait_for_recaptcha:
        return
    if not await _detect_recaptcha(page):
        return
    if headless:
        raise RecaptchaRequired(
            "Recaptcha detected while running headless; rerun with headless=False."
        )

    logger.warning(
        "Recaptcha detected. Waiting up to %.0f seconds for manual completion.",
        recaptcha_timeout,
    )
    try:
        await page.bring_to_front()
    except Exception:
        pass
    await _wait_for_recaptcha_clear(
        page,
        timeout_seconds=recaptcha_timeout,
        poll_interval_seconds=recaptcha_poll_interval,
        bring_to_front=True,
    )
    try:
        await page.bring_to_front()
    except Exception:
        pass


def _install_playwright_browsers() -> bool:
    """Install Playwright-managed browsers if they are missing.

    Returns:
        bool: True if an install was attempted in this process, False otherwise.
    """

    global _PLAYWRIGHT_INSTALL_ATTEMPTED
    if _PLAYWRIGHT_INSTALL_ATTEMPTED:
        return False

    _PLAYWRIGHT_INSTALL_ATTEMPTED = True
    try:
        import sys
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "playwright",
                "install",
                "chromium",
                "msedge",
                "webkit",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            logger.info("Playwright browsers are installed or already present.")
        else:  # pragma: no cover - dependent on environment
            logger.warning(
                "Playwright browser installation returned non-zero exit %s: %s",
                result.returncode,
                (result.stderr or result.stdout).strip(),
            )
    except FileNotFoundError:  # pragma: no cover - environment specific
        logger.error("Playwright CLI is unavailable; cannot install browsers automatically.")
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Unexpected error while installing Playwright browsers: %s", exc)

    return True


async def fetch_la_refinery_data(
    start_date: str = "01-JAN-2018",
    end_date: Optional[str] = None,
    *,
    headless: bool = True,
    wait_for_recaptcha: bool = True,
    recaptcha_timeout: float = 25.0,
    recaptcha_poll_interval: float = 0.25,
    install_browsers: bool = False,
) -> pd.DataFrame:
    """
    Scrapes the LA DNR SONRIS Data Portal for Refinery Activity Reports.

    Args:
        start_date: Date string in 'DD-MON-YYYY' format (e.g., '01-JAN-2018').
        end_date: Optional end date string in 'DD-MON-YYYY' format.
        headless: Whether to run the browser in headless mode.
        wait_for_recaptcha: If True, wait for manual recaptcha completion when detected.
        recaptcha_timeout: Seconds to wait for recaptcha completion before failing.
        recaptcha_poll_interval: Seconds between recaptcha status checks.
        install_browsers: If True, attempt to install Playwright browsers automatically.

    Returns:
        pd.DataFrame: The resulting dataset.

    Raises:
        ScraperError: If the scraping process fails at any critical step.
    """

    url = LA_REFINERY_URL

    # Ensure Playwright-managed browser binaries are present before attempting to launch
    if install_browsers:
        _install_playwright_browsers()

    async def _run_scrape(*, headless_mode: bool) -> pd.DataFrame:
        # Create a temporary directory for the download to ensure thread safety and cleanliness
        with tempfile.TemporaryDirectory() as temp_dir:
            async with async_playwright() as p:
                browser = None
                context = None
                # Attempt preferred browsers in order: Chrome, Edge, Safari/WebKit, Chromium fallback
                for launcher in _iter_browser_launchers(p, headless=headless_mode):
                    try:
                        browser = await launcher()
                        break
                    except Exception as exc:  # pragma: no cover - dependent on environment
                        logger.info("Browser launch failed, trying next option: %s", exc)
                        continue

                if browser is None:
                    raise ScraperError(
                        "Unable to launch any supported browser (Chrome, Edge, Safari/WebKit)."
                    )

                context = await browser.new_context(
                    accept_downloads=True,
                    viewport={"width": 1920, "height": 1080},
                )
                page = await context.new_page()

                try:
                    logger.info("Navigating to %s", url)
                    await page.goto(url, timeout=60000)
                    await _handle_recaptcha(
                        page,
                        headless=headless_mode,
                        wait_for_recaptcha=wait_for_recaptcha,
                        recaptcha_timeout=recaptcha_timeout,
                        recaptcha_poll_interval=recaptcha_poll_interval,
                    )
                    await page.wait_for_load_state("domcontentloaded")
                    await _try_expand_report_parameters(page)

                    # 1. Set Dates
                    logger.info("Setting report begin date to %s", start_date)
                    date_selectors = [
                        "input[aria-label*='Begin Report Date']",
                        "input[id*='BEGIN_REPORT_DATE']",
                        "input[name*='BEGIN_REPORT_DATE']",
                        "label:has-text('Begin Report Date') >> .. >> input",
                        "label:has-text('Begin Report Date') >> xpath=following::input[1]",
                        "label:has-text('Begin Report Date') + input",
                        "label:has-text('Begin Report Date') ~ input",
                        "input[placeholder*='DD-MON-YYYY']",
                    ]
                    if not await safe_fill(
                        page,
                        date_selectors,
                        start_date,
                        "Begin Date Input",
                        timeout_ms=500,
                    ):
                        await _try_expand_report_parameters(page)
                        if not await _wait_for_input_ready(
                            page,
                            date_selectors,
                            timeout_seconds=20.0,
                            poll_interval_seconds=_INPUT_POLL_INTERVAL,
                        ):
                            raise ScraperError("Could not locate Begin Date input field.")
                        if not await safe_fill(page, date_selectors, start_date, "Begin Date Input"):
                            raise ScraperError("Could not populate Begin Date input field.")

                    if end_date:
                        logger.info("Setting report end date to %s", end_date)
                        end_date_selectors = [
                            "input[aria-label*='End Report Date']",
                            "input[id*='END_REPORT_DATE']",
                            "input[name*='END_REPORT_DATE']",
                            "label:has-text('End Report Date') >> .. >> input",
                            "label:has-text('End Report Date') >> xpath=following::input[1]",
                            "label:has-text('End Report Date') + input",
                            "label:has-text('End Report Date') ~ input",
                        ]
                        if not await safe_fill(
                            page,
                            end_date_selectors,
                            end_date,
                            "End Date Input",
                            timeout_ms=500,
                        ):
                            await _try_expand_report_parameters(page)
                            if not await _wait_for_input_ready(
                                page,
                                end_date_selectors,
                                timeout_seconds=15.0,
                                poll_interval_seconds=_INPUT_POLL_INTERVAL,
                            ):
                                raise ScraperError("Could not locate End Date input field.")
                            if not await safe_fill(
                                page, end_date_selectors, end_date, "End Date Input"
                            ):
                                raise ScraperError("Could not populate End Date input field.")

                    # 2. Execute Report
                    logger.info("Executing report generation")
                    execute_selectors = [
                        "button:has-text('Execute')",
                        ".t-Button--hot",
                        "[id*='EXECUTE']",
                    ]
                    if not await safe_click(page, execute_selectors, "Execute Button"):
                        raise ScraperError("Could not click Execute button.")

                    # 3. Wait for Grid Refresh
                    await wait_for_apex_processing(page)
                    await _handle_recaptcha(
                        page,
                        headless=headless_mode,
                        wait_for_recaptcha=wait_for_recaptcha,
                        recaptcha_timeout=recaptcha_timeout,
                        recaptcha_poll_interval=recaptcha_poll_interval,
                    )
                    # Wait for table rows to be ready (or no-data message)
                    if not await _wait_for_results_ready(
                        page,
                        timeout_seconds=30.0,
                        poll_interval_seconds=0.5,
                    ):
                        await page.wait_for_timeout(1500)

                    # 4. Open Actions Menu
                    logger.info("Opening Actions menu")
                    await _handle_recaptcha(
                        page,
                        headless=headless_mode,
                        wait_for_recaptcha=wait_for_recaptcha,
                        recaptcha_timeout=recaptcha_timeout,
                        recaptcha_poll_interval=recaptcha_poll_interval,
                    )
                    actions_selectors = [
                        "button:has-text('Actions')",
                        ".a-IRR-actions-button",
                        "[id$='_actions_button']",
                    ]
                    if not await safe_click(page, actions_selectors, "Actions Button"):
                        raise ScraperError("Could not open Actions menu.")

                    # 5. Select Download
                    logger.info("Selecting Download option")
                    await _handle_recaptcha(
                        page,
                        headless=headless_mode,
                        wait_for_recaptcha=wait_for_recaptcha,
                        recaptcha_timeout=recaptcha_timeout,
                        recaptcha_poll_interval=recaptcha_poll_interval,
                    )
                    download_menu_selectors = [
                        "menuitem:has-text('Download')",
                        "div.a-Menu-content >> text='Download'",
                    ]
                    if not await safe_click(page, download_menu_selectors, "Download Menu Item"):
                        raise ScraperError("Could not find Download option in menu.")

                    # 6. Trigger CSV Download
                    logger.info("Initiating CSV download")
                    await _handle_recaptcha(
                        page,
                        headless=headless_mode,
                        wait_for_recaptcha=wait_for_recaptcha,
                        recaptcha_timeout=recaptcha_timeout,
                        recaptcha_poll_interval=recaptcha_poll_interval,
                    )
                    csv_selectors = [
                        "label:has-text('CSV')",
                        "input[type='radio'][value*='CSV']",
                        ".a-IRR-dialog-content >> text='CSV'",
                    ]
                    await safe_click(page, csv_selectors, "CSV Format Selection")

                    async with page.expect_download() as download_info:
                        download_button_selectors = [
                            "button:has-text('Download')",
                            "button[id*='DOWNLOAD']",
                            ".ui-dialog button:has-text('Download')",
                        ]
                        if not await safe_click(
                            page,
                            download_button_selectors,
                            "Download Button",
                        ):
                            raise ScraperError("Could not click Download button.")

                    download = await download_info.value
                    target_path = Path(temp_dir) / "data.csv"
                    await download.save_as(target_path)

                    logger.info("File downloaded successfully to %s", target_path)

                    # 7. Parse Data
                    if target_path.stat().st_size == 0:
                        raise ScraperError("Downloaded CSV file is empty.")

                    df = pd.read_csv(target_path)
                    logger.info("Successfully loaded DataFrame with %s rows.", len(df))

                    return df

                except Exception as exc:
                    logger.error("Scraping failed: %s", exc)
                    # Take screenshot on failure for debugging (optional, saves to local dir)
                    try:
                        await page.screenshot(path="error_screenshot.png")
                        logger.info("Error screenshot saved to 'error_screenshot.png'")
                    except Exception:
                        pass
                    raise
                finally:
                    if context is not None:
                        await context.close()
                    if browser is not None:
                        await browser.close()

    try:
        return await _run_scrape(headless_mode=headless)
    except RecaptchaRequired:
        if headless and wait_for_recaptcha:
            logger.warning("Recaptcha detected; rerunning with headless=False for manual solve.")
            return await _run_scrape(headless_mode=False)
        raise


# Wrapper to run the async function from synchronous code if needed
def LA_refinery(
    start_date: str = "01-JAN-2018",
    end_date: Optional[str] = None,
    *,
    engine: str = "selenium",
    headless: bool = True,
    wait_for_recaptcha: bool = True,
    recaptcha_timeout: float = 25.0,
    recaptcha_poll_interval: float = 5.0,
    install_browsers: bool = False,
    download_timeout: float = 10.0,
    use_stealth: bool = True,
    page_ready_timeout: float = 5.0,
    use_cache_if_available: bool = True,
    update_cache: bool = False,
    cache_csv_path: Optional[str | Path] = None,
    cache_pkl_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    if use_cache_if_available and not update_cache:
        cached = load_la_refinery_cache(cache_csv_path, cache_pkl_path)
        if cached is not None and not cached.empty:
            return cached

    try:
        if engine == "seleniumbase":
            try:
                df = fetch_la_refinery_data_seleniumbase(
                    start_date,
                    end_date,
                    headless=headless,
                    wait_for_recaptcha=wait_for_recaptcha,
                    recaptcha_timeout=recaptcha_timeout,
                    recaptcha_poll_interval=recaptcha_poll_interval,
                    download_timeout=download_timeout,
                    page_ready_timeout=page_ready_timeout,
                )
            except Exception as exc:
                logger.warning("SeleniumBase failed; falling back to Playwright. Error: %s", exc)
                df = asyncio.run(
                    fetch_la_refinery_data(
                        start_date,
                        end_date,
                        headless=headless,
                        wait_for_recaptcha=wait_for_recaptcha,
                        recaptcha_timeout=recaptcha_timeout,
                        recaptcha_poll_interval=recaptcha_poll_interval,
                        install_browsers=install_browsers,
                    )
                )
        elif engine == "selenium":
            try:
                df = fetch_la_refinery_data_selenium(
                    start_date,
                    end_date,
                    headless=headless,
                    wait_for_recaptcha=wait_for_recaptcha,
                recaptcha_timeout=recaptcha_timeout,
                recaptcha_poll_interval=recaptcha_poll_interval,
                download_timeout=download_timeout,
                    use_stealth=use_stealth,
                    page_ready_timeout=page_ready_timeout,
                )
            except RecaptchaRequired:
                if headless and wait_for_recaptcha:
                    logger.warning(
                        "Recaptcha detected; rerunning Selenium with headless=False for manual solve."
                    )
                    df = fetch_la_refinery_data_selenium(
                        start_date,
                        end_date,
                        headless=False,
                        wait_for_recaptcha=wait_for_recaptcha,
                        recaptcha_timeout=recaptcha_timeout,
                        recaptcha_poll_interval=recaptcha_poll_interval,
                        download_timeout=download_timeout,
                        use_stealth=use_stealth,
                        page_ready_timeout=page_ready_timeout,
                    )
                else:
                    raise
        elif engine == "playwright":
            df = asyncio.run(
                fetch_la_refinery_data(
                    start_date,
                    end_date,
                    headless=headless,
                    wait_for_recaptcha=wait_for_recaptcha,
                    recaptcha_timeout=recaptcha_timeout,
                    recaptcha_poll_interval=recaptcha_poll_interval,
                    install_browsers=install_browsers,
                )
            )
        else:
            raise ScraperError(f"Unsupported engine: {engine}")
    except Exception as exc:
        logger.critical("Critical failure in LA_refinery: %s", exc)
        raise

    df = apply_la_refinery_name_map_to_df(df)

    if update_cache:
        save_la_refinery_cache(df, cache_csv_path, cache_pkl_path)

    return df


def update_la_refinery_cache(
    start_date: str = "01-JAN-2018",
    end_date: Optional[str] = None,
    *,
    engine: str = "selenium",
    headless: bool = True,
    wait_for_recaptcha: bool = True,
    recaptcha_timeout: float = 25.0,
    recaptcha_poll_interval: float = 5.0,
    install_browsers: bool = False,
    download_timeout: float = 10.0,
    use_stealth: bool = True,
    page_ready_timeout: float = 5.0,
    cache_csv_path: Optional[str | Path] = None,
    cache_pkl_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    """Refresh the packaged LA refinery CSV/PKL snapshot."""
    df = LA_refinery(
        start_date,
        end_date,
        engine=engine,
        headless=headless,
        wait_for_recaptcha=wait_for_recaptcha,
        recaptcha_timeout=recaptcha_timeout,
        recaptcha_poll_interval=recaptcha_poll_interval,
        install_browsers=install_browsers,
        download_timeout=download_timeout,
        use_stealth=use_stealth,
        page_ready_timeout=page_ready_timeout,
        use_cache_if_available=False,
        update_cache=False,
        cache_csv_path=cache_csv_path,
        cache_pkl_path=cache_pkl_path,
    )
    save_la_refinery_cache(df, cache_csv_path, cache_pkl_path)
    return df


__all__ = [
    "LA_refinery",
    "ScraperError",
    "fetch_la_refinery_data",
    "fetch_la_refinery_data_selenium",
    "fetch_la_refinery_data_seleniumbase",
    "load_la_refinery_cache",
    "save_la_refinery_cache",
    "update_la_refinery_cache",
]
