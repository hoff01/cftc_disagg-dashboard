"""
High-throughput helpers for ingesting API responses into pandas DataFrames.

The :func:`fetch_apis_to_dataframe` function is designed for bulk data
collection scenarios where many REST endpoints must be queried as quickly as
possible.  It combines several performance techniques:

* HTTP/2 with pooled keep-alive connections via ``httpx`` for higher throughput.
* Async event loop + bounded concurrency so endpoints are fetched in parallel
  without overwhelming the remote service.
* Aggressive compression support (``gzip``/``deflate``/``br``) by default to
  minimize wire bytes.
* Automatic pagination-like batching when callers supply multiple endpoints in a
  list; the results are concatenated into a single DataFrame.
* Minimal copying when normalizing JSON payloads into tidy tabular structures.

The function accepts either bare URL strings or dictionaries with per-endpoint
headers/parameters, making it flexible enough to handle heterogeneous APIs in a
single call.  Optionally, the combined DataFrame can be written to a user
specified directory with a custom filename for downstream automation.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import Future
from inspect import iscoroutine
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, MutableMapping, Optional, Sequence, Union
import io
import json
import random
import threading

import httpx
import pandas as pd

Endpoint = Union[str, Mapping[str, Any]]


def _normalize_endpoint(endpoint: Endpoint) -> Dict[str, Any]:
    if isinstance(endpoint, str):
        return {"url": endpoint, "params": {}, "headers": {}}
    if isinstance(endpoint, Mapping):
        if "url" not in endpoint:
            raise ValueError("Endpoint mappings must include a 'url' key")
        return {
            "url": str(endpoint["url"]),
            "params": dict(endpoint.get("params", {})),
            "headers": dict(endpoint.get("headers", {})),
        }
    raise TypeError("Endpoints must be strings or mappings with a 'url' field")


def _unwrap_payload(payload: Any) -> Any:
    if isinstance(payload, Mapping):
        for key in ("data", "results", "items", "records"):  # common API wrappers
            if key in payload and isinstance(payload[key], (list, Mapping)):
                return payload[key]
    return payload


def _payload_to_frame(payload: Any) -> pd.DataFrame:
    unwrapped = _unwrap_payload(payload)
    if isinstance(unwrapped, pd.DataFrame):
        return unwrapped.copy(deep=False)
    if isinstance(unwrapped, list):
        return pd.json_normalize(unwrapped, sep=".")
    if isinstance(unwrapped, Mapping):
        return pd.json_normalize(unwrapped, sep=".")
    raise ValueError("Unsupported payload type for DataFrame conversion")


def _ndjson_bytes_to_frame(raw: bytes) -> pd.DataFrame:
    records = [json.loads(line) for line in raw.splitlines() if line.strip()]
    return pd.json_normalize(records, sep=".")


def _detect_and_parse_response(response: Any) -> pd.DataFrame:
    content_type = response.headers.get("Content-Type", "").lower()
    if "ndjson" in content_type:
        return _ndjson_bytes_to_frame(response.content)
    if "json" in content_type or content_type.endswith("/javascript"):
        return _payload_to_frame(response.json())
    if "csv" in content_type or "text/plain" in content_type:
        return pd.read_csv(io.BytesIO(response.content))

    # Fallbacks: try JSON first, then CSV parsing from bytes
    try:
        return _payload_to_frame(response.json())
    except Exception:
        try:
            # NDJSON served with plain text headers
            return _ndjson_bytes_to_frame(response.content)
        except Exception:
            return pd.read_csv(io.BytesIO(response.content))


class _AdaptiveThrottle:
    """Shared throttle that injects small sleeps after backpressure signals.

    The throttle is intentionally conservative: it only grows when a request
    times out or receives a 429/5xx response, and it decays on successful
    responses so the client can ramp back up as conditions improve.
    """

    def __init__(self, ceiling: float) -> None:
        self._delay = 0.0
        self._ceiling = ceiling
        self._lock = asyncio.Lock()

    async def wait(self) -> None:
        delay = self._delay
        if delay <= 0:
            return
        jitter = delay * (0.1 + random.random() * 0.2)  # 10-30% jitter to stagger
        await asyncio.sleep(delay + jitter)

    async def increase(self, hinted: Optional[float] = None) -> None:
        async with self._lock:
            baseline = max(self._delay * 1.5, hinted or 0.08)
            self._delay = min(self._ceiling, baseline)

    async def relax(self) -> None:
        async with self._lock:
            self._delay *= 0.6
            if self._delay < 0.02:
                self._delay = 0.0


def fetch_apis_to_dataframe(
    endpoints: Sequence[Endpoint],
    *,
    common_headers: Optional[Mapping[str, str]] = None,
    common_params: Optional[Mapping[str, Any]] = None,
    timeout: float = 15.0,
    max_workers: int = 8,
    retries: int = 3,
    output_dir: Union[str, Path, None] = None,
    file_name: Optional[str] = None,
    deduplicate: bool = True,
    sort_by: Union[str, Sequence[str], None] = None,
    transform: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
    run_in_background: bool = False,
) -> Union[pd.DataFrame, Future[pd.DataFrame]]:
    """Fetch multiple API endpoints in parallel and return a combined DataFrame.

    Parameters
    ----------
    endpoints : sequence of str or mapping
        Each element can be a URL string or a mapping containing ``url`` along
        with optional ``params`` and ``headers`` entries.  All endpoints are
        fetched concurrently.
    common_headers : mapping, optional
        Headers applied to every request; endpoint-specific headers override
        duplicates.
    common_params : mapping, optional
        Query parameters applied to every request; endpoint-specific parameters
        override duplicates.
    timeout : float, default 15.0
        Initial per-request timeout in seconds. Requests that hit this limit
        automatically grow their timeout budget on subsequent retries to avoid
        slow endpoints failing permanently.
    max_workers : int, default 8
        Maximum number of concurrent requests.
    retries : int, default 3
        Number of retry attempts for transient HTTP errors (5xx/429).
    output_dir : str or pathlib.Path, optional
        Directory where the combined CSV should be written.  When omitted, the
        DataFrame is returned without writing to disk.
    file_name : str, optional
        Custom filename (without extension) for the CSV.  Defaults to a
        timestamped name when ``output_dir`` is provided.
    deduplicate : bool, default True
        Drop duplicate rows after concatenation.
    sort_by : str or sequence of str, optional
        Column(s) to sort the final DataFrame by before returning/saving.
    transform : callable, optional
        A function applied to the combined DataFrame before deduplication and
        sorting. It must accept and return a ``pandas.DataFrame``. Useful for
        lightweight post-processing such as adding derived columns or filtering
        rows without requiring a second pass by the caller.
    run_in_background : bool, default False
        When True, the function immediately returns a ``concurrent.futures.Future``
        while the fetch runs in a background thread. ``future.result()`` will yield
        the same DataFrame as the synchronous path, enabling non-blocking usage in
        notebooks or services that must keep their event loop responsive.

    Returns
    -------
    pandas.DataFrame or concurrent.futures.Future
        Concatenated data from all endpoints. When ``run_in_background`` is set,
        a ``Future`` that yields the DataFrame is returned instead.

    Examples
    --------
    >>> endpoints = [
    ...     "https://api.example.com/v1/events",
    ...     {"url": "https://api.example.com/v1/users", "params": {"page": 1}},
    ... ]
    >>> df = fetch_apis_to_dataframe(
    ...     endpoints,
    ...     max_workers=16,
    ...     output_dir="./exports",
    ...     file_name="daily_snapshot",
    ... )
    >>> df.head()
    """

    if not endpoints:
        raise ValueError("Provide at least one endpoint to fetch")

    normalized = [_normalize_endpoint(ep) for ep in endpoints]
    base_headers: MutableMapping[str, str] = {
        "Accept": "application/json, text/csv;q=0.9, */*;q=0.8",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
    }
    if common_headers:
        base_headers.update(common_headers)
    base_params: MutableMapping[str, Any] = dict(common_params or {})

    def _build_httpx_client() -> httpx.AsyncClient:
        transport = httpx.AsyncHTTPTransport(
            http2=True,
            retries=retries,
            limits=httpx.Limits(
                max_keepalive_connections=max(max_workers * 2, 16),
                max_connections=max(max_workers * 4, 32),
            ),
        )
        return httpx.AsyncClient(
            headers=base_headers,
            transport=transport,
            timeout=httpx.Timeout(timeout),
        )

    semaphore = asyncio.Semaphore(max_workers)

    def _next_timeout(previous: float) -> float:
        """Grow timeout with jitter to adapt to slow endpoints without hanging."""

        growth = previous * 1.6  # gentle ramp-up to avoid long stalls
        jitter = growth * (0.05 + random.random() * 0.1)  # 5-15% jitter
        return min(timeout * 8, growth + jitter)

    throttle = _AdaptiveThrottle(timeout)

    async def fetch_one(entry: Dict[str, Any], client: httpx.AsyncClient) -> pd.DataFrame:
        headers = {**client.headers, **entry.get("headers", {})}
        params = {**base_params, **entry.get("params", {})}
        attempt_timeout = timeout
        for attempt in range(retries + 1):
            try:
                await throttle.wait()
                async with semaphore:
                    response = await client.get(
                        entry["url"],
                        params=params,
                        headers=headers,
                        timeout=attempt_timeout,
                    )
                response.raise_for_status()
                await throttle.relax()
                return _detect_and_parse_response(response)
            except httpx.TimeoutException:
                await throttle.increase()
                if attempt == retries:
                    raise
                attempt_timeout = _next_timeout(attempt_timeout)
                await asyncio.sleep(0.1)
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code in {429, 500, 502, 503, 504} and attempt < retries:
                    hinted_retry = None
                    retry_after = exc.response.headers.get("Retry-After")
                    if retry_after:
                        try:
                            hinted_retry = float(retry_after)
                        except ValueError:
                            hinted_retry = None
                    await throttle.increase(hinted_retry)
                    attempt_timeout = _next_timeout(attempt_timeout)
                    await asyncio.sleep(0.2 * (attempt + 1))
                    continue
                raise
            except httpx.TransportError:
                await throttle.increase()
                if attempt == retries:
                    raise
                attempt_timeout = _next_timeout(attempt_timeout)
                await asyncio.sleep(0.2 * (attempt + 1))

    async def run() -> list[pd.DataFrame]:
        async with _build_httpx_client() as client:
            tasks = [fetch_one(entry, client) for entry in normalized]
            return await asyncio.gather(*tasks)

    def _run_coro(coro: Any) -> list[pd.DataFrame]:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)

        if loop.is_running():
            new_loop = asyncio.new_event_loop()
            try:
                return new_loop.run_until_complete(coro)
            finally:
                new_loop.close()

        return loop.run_until_complete(coro)

    def _execute() -> pd.DataFrame:
        frames = _run_coro(run())
        combined = pd.concat(frames, axis=0, ignore_index=True)

        if transform is not None:
            try:
                transformed = transform(combined)
                if iscoroutine(transformed):
                    transformed = _run_coro(transformed)
            except Exception as exc:  # pragma: no cover - defensive context
                raise RuntimeError("transform callback failed") from exc

            if not isinstance(transformed, pd.DataFrame):
                raise TypeError(
                    "transform must return a pandas DataFrame; got "
                    f"{type(transformed).__name__}"
                )

            combined = transformed
        if deduplicate:
            combined = combined.drop_duplicates()
        if sort_by:
            combined = combined.sort_values(sort_by).reset_index(drop=True)

        if output_dir:
            out_dir = Path(output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            final_name = file_name or f"api_export_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            output_path = out_dir / f"{final_name}.csv"
            combined.to_csv(output_path, index=False)

        return combined

    if run_in_background:
        result: Future[pd.DataFrame] = Future()

        def _worker():
            try:
                result.set_result(_execute())
            except BaseException as exc:  # pragma: no cover - defensive context
                result.set_exception(exc)

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()
        return result

    return _execute()

__all__ = ["fetch_apis_to_dataframe"]
