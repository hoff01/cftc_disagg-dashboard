"""Client wrapper for the DTN Refined Fuels USMD API.

This module provides a small, dependency‑light helper to download
refined fuel demand datasets into :class:`pandas.DataFrame` objects.
All query parameters mirror the upstream API; they default to ``None``
so callers can either pull the full dataset or request filtered slices
by specifying the desired arguments.

Example
-------
.. code-block:: python

    from analysis3054.refined_fuels_api import RefinedFuelsUSMDClient

    client = RefinedFuelsUSMDClient(api_key="<your api key>")
    df = client.fetch_padd_daily(regions="1,2,3", startDate="2024-01-01")
    print(df.head())

If you prefer to use OAuth instead of an API key, you can obtain an
access token via :func:`request_access_token` and pass it to the client
using the ``access_token`` argument.

The client automatically paginates responses using the maximum allowed
page size for efficiency.  It raises :class:`requests.HTTPError` when
the API responds with an error status.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd
import requests

DEFAULT_BASE_URL = "https://traderinsights-cat.dtn.com/repo_usmd_api"
TOKEN_URL = "https://api.auth.dtn.com/v1/tokens/authorize"
DEFAULT_AUDIENCE = DEFAULT_BASE_URL
MAX_PAGE_SIZE = 5000


def request_access_token(
    *,
    client_id: str,
    client_secret: str,
    audience: str = DEFAULT_AUDIENCE,
    timeout: float = 10.0,
) -> str:
    """Generate an OAuth access token via the DTN DAIS endpoint.

    Parameters
    ----------
    client_id : str
        The client identifier provided by DTN.
    client_secret : str
        Secret associated with the client ID.
    audience : str, optional
        Audience for the requested token.  Defaults to the production
        Refined Fuels USMD API audience.
    timeout : float, optional
        Request timeout in seconds.

    Returns
    -------
    str
        The access token string to be used as a Bearer token.

    Raises
    ------
    requests.HTTPError
        If the token endpoint responds with an error status.
    ValueError
        If a token cannot be located in the response payload.
    """

    payload = {
        "grant_type": "client_credentials",
        "client_id": client_id,
        "client_secret": client_secret,
        "audience": audience,
    }
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    response = requests.post(TOKEN_URL, json=payload, headers=headers, timeout=timeout)
    response.raise_for_status()
    body = response.json()
    token = body.get("data", {}).get("access_token") or body.get("access_token")
    if not token:
        raise ValueError("Access token not found in token response")
    return token


class RefinedFuelsUSMDClient:
    """Lightweight wrapper to download USMD datasets into DataFrames.

    Parameters
    ----------
    api_key : str, optional
        API key to send via the ``apiKey`` header.
    access_token : str, optional
        OAuth access token to send as a Bearer token.  Either ``api_key``
        or ``access_token`` must be provided.
    base_url : str, optional
        Base URL for the USMD API.  Defaults to the production URL.
    timeout : float, optional
        Timeout (seconds) applied to each HTTP request.
    session : requests.Session, optional
        Reusable HTTP session.  A new session will be created when not
        provided.
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        access_token: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 10.0,
        session: Optional[requests.Session] = None,
    ) -> None:
        if not api_key and not access_token:
            raise ValueError("Provide either an api_key or access_token")
        self.api_key = api_key
        self.access_token = access_token
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = session or requests.Session()

    def fetch_padd_daily(
        self,
        *,
        regions: Optional[str] = None,
        products: Optional[str] = None,
        grades: Optional[str] = None,
        pageSize: Optional[int] = None,
        page: Optional[int] = None,
        startDate: Optional[str] = None,
        endDate: Optional[str] = None,
        format: Optional[str] = None,
    ) -> pd.DataFrame:
        """Retrieve daily PADD summaries as a DataFrame."""

        data = self._paginate(
            "/padd-daily",
            params={
                "regions": regions,
                "products": products,
                "grades": grades,
                "pageSize": pageSize,
                "page": page,
                "startDate": startDate,
                "endDate": endDate,
                "format": format,
            },
        )
        return pd.DataFrame(data)

    def fetch_rack_daily(
        self,
        *,
        regions: Optional[str] = None,
        products: Optional[str] = None,
        grades: Optional[str] = None,
        rackAverage: Optional[str] = None,
        states: Optional[str] = None,
        pageSize: Optional[int] = None,
        page: Optional[int] = None,
        startDate: Optional[str] = None,
        endDate: Optional[str] = None,
        format: Optional[str] = None,
    ) -> pd.DataFrame:
        """Retrieve rack-level daily summaries as a DataFrame."""

        data = self._paginate(
            "/rack-daily",
            params={
                "regions": regions,
                "products": products,
                "grades": grades,
                "rackAverage": rackAverage,
                "states": states,
                "pageSize": pageSize,
                "page": page,
                "startDate": startDate,
                "endDate": endDate,
                "format": format,
            },
        )
        return pd.DataFrame(data)

    def fetch_supplier_terminal_monthly(
        self,
        *,
        regions: Optional[str] = None,
        products: Optional[str] = None,
        grades: Optional[str] = None,
        rackNames: Optional[str] = None,
        states: Optional[str] = None,
        supplierMinCount: Optional[int] = None,
        terminalMinCount: Optional[int] = None,
        pageSize: Optional[int] = None,
        page: Optional[int] = None,
        startDate: Optional[str] = None,
        endDate: Optional[str] = None,
        format: Optional[str] = None,
    ) -> pd.DataFrame:
        """Retrieve supplier/terminal monthly counts as a DataFrame."""

        data = self._paginate(
            "/supplier-terminal-monthly",
            params={
                "regions": regions,
                "products": products,
                "grades": grades,
                "rackNames": rackNames,
                "states": states,
                "supplierMinCount": supplierMinCount,
                "terminalMinCount": terminalMinCount,
                "pageSize": pageSize,
                "page": page,
                "startDate": startDate,
                "endDate": endDate,
                "format": format,
            },
        )
        return pd.DataFrame(data)

    def fetch_coverage_daily(
        self,
        *,
        regions: Optional[str] = None,
        pageSize: Optional[int] = None,
        page: Optional[int] = None,
        startDate: Optional[str] = None,
        endDate: Optional[str] = None,
        format: Optional[str] = None,
    ) -> pd.DataFrame:
        """Retrieve daily coverage factors as a DataFrame."""

        data = self._paginate(
            "/coverage-daily",
            params={
                "regions": regions,
                "pageSize": pageSize,
                "page": page,
                "startDate": startDate,
                "endDate": endDate,
                "format": format,
            },
        )
        return pd.DataFrame(data)

    def _headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {}
        if self.access_token:
            headers["Authorization"] = f"Bearer {self.access_token}"
        if self.api_key:
            headers["apiKey"] = self.api_key
        return headers

    def _paginate(self, path: str, params: Dict[str, Optional[object]]) -> List[dict]:
        """Iterate through all pages for a given endpoint."""

        clean_params = {k: v for k, v in params.items() if v is not None}
        page_size_param = clean_params.pop("pageSize", None)
        page_param = clean_params.pop("page", None)
        page_size = int(MAX_PAGE_SIZE if page_size_param is None else page_size_param)
        start_page = int(1 if page_param is None else page_param)
        if page_size <= 0:
            raise ValueError("pageSize must be a positive integer")
        if start_page <= 0:
            raise ValueError("page must be a positive integer")
        results: List[dict] = []

        page = start_page
        while True:
            query = {**clean_params, "page": page, "pageSize": page_size}
            response = self.session.get(
                f"{self.base_url}{path}",
                headers=self._headers(),
                params=query,
                timeout=self.timeout,
            )
            response.raise_for_status()
            payload = response.json()
            if isinstance(payload, dict) and "data" in payload:
                payload = payload["data"]
            if not payload:
                break
            if isinstance(payload, dict) and "items" in payload:
                # Support potential envelope structures.
                payload = payload.get("items", [])
            elif isinstance(payload, dict):
                raise ValueError("Unexpected response payload; expected list of records")
            if not isinstance(payload, list):
                raise ValueError("Unexpected response payload; expected list of records")
            batch = payload
            results.extend(batch)
            if len(batch) < page_size:
                break
            page += 1
        return results

