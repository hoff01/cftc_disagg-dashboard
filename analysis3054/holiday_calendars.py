"""Holiday calendar utilities for energy and financial workflows.

This module centralises several commonly used holiday calendars
relevant to commodity trading and analytics.  It provides
functions to retrieve dates for individual calendars or to
combine multiple calendars into a single, deduplicated schedule.

Supported calendars
-------------------
* ``us_federal`` – U.S. Federal Reserve holidays, including Juneteenth.
* ``cme`` – Standard CME Globex futures holiday closures (observed dates).
* ``ice`` – ICE U.S. futures holiday closures (observed dates).
* ``ice_europe`` – ICE Futures Europe (London) market closures.
* ``platts_us`` – Typical S&P Global Commodity Insights (Platts) U.S. office
  holidays (aligns closely with U.S. federal plus Good Friday).

Examples
--------
>>> from analysis3054.holiday_calendars import get_holidays
>>> get_holidays(2024, calendars=["cme", "ice"]).strftime('%Y-%m-%d').tolist()
['2024-01-01', '2024-01-15', '2024-02-19', '2024-03-29', '2024-05-27',
 '2024-06-19', '2024-07-04', '2024-09-02', '2024-11-28', '2024-12-25']

Use ``return_map=True`` to see which calendar contributes each date:
>>> get_holidays(2024, calendars=["us_federal", "cme"], return_map=True).head()
            us_federal    cme
2024-01-01        True    True
2024-01-15        True    True
2024-02-19        True    True
2024-03-29       False    True
2024-05-27        True    True
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Dict, Iterable, List, Sequence, Union

import pandas as pd
from pandas.tseries.holiday import (
    AbstractHolidayCalendar,
    EasterMonday,
    GoodFriday,
    Holiday,
    MO,
    USColumbusDay,
    USLaborDay,
    USMartinLutherKingJr,
    USMemorialDay,
    USPresidentsDay,
    USThanksgivingDay,
)
from pandas.tseries.offsets import DateOffset


# Additional U.S. observances not provided by pandas out of the box
Juneteenth = Holiday("Juneteenth National Independence Day", month=6, day=19, observance=lambda dt: dt if dt.weekday() < 5 else dt + DateOffset(days=1), start_date="2021")
USVeteransDay = Holiday("Veterans Day", month=11, day=11, observance=lambda dt: dt if dt.weekday() < 5 else dt + DateOffset(days=1))


def _uk_monday_or_tuesday_observance(dt):
    """Shift weekend holidays to the next available weekday.

    UK-style observance often moves Saturday holidays to Monday and Sunday
    holidays to Tuesday when Monday is already occupied.  This helper keeps the
    observance deterministic without embedding historical special cases while
    preferring the following Monday when it is available.
    """

    if dt.month == 12 and dt.day == 26 and dt.weekday() == 0:
        # Boxing Day on Monday when Christmas fell on Sunday should move to Tuesday
        if date(dt.year, 12, 25).weekday() == 6:
            return dt + DateOffset(days=1)

    if dt.weekday() < 5:
        return dt
    if dt.weekday() == 5:  # Saturday -> Monday
        return dt + DateOffset(days=2)
    if dt.month == 12 and dt.day == 26 and date(dt.year, 12, 25).weekday() == 5:
        # Boxing Day on Sunday when Christmas fell on Saturday moves to Tuesday
        return dt + DateOffset(days=2)
    return dt + DateOffset(days=1)  # Sunday -> Monday when free


class USFederalHolidayCalendar(AbstractHolidayCalendar):
    """U.S. Federal Reserve holiday calendar with observed dates."""

    rules = [
        Holiday("New Year's Day", month=1, day=1, observance=lambda dt: dt if dt.weekday() < 5 else dt + DateOffset(days=1)),
        USMartinLutherKingJr,
        USPresidentsDay,
        USMemorialDay,
        Juneteenth,
        Holiday("Independence Day", month=7, day=4, observance=lambda dt: dt if dt.weekday() < 5 else dt + DateOffset(days=1)),
        USLaborDay,
        USColumbusDay,
        USVeteransDay,
        USThanksgivingDay,
        Holiday("Christmas Day", month=12, day=25, observance=lambda dt: dt if dt.weekday() < 5 else dt + DateOffset(days=1)),
    ]


class CMEHolidayCalendar(AbstractHolidayCalendar):
    """CME Globex futures holiday calendar (full-day closures)."""

    rules = [
        Holiday("New Year's Day", month=1, day=1, observance=lambda dt: dt if dt.weekday() < 5 else dt + DateOffset(days=1)),
        USMartinLutherKingJr,
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        Juneteenth,
        Holiday("Independence Day", month=7, day=4, observance=lambda dt: dt if dt.weekday() < 5 else dt + DateOffset(days=1)),
        USLaborDay,
        USThanksgivingDay,
        Holiday("Christmas Day", month=12, day=25, observance=lambda dt: dt if dt.weekday() < 5 else dt + DateOffset(days=1)),
    ]


class ICEHolidayCalendar(AbstractHolidayCalendar):
    """ICE U.S. futures holiday calendar (full-day closures)."""

    rules = CMEHolidayCalendar.rules + []  # Same core holiday set for U.S. futures


class ICEEuropeHolidayCalendar(AbstractHolidayCalendar):
    """ICE Futures Europe (London) market closures."""

    rules = [
        Holiday("New Year's Day", month=1, day=1, observance=_uk_monday_or_tuesday_observance),
        GoodFriday,
        EasterMonday,
        Holiday("Early May Bank Holiday", month=5, day=1, offset=DateOffset(weekday=MO(1))),
        Holiday("Spring Bank Holiday", month=5, day=31, offset=DateOffset(weekday=MO(-1))),
        Holiday("Summer Bank Holiday", month=8, day=31, offset=DateOffset(weekday=MO(-1))),
        Holiday("Christmas Day", month=12, day=25, observance=_uk_monday_or_tuesday_observance),
        Holiday("Boxing Day", month=12, day=26, observance=_uk_monday_or_tuesday_observance),
    ]


class PlattsUSHolidayCalendar(AbstractHolidayCalendar):
    """Typical S&P Global Commodity Insights (Platts) U.S. office holidays."""

    rules = [
        Holiday("New Year's Day", month=1, day=1, observance=lambda dt: dt if dt.weekday() < 5 else dt + DateOffset(days=1)),
        USMartinLutherKingJr,
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        Juneteenth,
        Holiday("Independence Day", month=7, day=4, observance=lambda dt: dt if dt.weekday() < 5 else dt + DateOffset(days=1)),
        USLaborDay,
        USThanksgivingDay,
        Holiday("Christmas Day", month=12, day=25, observance=lambda dt: dt if dt.weekday() < 5 else dt + DateOffset(days=1)),
    ]


@dataclass(frozen=True)
class HolidayCalendarInfo:
    """Metadata for an available holiday calendar."""

    name: str
    description: str
    calendar: AbstractHolidayCalendar


_CALENDARS: Dict[str, HolidayCalendarInfo] = {
    "us_federal": HolidayCalendarInfo(
        name="us_federal",
        description="U.S. Federal Reserve observed holidays (includes Juneteenth).",
        calendar=USFederalHolidayCalendar(),
    ),
    "cme": HolidayCalendarInfo(
        name="cme",
        description="CME Globex U.S. futures full-day closures.",
        calendar=CMEHolidayCalendar(),
    ),
    "ice": HolidayCalendarInfo(
        name="ice",
        description="ICE U.S. futures full-day closures.",
        calendar=ICEHolidayCalendar(),
    ),
    "ice_europe": HolidayCalendarInfo(
        name="ice_europe",
        description="ICE Futures Europe (London) market closures.",
        calendar=ICEEuropeHolidayCalendar(),
    ),
    "platts_us": HolidayCalendarInfo(
        name="platts_us",
        description="Platts U.S. office holidays (federal-style plus Good Friday).",
        calendar=PlattsUSHolidayCalendar(),
    ),
}


def available_holiday_calendars() -> Dict[str, str]:
    """Return a mapping of calendar names to human-readable descriptions."""

    return {name: info.description for name, info in _CALENDARS.items()}


def _normalize_years(years: Union[int, Sequence[int]]) -> List[int]:
    """Normalize a year or iterable of years into a sorted list."""

    if isinstance(years, int):
        return [years]
    if isinstance(years, Iterable):
        year_list = [int(y) for y in years]
        if not year_list:
            raise ValueError("years must contain at least one year")
        return sorted(set(year_list))
    raise TypeError("years must be an int or an iterable of ints")


def _generate_calendar_dates(calendar: AbstractHolidayCalendar, years: List[int]) -> pd.DatetimeIndex:
    start = f"{min(years)}-01-01"
    end = f"{max(years)}-12-31"
    return calendar.holidays(start=start, end=end)


def _generate_calendar_dates_between(
    calendar: AbstractHolidayCalendar, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DatetimeIndex:
    """Generate holiday dates for a calendar between two endpoints."""

    if end < start:
        raise ValueError("end_date must be on or after start_date")
    return calendar.holidays(start=start.normalize(), end=end.normalize())


def get_holidays(
    years: Union[int, Sequence[int]],
    *,
    calendars: Union[str, Sequence[str]] = "all",
    return_map: bool = False,
) -> Union[pd.DatetimeIndex, pd.DataFrame]:
    """Return holiday dates for one or more calendars.

    Parameters
    ----------
    years : int or sequence of int
        Year or years to include (used to bound the date range).
    calendars : {'all', str, sequence of str}, default 'all'
        Calendar names to include.  Use ``'all'`` to combine every
        supported calendar.  Call :func:`available_holiday_calendars`
        to see available names.
    return_map : bool, default False
        If ``True``, return a DataFrame indexed by date with boolean
        columns indicating which calendars observe that date.  If
        ``False``, return a sorted :class:`pandas.DatetimeIndex` of
        unique holiday dates.

    Returns
    -------
    pandas.DatetimeIndex or pandas.DataFrame
        Combined holiday dates, optionally with a calendar membership map.

    Raises
    ------
    KeyError
        If an unknown calendar name is provided.
    TypeError
        If ``years`` or ``calendars`` have invalid types.
    """

    years_list = _normalize_years(years)

    if calendars == "all":
        cal_names = list(_CALENDARS.keys())
    elif isinstance(calendars, str):
        cal_names = [calendars]
    elif isinstance(calendars, Sequence):
        cal_names = list(calendars)
    else:
        raise TypeError("calendars must be 'all', a string, or a sequence of strings")

    selected = []
    for name in cal_names:
        if name not in _CALENDARS:
            raise KeyError(f"Unknown calendar '{name}'. Use available_holiday_calendars() for options.")
        selected.append(_CALENDARS[name])

    calendar_dates: Dict[str, pd.DatetimeIndex] = {}
    combined_dates: List[pd.Timestamp] = []

    for info in selected:
        dates = _generate_calendar_dates(info.calendar, years_list)
        calendar_dates[info.name] = dates
        combined_dates.extend(dates.tolist())

    combined_index = pd.DatetimeIndex(sorted(set(combined_dates)))

    if not return_map:
        return combined_index

    membership = {
        name: combined_index.isin(dates)
        for name, dates in calendar_dates.items()
    }
    result = pd.DataFrame(membership, index=combined_index)
    result.index.name = "date"
    return result


def get_holidays_between(
    start_date: Union[str, date, pd.Timestamp],
    end_date: Union[str, date, pd.Timestamp],
    *,
    calendars: Union[str, Sequence[str]] = "all",
    return_map: bool = False,
) -> Union[pd.DatetimeIndex, pd.DataFrame]:
    """Return holiday dates between two endpoints.

    This is a convenience wrapper for requesting a continuous date
    range without needing to enumerate individual years.  It supports
    past and future dates so long as the underlying calendar rules are
    defined for those years.

    Parameters
    ----------
    start_date, end_date : str, datetime.date, or pandas.Timestamp
        Inclusive date boundaries.  Strings are parsed with
        :func:`pandas.to_datetime`.
    calendars : {'all', str, sequence of str}, default 'all'
        Calendar names to include.  See :func:`available_holiday_calendars`.
    return_map : bool, default False
        Same semantics as :func:`get_holidays`.
    """

    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date)

    if calendars == "all":
        cal_names = list(_CALENDARS.keys())
    elif isinstance(calendars, str):
        cal_names = [calendars]
    elif isinstance(calendars, Sequence):
        cal_names = list(calendars)
    else:
        raise TypeError("calendars must be 'all', a string, or a sequence of strings")

    selected = []
    for name in cal_names:
        if name not in _CALENDARS:
            raise KeyError(f"Unknown calendar '{name}'. Use available_holiday_calendars() for options.")
        selected.append(_CALENDARS[name])

    calendar_dates: Dict[str, pd.DatetimeIndex] = {}
    combined_dates: List[pd.Timestamp] = []

    for info in selected:
        dates = _generate_calendar_dates_between(info.calendar, start_ts, end_ts)
        calendar_dates[info.name] = dates
        combined_dates.extend(dates.tolist())

    combined_index = pd.DatetimeIndex(sorted(set(combined_dates)))

    if not return_map:
        return combined_index

    membership = {name: combined_index.isin(dates) for name, dates in calendar_dates.items()}
    result = pd.DataFrame(membership, index=combined_index)
    result.index.name = "date"
    return result
