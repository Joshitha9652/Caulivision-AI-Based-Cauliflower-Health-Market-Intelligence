"""Fetch official daily mandi prices, with clearly labelled static reference data."""

import json
import os
import time
from datetime import datetime
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from ap_locations import get_all_ap_mandis, get_all_districts

RESOURCE_URL = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"
CACHE_SECONDS = 15 * 60
REQUEST_TIMEOUT_SECONDS = 12

_cached_markets: list[dict] | None = None
_cache_expires_at = 0.0


class MarketDataError(RuntimeError):
    """Raised when the official market-price feed cannot be read or parsed."""


def get_market_data(district: str | None = None) -> dict:
    """Return live data when configured, otherwise clearly labelled reference data."""
    api_key = os.getenv("DATA_GOV_IN_API_KEY", "").strip()
    if not api_key:
        return _reference_response(district)

    try:
        markets = _fetch_live_markets(api_key)
    except MarketDataError as exc:
        return _reference_response(district, warning=str(exc))

    if district:
        markets = [
            market
            for market in markets
            if market["district"].casefold() == district.casefold()
        ]
    return {
        "ok": True,
        "source": "data.gov.in",
        "last_updated": datetime.now().astimezone().isoformat(timespec="minutes"),
        "warning": None,
        "districts": sorted({market["district"] for market in markets}),
        "markets": markets,
    }


def _fetch_live_markets(api_key: str) -> list[dict]:
    global _cached_markets, _cache_expires_at

    if _cached_markets is not None and time.monotonic() < _cache_expires_at:
        return _cached_markets

    query = urlencode(
        {
            "api-key": api_key,
            "format": "json",
            "limit": "1000",
            "filters[state]": "Andhra Pradesh",
            "filters[commodity]": "Cauliflower",
            "sort[arrival_date]": "desc",
        }
    )
    request = Request(
        f"{RESOURCE_URL}?{query}",
        headers={"Accept": "application/json", "User-Agent": "CauliflowerFarmerDashboard/1.0"},
    )
    try:
        with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            payload = json.load(response)
    except HTTPError as exc:
        raise MarketDataError(
            f"Official data.gov.in request failed with HTTP {exc.code}."
        ) from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise MarketDataError(
            "Could not connect to the official data.gov.in market-price feed."
        ) from exc
    except json.JSONDecodeError as exc:
        raise MarketDataError("The official market-price feed returned invalid JSON.") from exc

    if not isinstance(payload, dict):
        raise MarketDataError("The official market-price feed returned an unexpected response.")
    records = payload.get("records")
    if not isinstance(records, list):
        raise MarketDataError("The official market-price feed response has no records list.")

    markets = _normalize_records(records)
    _cached_markets = markets
    _cache_expires_at = time.monotonic() + CACHE_SECONDS
    return markets


def _normalize_records(records: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, str], list[dict]] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        district = str(record.get("district", "")).strip()
        market_name = str(record.get("market", "")).strip()
        if not district or not market_name:
            continue

        variety = str(record.get("variety", "")).strip()
        key = (district, market_name, variety)
        entry_date = _parse_arrival_date(str(record.get("arrival_date", "")))
        try:
            minimum = float(record["min_price"]) / 100
            modal = float(record["modal_price"]) / 100
            maximum = float(record["max_price"]) / 100
        except (KeyError, TypeError, ValueError):
            continue

        grouped.setdefault(key, []).append(
            {
                "date": entry_date,
                "min_price": minimum,
                "modal_price": modal,
                "max_price": maximum,
            }
        )

    markets = []
    for (district, market_name, variety), observations in grouped.items():
        observations.sort(key=lambda item: item["date"], reverse=True)
        latest = observations[0]
        daily_prices = []
        seen_dates = set()
        for observation in observations:
            date = observation["date"]
            if date in seen_dates:
                continue
            seen_dates.add(date)
            daily_prices.append({"date": date, "price": observation["modal_price"]})
            if len(daily_prices) == 7:
                break

        markets.append(
            {
                "district": district,
                "region": "",
                "mandi_name": market_name,
                "market_type": "Official mandi market",
                "variety": variety,
                "min_price": round(latest["min_price"], 2),
                "modal_price": round(latest["modal_price"], 2),
                "max_price": round(latest["max_price"], 2),
                "price_unit": "₹/kg",
                "arrival_date": latest["date"],
                "recent_prices": list(reversed(daily_prices)),
                "notes": "Official daily mandi price from data.gov.in (AGMARKNET).",
            }
        )
    return sorted(markets, key=lambda market: (market["district"], market["mandi_name"]))


def _parse_arrival_date(value: str) -> str:
    for date_format in ("%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y"):
        try:
            return datetime.strptime(value, date_format).date().isoformat()
        except ValueError:
            pass
    return value


def _reference_response(district: str | None, warning: str | None = None) -> dict:
    markets = get_all_ap_mandis()
    if district:
        markets = [
            market
            for market in markets
            if market["district"].casefold() == district.casefold()
        ]

    reference_markets = [
        {
            **market,
            "price_unit": "₹/kg",
            "arrival_date": None,
            "recent_prices": [],
        }
        for market in markets
    ]
    return {
        "ok": True,
        "source": "reference",
        "last_updated": None,
        "warning": warning,
        "districts": get_all_districts(),
        "markets": reference_markets,
    }
