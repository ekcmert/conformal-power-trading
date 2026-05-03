from __future__ import annotations

import os
from datetime import date, datetime, time
from pathlib import Path
from typing import Any, Literal, Sequence

import pandas as pd

try:
    from energyquantified import EnergyQuantified
    from energyquantified.time import Frequency
except ImportError:
    EnergyQuantified = None
    Frequency = None


DateLike = str | date | datetime
IssuePolicy = Literal["latest", "earliest"]
DEFAULT_API_KEY_ENV_VARS = ("ENERGYQUANTIFIED_API_KEY", "EQ_API_KEY")
DEFAULT_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


class EQClient:
    """Thin wrapper for the four Energy Quantified data types used here."""

    def __init__(
        self,
        api_key: str | None = None,
        client: Any | None = None,
        api_key_env_vars: Sequence[str] = DEFAULT_API_KEY_ENV_VARS,
    ) -> None:
        self.api_key_env_vars = tuple(api_key_env_vars)
        self.client = client or self._build_client(api_key)

    def load_timeseries(
        self,
        curve: Any,
        begin: DateLike,
        end: DateLike,
        **kwargs: Any,
    ) -> pd.DataFrame:
        timeseries = self.client.timeseries.load(curve, begin=begin, end=end, **kwargs)
        return self._to_dataframe(timeseries)

    def load_scenario(
        self,
        curve: Any,
        begin: DateLike,
        end: DateLike,
        **kwargs: Any,
    ) -> pd.DataFrame:
        return self.load_timeseries(curve=curve, begin=begin, end=end, **kwargs)

    def load_instance(
        self,
        curve: Any,
        begin: DateLike,
        end: DateLike,
        tag: str,
        days_ahead: int = 1,
        before_time_of_day: time = time(12, 0),
        issued: IssuePolicy = "latest",
        frequency: Any = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        params: dict[str, Any] = {
            "begin": begin,
            "end": end,
            "tag": tag,
            "days_ahead": days_ahead,
            "before_time_of_day": before_time_of_day,
            "issued": issued,
            **kwargs,
        }

        resolved_frequency = frequency or self._default_instance_frequency()
        if resolved_frequency is not None:
            params["frequency"] = resolved_frequency

        timeseries = self.client.instances.relative(curve, **params)
        return self._to_dataframe(timeseries)

    def load_ohlc(
        self,
        curve: Any,
        begin: DateLike,
        end: DateLike,
        **kwargs: Any,
    ) -> pd.DataFrame:
        ohlc = self.client.ohlc.load(curve, begin=begin, end=end, **kwargs)
        return self._to_dataframe(ohlc)

    def list_instance_tags(self, curve: Any) -> list[str]:
        return sorted(str(tag) for tag in self.client.instances.tags(curve))

    def _build_client(self, api_key: str | None) -> Any:
        if EnergyQuantified is None:
            raise ImportError(
                "The `energyquantified` package is not installed. Install it with "
                "`pip install energyquantified` before using EQClient."
            )

        resolved_api_key = api_key or self._resolve_api_key()
        if not resolved_api_key:
            env_var_list = ", ".join(self.api_key_env_vars)
            raise ValueError(
                "Missing Energy Quantified API key. Pass `api_key=` or set one of "
                f"these environment variables: {env_var_list}."
            )

        return EnergyQuantified(api_key=resolved_api_key)

    def _resolve_api_key(self) -> str | None:
        for env_var in self.api_key_env_vars:
            value = os.getenv(env_var)
            if value:
                return value
        return self._resolve_api_key_from_env_file()

    def _resolve_api_key_from_env_file(self) -> str | None:
        if not DEFAULT_ENV_FILE.exists():
            return None

        for raw_line in DEFAULT_ENV_FILE.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("\"'")
            if key in self.api_key_env_vars and value:
                return value
        return None

    @staticmethod
    def _to_dataframe(payload: Any) -> pd.DataFrame:
        if payload is None:
            return pd.DataFrame()
        if isinstance(payload, pd.DataFrame):
            return payload.copy()
        if hasattr(payload, "to_pandas_dataframe"):
            return payload.to_pandas_dataframe()
        raise TypeError(
            "Expected an Energy Quantified object with `to_pandas_dataframe()`."
        )

    @staticmethod
    def _default_instance_frequency() -> Any:
        if Frequency is None:
            return None
        return Frequency.PT1H


__all__ = ["DEFAULT_API_KEY_ENV_VARS", "EQClient"]
