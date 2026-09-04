from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from contextlib import nullcontext
from datetime import date, timedelta
from pathlib import Path

from pydantic import Field

from .contracts import FrozenModel
from .market_data import DailyBar


class KronosForecast(FrozenModel):
    model: str
    as_of: date
    horizon: int = Field(gt=0)
    paths: int = Field(gt=0)
    median_return: float
    mean_return: float
    probability_positive: float = Field(ge=0, le=1)
    bear_return: float
    bull_return: float
    predicted_volatility: float = Field(ge=0)
    predicted_max_drawdown: float = Field(le=0)
    forecast_dispersion: float = Field(ge=0)


def summarize_paths(
    paths: list[list[float]], last_close: float, model: str, as_of: date,
) -> KronosForecast:
    if not paths or any(not path for path in paths) or last_close <= 0:
        raise ValueError("forecast paths and a positive last close are required")
    returns = [path[-1] / last_close - 1 for path in paths]
    ordered = sorted(returns)
    daily_returns = [
        path[index] / (last_close if index == 0 else path[index - 1]) - 1
        for path in paths for index in range(len(path))
    ]
    mean_daily = sum(daily_returns) / len(daily_returns)
    variance = sum((value - mean_daily) ** 2 for value in daily_returns) / max(
        1, len(daily_returns) - 1,
    )
    drawdowns = []
    for path in paths:
        peak = last_close
        drawdown = 0.0
        for price in path:
            peak = max(peak, price)
            drawdown = min(drawdown, price / peak - 1)
        drawdowns.append(drawdown)
    quantile = lambda fraction: ordered[round((len(ordered) - 1) * fraction)]
    mean_return = sum(returns) / len(returns)
    return KronosForecast(
        model=model, as_of=as_of, horizon=len(paths[0]), paths=len(paths),
        median_return=quantile(0.5), mean_return=mean_return,
        probability_positive=sum(value > 0 for value in returns) / len(returns),
        bear_return=quantile(0.1), bull_return=quantile(0.9),
        predicted_volatility=math.sqrt(variance * 252),
        predicted_max_drawdown=sum(drawdowns) / len(drawdowns),
        forecast_dispersion=math.sqrt(sum(
            (value - mean_return) ** 2 for value in returns
        ) / max(1, len(returns) - 1)),
    )


class KronosForecaster:
    """Lazy, cached adapter for the upstream Kronos 102M base checkpoint."""

    def __init__(
        self, repository: Path, cache_directory: Path,
        model_id: str = "NeoQuasar/Kronos-base",
        tokenizer_id: str = "NeoQuasar/Kronos-Tokenizer-base",
        lookback: int = 512, horizon: int = 21, paths: int = 10, device: str = "auto",
        recycle_every: int = 20, cooldown_seconds: float = .1,
    ):
        if recycle_every < 0 or cooldown_seconds < 0:
            raise ValueError("invalid Kronos stability controls")
        self.repository, self.cache_directory = repository, cache_directory
        self.model_id, self.tokenizer_id = model_id, tokenizer_id
        self.lookback, self.horizon, self.paths, self.device = lookback, horizon, paths, device
        self.recycle_every, self.cooldown_seconds = recycle_every, cooldown_seconds
        self._predictor = None
        self._uncached_forecasts = 0
        cache_directory.mkdir(parents=True, exist_ok=True)

    def __call__(self, bars: tuple[DailyBar, ...]) -> KronosForecast:
        usable = bars[-self.lookback:]
        if len(usable) < 64:
            raise ValueError("Kronos requires at least 64 historical sessions")
        fingerprint = hashlib.sha256(json.dumps({
            "model": self.model_id, "horizon": self.horizon, "paths": self.paths,
            "bars": [[str(bar.session), str(bar.open), str(bar.high), str(bar.low),
                      str(bar.close), bar.volume] for bar in usable],
        }, separators=(",", ":"), sort_keys=True).encode()).hexdigest()
        path = self.cache_directory / f"{fingerprint}.json"
        if path.exists():
            return KronosForecast.model_validate_json(path.read_text(encoding="utf-8"))
        predictor = self._load()
        import pandas as pd
        import torch

        frame = pd.DataFrame([{
            "open": float(bar.open), "high": float(bar.high), "low": float(bar.low),
            "close": float(bar.close), "volume": bar.volume,
            "amount": float((bar.high + bar.low + bar.close) / 3) * bar.volume,
        } for bar in usable])
        timestamps = pd.Series(pd.to_datetime([bar.session for bar in usable]))
        future = []
        cursor = usable[-1].session
        while len(future) < self.horizon:
            cursor += timedelta(days=1)
            if cursor.weekday() < 5:
                future.append(cursor)
        forecast_paths = []
        inference_context = torch.inference_mode if hasattr(torch, "inference_mode") else nullcontext
        with inference_context():
            for seed in range(self.paths):
                torch.manual_seed(seed)
                predicted = predictor.predict(
                    frame, timestamps, pd.Series(pd.to_datetime(future)),
                    pred_len=self.horizon, T=1.0, top_p=0.9, sample_count=1, verbose=False,
                )
                forecast_paths.append([float(value) for value in predicted["close"]])
        result = summarize_paths(
            forecast_paths, float(usable[-1].close), self.model_id, usable[-1].session,
        )
        temporary = path.with_suffix(".tmp")
        temporary.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        temporary.replace(path)
        self._uncached_forecasts += 1
        if self.cooldown_seconds:
            time.sleep(self.cooldown_seconds)
        if self.recycle_every and self._uncached_forecasts % self.recycle_every == 0:
            self._recycle()
        return result

    def _recycle(self) -> None:
        """Release long-lived CUDA allocations before the next bounded inference batch."""
        self._predictor = None
        import gc

        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        except RuntimeError:
            # A driver reset is unrecoverable in-process; the durable cache still permits restart.
            pass

    def _load(self):
        if self._predictor is not None:
            return self._predictor
        if not (self.repository / "model").exists():
            raise FileNotFoundError(f"Kronos repository not found at {self.repository}")
        sys.path.insert(0, str(self.repository))
        import torch
        from model import Kronos, KronosPredictor, KronosTokenizer

        device = "cuda" if self.device == "auto" and torch.cuda.is_available() else self.device
        device = "cpu" if device == "auto" else device
        tokenizer = KronosTokenizer.from_pretrained(self.tokenizer_id)
        model = Kronos.from_pretrained(self.model_id)
        self._predictor = KronosPredictor(model, tokenizer, device=device, max_context=512)
        return self._predictor
