from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from statistics import pstdev

from pydantic import Field

from .backtest import total_return_ratio
from .contracts import FrozenModel
from .market_data import DailyBar

PROMPT_VERSION = "anonymous-quant-ranker-v1"


class RankedAssets(FrozenModel):
    ranked_ids: tuple[str, ...] = Field(min_length=20)


def feature_packet(history: dict[str, tuple[DailyBar, ...]]) -> tuple[dict[str, object], dict[str, str]]:
    eligible = sorted((symbol, bars) for symbol, bars in history.items() if len(bars) > 252)
    aliases = {f"asset_{index:03d}": symbol for index, (symbol, _) in enumerate(eligible, 1)}
    candidates = []
    for alias, (_, bars) in zip(aliases, eligible, strict=True):
        returns = [float(bars[index].close / bars[index - 1].close - 1)
                   for index in range(len(bars) - 62, len(bars))]
        candidates.append({
            "id": alias,
            "momentum_21": round(float(total_return_ratio(bars[-22:]) - 1), 8),
            "momentum_63": round(float(total_return_ratio(bars[-64:]) - 1), 8),
            "momentum_126": round(float(total_return_ratio(bars[-127:]) - 1), 8),
            "momentum_252": round(float(total_return_ratio(bars[-253:]) - 1), 8),
            "volatility_63": round(pstdev(returns) * (252 ** 0.5), 8),
        })
    return {"prompt_version": PROMPT_VERSION, "candidates": candidates}, aliases


class OpenAIRanker:
    def __init__(
        self, api_key: str, model: str, cache_directory: Path,
        maximum_attempts: int = 5, initial_delay: float = 1.0,
    ):
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required")
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.cache_directory = cache_directory
        self.maximum_attempts = maximum_attempts
        self.initial_delay = initial_delay
        if maximum_attempts <= 0 or initial_delay < 0:
            raise ValueError("invalid retry policy")
        cache_directory.mkdir(parents=True, exist_ok=True)

    def __call__(self, packet: dict[str, object]) -> tuple[str, ...]:
        encoded = json.dumps(packet, sort_keys=True, separators=(",", ":"))
        fingerprint = hashlib.sha256(
            f"{self.model}:{encoded}".encode()
        ).hexdigest()
        path = self.cache_directory / f"{fingerprint}.json"
        if path.exists():
            return RankedAssets.model_validate_json(path.read_text(encoding="utf-8")).ranked_ids
        for attempt in range(self.maximum_attempts):
            try:
                response = self.client.responses.parse(
                    model=self.model,
                    input=[
                        {"role": "system", "content": (
                            "Rank anonymous assets using only the supplied numeric features. Prefer robust "
                            "positive momentum across horizons and penalize excessive volatility. Do not use "
                            "external knowledge. Return at least the 20 strongest candidate IDs, strongest first, "
                            "with no duplicate or unknown IDs."
                        )},
                        {"role": "user", "content": encoded},
                    ],
                    text_format=RankedAssets,
                    temperature=0,
                )
                break
            except Exception:
                if attempt + 1 == self.maximum_attempts:
                    raise
                time.sleep(self.initial_delay * (2 ** attempt))
        ranking = response.output_parsed
        if ranking is None:
            raise ValueError("model returned no structured ranking")
        expected = {item["id"] for item in packet["candidates"]}  # type: ignore[index]
        normalized = tuple(dict.fromkeys(item for item in ranking.ranked_ids if item in expected))
        if len(normalized) < 20:
            candidates = packet["candidates"]  # type: ignore[assignment]
            fallback = sorted(
                (item for item in candidates if item["id"] not in normalized),  # type: ignore[union-attr]
                key=lambda item: (-(item["momentum_63"] + item["momentum_126"]  # type: ignore[index]
                                  - item["volatility_63"] * 0.25), item["id"]),  # type: ignore[index]
            )
            normalized += tuple(item["id"] for item in fallback[:20 - len(normalized)])  # type: ignore[index]
        ranking = RankedAssets(ranked_ids=normalized)
        path.write_text(ranking.model_dump_json(), encoding="utf-8")
        path.with_suffix(".audit.json").write_text(json.dumps({
            "model": self.model, "prompt_version": PROMPT_VERSION,
            "ai_valid_count": min(len(set(response.output_parsed.ranked_ids)), len(expected)),
            "deterministic_fallback_count": max(0, 20 - len(set(response.output_parsed.ranked_ids))),
        }, sort_keys=True), encoding="utf-8")
        return ranking.ranked_ids


@dataclass(frozen=True)
class AIRankedPortfolio:
    ranker: Callable[[dict[str, object]], tuple[str, ...]]
    top_n: int = 20
    name: str = "ai_anonymous_quant_ranking"

    def targets(self, history: dict[str, tuple[DailyBar, ...]]) -> dict[str, Decimal]:
        packet, aliases = feature_packet(history)
        if len(aliases) < self.top_n:
            return {}
        ranking = self.ranker(packet)
        selected = [aliases[item] for item in ranking[:self.top_n]]
        weight = Decimal(1) / Decimal(len(selected))
        return {symbol: weight for symbol in selected}
