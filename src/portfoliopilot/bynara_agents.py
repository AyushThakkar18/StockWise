from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pydantic import Field

from .contracts import FrozenModel

PROMPT_VERSION = "bounded-factor-agents-v3-relative-signal-rubric"
ROLES = {
    "TECHNICAL": (
        "Evaluate persistence versus reversal over the next 21 sessions. Emphasize relative trend, "
        "multi-horizon agreement, volatility and drawdown. Do not assume momentum always persists."
    ),
    "QUALITY": (
        "Evaluate whether supplied period-aligned growth, profitability, balance-sheet and data-"
        "completeness features support resilience. Missing data is uncertainty, not evidence."
    ),
    "RISK": (
        "Act adversarially. Evaluate downside, beta, drawdown, volatility, liquidity and crowded-"
        "trend risk. A positive signal means favorable risk-adjusted setup; negative means avoid."
    ),
}


class AgentFeature(FrozenModel):
    role: str
    signal: int = Field(ge=-2, le=2)
    confidence: float = Field(ge=0, le=1)
    reason: str = Field(min_length=1, max_length=300)
    risk_flags: tuple[str, ...] = Field(max_length=4)


class CrossSectionFeature(FrozenModel):
    id: str
    signal: int = Field(ge=-2, le=2)
    confidence: float = Field(ge=0, le=1)
    reason: str = Field(min_length=1, max_length=300)
    risk_flags: tuple[str, ...] = Field(max_length=4)


class CrossSectionAssessment(FrozenModel):
    assessments: tuple[CrossSectionFeature, ...]


class BynaraFeatureAgent:
    def __init__(
        self, api_key: str, model: str, base_url: str, cache_directory: Path,
        attempts: int = 2, timeout: int = 45,
    ):
        if not api_key:
            raise ValueError("BYNARA_API_KEY or bynaraKey is required")
        self.api_key, self.model, self.base_url = api_key, model, base_url.rstrip("/")
        self.cache_directory, self.attempts, self.timeout = cache_directory, attempts, timeout
        cache_directory.mkdir(parents=True, exist_ok=True)

    def evaluate(self, role: str, packet: dict[str, object]) -> AgentFeature:
        if role not in ROLES:
            raise ValueError(f"unknown agent role {role}")
        encoded = json.dumps(packet, sort_keys=True, separators=(",", ":"))
        fingerprint = hashlib.sha256(
            f"{self.model}:{PROMPT_VERSION}:{role}:{encoded}".encode(),
        ).hexdigest()
        path = self.cache_directory / f"{fingerprint}.json"
        if path.exists():
            return AgentFeature.model_validate_json(path.read_text(encoding="utf-8"))
        body = json.dumps({
            "model": self.model,
            "messages": [
                {"role": "system", "content": (
                    "You are one bounded feature-extraction agent in a systematic strategy. "
                    "Use only anonymous numeric data supplied. Return JSON only with role, integer "
                    "signal (-2 strongly unfavorable to +2 strongly favorable), confidence (0-1), "
                    "short reason, and risk_flags. Never identify the stock, predict exact prices, "
                    "or override portfolio rules. " + ROLES[role]
                )},
                {"role": "user", "content": encoded},
            ],
            "response_format": {"type": "json_object"},
        }).encode()
        request = Request(
            f"{self.base_url}/chat/completions", data=body, method="POST",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
        )
        for attempt in range(self.attempts):
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    payload = json.loads(response.read())
                raw = json.loads(payload["choices"][0]["message"]["content"])
                raw["role"] = role
                result = AgentFeature.model_validate(raw)
                path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
                return result
            except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, ValueError, KeyError):
                if attempt + 1 == self.attempts:
                    raise
                time.sleep(2 ** attempt)
        raise RuntimeError("unreachable")

    def evaluate_many(
        self, packets: dict[str, dict[str, object]], workers: int = 2,
    ) -> dict[str, dict[str, AgentFeature]]:
        output: dict[str, dict[str, AgentFeature]] = {symbol: {} for symbol in packets}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(self.evaluate, role, packet): (symbol, role)
                for symbol, packet in packets.items() for role in ROLES
            }
            for index, future in enumerate(as_completed(futures), 1):
                symbol, role = futures[future]
                output[symbol][role] = future.result()
                if index % 15 == 0 or index == len(futures):
                    print(f"Bynara agent features {index}/{len(futures)}", flush=True)
        return output

    def evaluate_cross_section(
        self, role: str, packets: dict[str, dict[str, object]],
    ) -> dict[str, AgentFeature]:
        if role not in ROLES:
            raise ValueError(f"unknown agent role {role}")
        aliases = {f"asset_{index:03d}": symbol for index, symbol in enumerate(packets, 1)}
        payload = {
            "prompt_version": PROMPT_VERSION,
            "horizon_sessions": 21,
            "required_ids": list(aliases),
            "candidates": [
                {"id": alias, "data": packets[symbol]} for alias, symbol in aliases.items()
            ],
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        fingerprint = hashlib.sha256(
            f"{self.model}:{PROMPT_VERSION}:cross-section:{role}:{encoded}".encode(),
        ).hexdigest()
        path = self.cache_directory / f"{fingerprint}.json"
        if path.exists():
            assessment = CrossSectionAssessment.model_validate_json(path.read_text(encoding="utf-8"))
        else:
            assessment = self._request_cross_section(role, encoded)
            path.write_text(assessment.model_dump_json(indent=2), encoding="utf-8")
        received = [item.id for item in assessment.assessments]
        if len(received) != len(aliases) or set(received) != set(aliases):
            raise ValueError("Bynara cross-sectional agent omitted, duplicated, or invented candidate IDs")
        return {
            aliases[item.id]: AgentFeature(
                role=role, signal=item.signal, confidence=item.confidence,
                reason=item.reason, risk_flags=item.risk_flags,
            ) for item in assessment.assessments
        }

    def _request_cross_section(self, role: str, encoded: str) -> CrossSectionAssessment:
        body = json.dumps({
            "model": self.model,
            "messages": [
                {"role": "system", "content": (
                    "You are one independent cross-sectional feature agent in a systematic strategy. "
                    "Compare every anonymous candidate using only supplied numeric data. Return JSON "
                    "with an assessments array containing exactly one object per required ID. Each "
                    "object needs id, integer signal (-2 to +2), confidence (0-1), short reason, and "
                    "risk_flags. Use the full signal range relatively; never identify companies, omit "
                    "IDs, choose portfolio weights, or use outside knowledge. " + ROLES[role]
                )},
                {"role": "user", "content": encoded},
            ],
            "response_format": {"type": "json_object"},
        }).encode()
        request = Request(
            f"{self.base_url}/chat/completions", data=body, method="POST",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
        )
        for attempt in range(self.attempts):
            try:
                with urlopen(request, timeout=self.timeout * 3) as response:
                    response_payload = json.loads(response.read())
                raw = json.loads(response_payload["choices"][0]["message"]["content"])
                return CrossSectionAssessment.model_validate(raw)
            except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, ValueError, KeyError):
                if attempt + 1 == self.attempts:
                    raise
                time.sleep(2 ** attempt)
        raise RuntimeError("unreachable")
