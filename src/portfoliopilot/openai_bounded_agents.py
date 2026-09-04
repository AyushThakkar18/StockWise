from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from .bynara_agents import PROMPT_VERSION, ROLES, AgentFeature, CrossSectionAssessment

ALLOWED_MODEL = "gpt-4o-mini"


class OpenAIBoundedAgent:
    """One-call-per-role, structured feature reviewer with a durable local cache."""

    def __init__(
        self, api_key: str, cache_directory: Path, model: str = ALLOWED_MODEL,
        attempts: int = 3, timeout: float = 120, horizon_sessions: int = 21,
        roles: dict[str, str] | None = None, prompt_version: str = PROMPT_VERSION,
    ):
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required")
        if model != ALLOWED_MODEL:
            raise ValueError(f"bounded council only permits {ALLOWED_MODEL}")
        if attempts < 1 or timeout <= 0:
            raise ValueError("invalid retry policy")
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key, timeout=timeout, max_retries=0)
        if horizon_sessions < 1:
            raise ValueError("horizon_sessions must be positive")
        self.model, self.cache_directory, self.attempts = model, cache_directory, attempts
        self.horizon_sessions = horizon_sessions
        self.roles = dict(roles or ROLES)
        self.prompt_version = prompt_version
        self.requests_made = 0
        cache_directory.mkdir(parents=True, exist_ok=True)

    def evaluate_cross_section(
        self, role: str, packets: dict[str, dict[str, object]],
    ) -> dict[str, AgentFeature]:
        if role not in self.roles:
            raise ValueError(f"unknown agent role {role}")
        aliases = {f"asset_{index:03d}": symbol for index, symbol in enumerate(packets, 1)}
        payload = {
            "prompt_version": self.prompt_version,
            "horizon_sessions": self.horizon_sessions,
            "required_ids": list(aliases),
            "candidates": [
                {"id": alias, "data": packets[symbol]} for alias, symbol in aliases.items()
            ],
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        fingerprint = hashlib.sha256(
            f"openai:{self.model}:{self.prompt_version}:{role}:{encoded}".encode(),
        ).hexdigest()
        path = self.cache_directory / f"{fingerprint}.json"
        if path.exists():
            assessment = CrossSectionAssessment.model_validate_json(path.read_text(encoding="utf-8"))
        else:
            assessment = self._request(role, encoded)
            path.write_text(assessment.model_dump_json(indent=2), encoding="utf-8")
        received = [item.id for item in assessment.assessments]
        if len(received) != len(aliases) or set(received) != set(aliases):
            raise ValueError("OpenAI agent omitted, duplicated, or invented candidate IDs")
        return {
            aliases[item.id]: AgentFeature(
                role=role, signal=item.signal, confidence=item.confidence,
                reason=item.reason, risk_flags=item.risk_flags,
            ) for item in assessment.assessments
        }

    def _request(self, role: str, encoded: str) -> CrossSectionAssessment:
        instructions = (
            "You are one independent cross-sectional feature agent in a systematic strategy. "
            "Compare every anonymous candidate using only supplied numeric data. Return exactly "
            "one assessment per required ID. Use the full signal range relatively; never identify "
            "companies, omit IDs, select securities, set weights, or use outside knowledge. "
            "Signal rubric: +2 strongly favorable relative setup, +1 favorable, 0 genuinely "
            "neutral or balanced, -1 unfavorable, -2 strongly unfavorable. Describing evidence "
            "as favorable or unfavorable while returning 0 is invalid. Judge candidates relative "
            "to this supplied cross-section and use at least three distinct signal values unless "
            "the numeric packets are effectively identical. For Kronos fields, compare candidate "
            "forecasts with the supplied benchmark forecast and relative score; do not reject a "
            "candidate merely because both candidate and benchmark forecasts are negative. "
            "Confidence measures reliability of your assessment, not bullishness. Use 0 only when "
            "the candidate packet is unusable; ordinary mixed evidence should normally receive "
            "0.3-0.7 and unusually complete, internally consistent evidence may receive 0.7-0.9. "
            + self.roles[role].replace("next 21 sessions", f"next {self.horizon_sessions} sessions")
        )
        for attempt in range(self.attempts):
            try:
                self.requests_made += 1
                response = self.client.responses.parse(
                    model=self.model,
                    instructions=instructions,
                    input=encoded,
                    text_format=CrossSectionAssessment,
                    temperature=0,
                    store=False,
                )
                if response.output_parsed is None:
                    raise ValueError("model returned no structured assessment")
                return response.output_parsed
            except Exception:
                if attempt + 1 == self.attempts:
                    raise
                time.sleep(2 ** attempt)
        raise RuntimeError("unreachable")
