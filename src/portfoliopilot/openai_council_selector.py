from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from pydantic import Field

from .contracts import FrozenModel
from .openai_bounded_agents import ALLOWED_MODEL

SELECTION_PROMPT_VERSION = "kronos-council-synthesis-v1"


class CouncilSelection(FrozenModel):
    selected_ids: tuple[str, ...] = Field(max_length=20)
    rationale: str = Field(min_length=1, max_length=1000)
    abstention_reason: str | None = Field(default=None, max_length=500)


class OpenAICouncilSelector:
    def __init__(
        self, api_key: str, cache_directory: Path, model: str = ALLOWED_MODEL,
        attempts: int = 3, timeout: float = 180,
    ):
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required")
        if model != ALLOWED_MODEL:
            raise ValueError(f"council selector only permits {ALLOWED_MODEL}")
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key, timeout=timeout, max_retries=0)
        self.cache_directory, self.model, self.attempts = cache_directory, model, attempts
        self.requests_made = 0
        cache_directory.mkdir(parents=True, exist_ok=True)

    def select(self, candidates: list[dict[str, object]]) -> CouncilSelection:
        aliases = {f"asset_{index:03d}": item["symbol"] for index, item in enumerate(candidates, 1)}
        payload = {
            "prompt_version": SELECTION_PROMPT_VERSION,
            "maximum_selections": 20,
            "required_ids": list(aliases),
            "candidates": [
                {"id": alias, **{key: value for key, value in item.items() if key != "symbol"}}
                for alias, item in zip(aliases, candidates, strict=True)
            ],
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        fingerprint = hashlib.sha256(
            f"{self.model}:{SELECTION_PROMPT_VERSION}:{encoded}".encode(),
        ).hexdigest()
        path = self.cache_directory / f"{fingerprint}.json"
        if path.exists():
            raw = CouncilSelection.model_validate_json(path.read_text(encoding="utf-8"))
        else:
            raw = self._request(encoded)
            path.write_text(raw.model_dump_json(indent=2), encoding="utf-8")
        selected = tuple(dict.fromkeys(raw.selected_ids))
        if len(selected) != len(raw.selected_ids) or not set(selected) <= set(aliases):
            raise ValueError("selector duplicated or invented candidate IDs")
        return CouncilSelection(
            selected_ids=tuple(aliases[item] for item in selected),
            rationale=raw.rationale, abstention_reason=raw.abstention_reason,
        )

    def _request(self, encoded: str) -> CouncilSelection:
        instructions = (
            "You are the synthesis member of a bounded investment council. Compare all 50 "
            "anonymous candidates using their deterministic rank, Kronos forecast relative to SPY, "
            "and technical, quality, and risk specialist reports. Select zero to twenty IDs with "
            "the strongest favorable risk-adjusted evidence for the next 21 sessions. You are not "
            "required to fill twenty slots. Prefer abstention to weak evidence, but judge forecasts "
            "relative to the benchmark: a negative candidate forecast can still be favorable if "
            "the benchmark is worse. Avoid strong risk objections and explain the cross-sectional "
            "tradeoff briefly. Do not identify companies, set weights, or use outside knowledge."
        )
        for attempt in range(self.attempts):
            try:
                self.requests_made += 1
                response = self.client.responses.parse(
                    model=self.model, instructions=instructions, input=encoded,
                    text_format=CouncilSelection, temperature=0, store=False,
                )
                if response.output_parsed is None:
                    raise ValueError("selector returned no structured output")
                return response.output_parsed
            except Exception:
                if attempt + 1 == self.attempts:
                    raise
                time.sleep(2 ** attempt)
        raise RuntimeError("unreachable")
