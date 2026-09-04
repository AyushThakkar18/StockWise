from __future__ import annotations

import math
from datetime import date


def forward_signal_diagnostics(
    audits: dict[date, dict[str, object]], universe: dict[str, tuple], horizon: int = 21,
) -> dict[str, dict[str, float]]:
    """Measure cross-sectional rank IC and quintile spread using future, never input, returns."""
    indexed = {
        symbol: ({bar.session: index for index, bar in enumerate(bars)}, bars)
        for symbol, bars in universe.items()
    }
    factor_rows: dict[str, list[tuple[float, float, date]]] = {}
    excluded = {"rank", "score", "symbol", "sector", "fundamental_completeness"}
    for decision_on, audit in audits.items():
        for row in audit.get("ranked", []):  # type: ignore[union-attr]
            symbol = row["symbol"]
            if symbol not in indexed or decision_on not in indexed[symbol][0]:
                continue
            positions, bars = indexed[symbol]
            start = positions[decision_on]
            if start + horizon >= len(bars):
                continue
            forward = float(bars[start + horizon].adjusted_close / bars[start].adjusted_close - 1)
            for factor, value in row.items():
                if factor in excluded or value is None or not isinstance(value, (int, float)):
                    continue
                if math.isfinite(float(value)):
                    factor_rows.setdefault(factor, []).append((float(value), forward, decision_on))
    return {factor: _summarize(rows) for factor, rows in sorted(factor_rows.items())}


def _summarize(rows: list[tuple[float, float, date]]) -> dict[str, float]:
    by_date: dict[date, list[tuple[float, float]]] = {}
    for signal, forward, decision_on in rows:
        by_date.setdefault(decision_on, []).append((signal, forward))
    correlations, spreads = [], []
    for values in by_date.values():
        if len(values) < 10:
            continue
        signals, forwards = zip(*values, strict=True)
        correlations.append(_spearman(signals, forwards))
        ordered = sorted(values)
        count = max(1, len(ordered) // 5)
        spreads.append(
            sum(value for _, value in ordered[-count:]) / count
            - sum(value for _, value in ordered[:count]) / count
        )
    return {
        "observations": float(len(rows)),
        "periods": float(len(correlations)),
        "mean_rank_ic": sum(correlations) / len(correlations) if correlations else 0.0,
        "positive_ic_rate": (
            sum(value > 0 for value in correlations) / len(correlations) if correlations else 0.0
        ),
        "mean_top_minus_bottom_quintile_return": (
            sum(spreads) / len(spreads) if spreads else 0.0
        ),
    }


def _spearman(left, right) -> float:
    a, b = _ranks(left), _ranks(right)
    am, bm = sum(a) / len(a), sum(b) / len(b)
    numerator = sum((x - am) * (y - bm) for x, y in zip(a, b, strict=True))
    denominator = math.sqrt(sum((x - am) ** 2 for x in a) * sum((y - bm) ** 2 for y in b))
    return numerator / denominator if denominator else 0.0


def _ranks(values) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    output = [0.0] * len(ordered)
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][1] == ordered[index][1]:
            end += 1
        rank = (index + end - 1) / 2
        for original, _ in ordered[index:end]:
            output[original] = rank
        index = end
    return output
