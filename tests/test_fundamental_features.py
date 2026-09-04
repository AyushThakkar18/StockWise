from datetime import date

from portfoliopilot.fundamental_features import build_fundamental_features


def test_fundamentals_use_only_filed_annual_values() -> None:
    def fact(values):
        return {"units": {"USD": values}}

    annual = [
        {"val": 100, "form": "10-K", "filed": "2023-02-01", "start": "2022-01-01", "end": "2022-12-31"},
        {"val": 120, "form": "10-K", "filed": "2024-02-01", "start": "2023-01-01", "end": "2023-12-31"},
        {"val": 999, "form": "10-K", "filed": "2025-02-01", "start": "2024-01-01", "end": "2024-12-31"},
    ]
    payload = {"facts": {"us-gaap": {
        "Revenues": fact(annual), "NetIncomeLoss": fact([{**item, "val": 12} for item in annual]),
        "OperatingIncomeLoss": fact([{**item, "val": 18} for item in annual]),
        "Assets": fact([{"val": 200, "form": "10-Q", "filed": "2024-01-15", "end": "2023-12-31"}]),
        "Liabilities": fact([{"val": 80, "form": "10-Q", "filed": "2024-01-15", "end": "2023-12-31"}]),
        "CashAndCashEquivalentsAtCarryingValue": fact([{"val": 20, "form": "10-Q", "filed": "2024-01-15", "end": "2023-12-31"}]),
    }}}
    result = build_fundamental_features(payload, date(2024, 6, 1))
    assert round(result.revenue_growth or 0, 4) == 0.2
    assert result.liabilities_to_assets == 0.4
    assert result.completeness == 1
