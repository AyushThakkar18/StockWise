from portfoliopilot.comparative_council import PortfolioSelection, validate_selection


def packet():
    return {
        "constraints": {"maximum_per_sector": 5},
        "candidates": [
            {"id": f"asset_{index:03d}", "sector": f"sector-{index % 5}"}
            for index in range(1, 101)
        ],
    }


def test_comparative_selection_requires_exact_diversified_top_20() -> None:
    selection = PortfolioSelection(
        selected_ids=tuple(f"asset_{index:03d}" for index in range(1, 21)),
        selection_summary="Diversified selection",
    )
    assert validate_selection(selection, packet()) is None


def test_comparative_selection_rejects_sector_concentration() -> None:
    payload = packet()
    for item in payload["candidates"][:20]:
        item["sector"] = "Technology"
    selection = PortfolioSelection(
        selected_ids=tuple(f"asset_{index:03d}" for index in range(1, 21)),
        selection_summary="Concentrated selection",
    )
    assert "sector limit" in validate_selection(selection, payload)
