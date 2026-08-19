import pytest

from portfoliopilot.contracts import PositionState
from portfoliopilot.state import transition


def test_valid_candidate_path() -> None:
    assert transition(PositionState.CANDIDATE, PositionState.APPROVED) == PositionState.APPROVED


def test_invalid_shortcut_is_rejected() -> None:
    with pytest.raises(ValueError):
        transition(PositionState.CANDIDATE, PositionState.OPEN)
