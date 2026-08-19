from .contracts import PositionState

_ALLOWED = {
    PositionState.CASH_OR_SPY: {PositionState.CANDIDATE},
    PositionState.CANDIDATE: {PositionState.APPROVED, PositionState.BLOCKED_DATA, PositionState.BLOCKED_RISK},
    PositionState.APPROVED: {PositionState.ORDER_PENDING, PositionState.BLOCKED_RISK},
    PositionState.ORDER_PENDING: {PositionState.OPEN, PositionState.BLOCKED_RISK},
    PositionState.OPEN: {PositionState.ADD_ALLOWED, PositionState.REDUCE_REQUIRED, PositionState.EXIT_REQUIRED},
    PositionState.ADD_ALLOWED: {PositionState.ORDER_PENDING, PositionState.OPEN},
    PositionState.REDUCE_REQUIRED: {PositionState.ORDER_PENDING},
    PositionState.EXIT_REQUIRED: {PositionState.ORDER_PENDING},
    PositionState.CLOSED: {PositionState.CANDIDATE},
    PositionState.BLOCKED_DATA: {PositionState.CANDIDATE},
    PositionState.BLOCKED_RISK: {PositionState.CANDIDATE, PositionState.OPEN},
}


def transition(current: PositionState, target: PositionState) -> PositionState:
    if target not in _ALLOWED.get(current, set()):
        raise ValueError(f"invalid transition: {current} -> {target}")
    return target

