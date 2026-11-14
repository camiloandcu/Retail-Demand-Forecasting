"""Single registry of information-availability policies and their feature rules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

INPUT_CONTRACT_VERSION: Final = 1


class PolicyMismatchError(ValueError):
    """Raised when a policy is unknown or differs from the expected regime."""

    code: Final = "POLICY_MISMATCH"


@dataclass(frozen=True)
class AvailabilityPolicy:
    """Declared inputs a forecast at an origin is allowed to read."""

    name: str
    includes_oil: bool
    excluded_calendar_types: frozenset[str]
    input_contract_version: int = INPUT_CONTRACT_VERSION


POLICIES: Final[dict[str, AvailabilityPolicy]] = {
    # Historical benchmark interface: realized target-date oil and the full table.
    "competition_v1": AvailabilityPolicy("competition_v1", True, frozenset()),
    # Operational simulation: no oil and no realized/unscheduled Event rows.
    "asof_v1": AvailabilityPolicy("asof_v1", False, frozenset({"Event"})),
}


def resolve_policy(name: object) -> AvailabilityPolicy:
    """Return a registered policy or fail closed."""

    if not isinstance(name, str) or name not in POLICIES:
        raise PolicyMismatchError(f"{PolicyMismatchError.code}: unknown availability policy")
    return POLICIES[name]


def require_same_policy(values: object, *, context: str) -> str:
    """Return the single policy among values or refuse to pool regimes."""

    distinct = {str(value) for value in values}  # type: ignore[union-attr]
    if len(distinct) != 1:
        raise PolicyMismatchError(
            f"{PolicyMismatchError.code}: {context} mixes availability policies"
        )
    (policy,) = distinct
    resolve_policy(policy)
    return policy
