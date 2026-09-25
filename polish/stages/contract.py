"""Stage contract as data. Contracts document section 6. D24.

A stage declares what it requires, what it guarantees, when it declines,
which earlier stages it depends on, and whether it alters data. Stages 0–4
cannot declare that they alter data.

No stage logic lives here.
"""

from __future__ import annotations

from dataclasses import dataclass

# Stages 0–10 inclusive. D6, D23. Stages 0–4 touch no data. D24.
_STAGE_MIN = 0
_STAGE_MAX = 10
_READ_ONLY_LAST = 4


def _nonempty(text: str) -> bool:
    return isinstance(text, str) and text.strip() != ""


def _text_tuple(value: object, what: str) -> tuple[str, ...]:
    if type(value) is not tuple or not all(isinstance(item, str) and _nonempty(item) for item in value):
        raise TypeError(f"{what} must be a tuple of non-empty strings")
    return value


@dataclass(frozen=True)
class StageRequires:
    """Entry conditions from contracts section 6.

    The three texts are the stage author's declaration, not a guessed check.
    ``preconditions`` may be empty when the author declares none.
    """

    units: str
    completeness: str
    preconditions: tuple[str, ...]

    def __post_init__(self) -> None:
        if not _nonempty(self.units):
            raise ValueError("requires.units must be a non-empty string")
        if not _nonempty(self.completeness):
            raise ValueError("requires.completeness must be a non-empty string")
        _text_tuple(self.preconditions, "requires.preconditions")


@dataclass(frozen=True)
class StageGuarantees:
    """Exit claims from contracts section 6. Each text is required."""

    row_count: str
    units: str
    validity: str

    def __post_init__(self) -> None:
        if not _nonempty(self.row_count):
            raise ValueError("guarantees.row_count must be a non-empty string")
        if not _nonempty(self.units):
            raise ValueError("guarantees.units must be a non-empty string")
        if not _nonempty(self.validity):
            raise ValueError("guarantees.validity must be a non-empty string")


@dataclass(frozen=True)
class StageContract:
    """One stage's declared contract. Contracts section 6, D24.

    ``alters_data`` must be False for stages 0–4. ``depends_on_stages`` lists
    earlier stage numbers only, so the run order cannot cycle.
    """

    stage_number: int
    name: str
    requires: StageRequires
    guarantees: StageGuarantees
    declines_when: tuple[str, ...]
    depends_on_stages: tuple[int, ...]
    alters_data: bool

    def __post_init__(self) -> None:
        if type(self.stage_number) is not int or not _STAGE_MIN <= self.stage_number <= _STAGE_MAX:
            raise ValueError("stage_number must be an int from 0 to 10")
        if not _nonempty(self.name):
            raise ValueError("name must be a non-empty string")
        if not isinstance(self.requires, StageRequires):
            raise TypeError("requires must be a StageRequires")
        if not isinstance(self.guarantees, StageGuarantees):
            raise TypeError("guarantees must be a StageGuarantees")
        _text_tuple(self.declines_when, "declines_when")
        if type(self.depends_on_stages) is not tuple:
            raise TypeError("depends_on_stages must be a tuple of stage numbers")
        seen: set[int] = set()
        for number in self.depends_on_stages:
            if type(number) is not int or not _STAGE_MIN <= number <= _STAGE_MAX:
                raise ValueError("depends_on_stages entries must be stage numbers")
            if number >= self.stage_number:
                raise ValueError("A stage can depend only on an earlier stage.")
            if number in seen:
                raise ValueError(f"Stage {number} is listed twice in depends_on_stages.")
            seen.add(number)
        if type(self.alters_data) is not bool:
            raise TypeError("alters_data must be a bool")
        if self.stage_number <= _READ_ONLY_LAST and self.alters_data:
            raise ValueError("Stages 0–4 must declare alters_data=False (D24).")
