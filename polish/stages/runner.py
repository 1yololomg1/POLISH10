"""Run registered stages in order, and stop a stage that breaks its contract.

Implements contracts section 6, D24, C5, D17, D80. No stage logic is included.
The runner does not load a file and does not open the network.

``StageResult.status`` is ``states.StageStatus``. The statuses this runner
emits are NOT_RUN, COMPLETE, REFUSED_STRUCTURAL, and STALE. Those are members
of the existing enum, not a new one.

A stage that declares ``alters_data=False`` and changes the overlay or swaps
the dataset fails loudly. Writing a source array fails loudly for every stage,
because the original frame is never overwritten.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

from polish.model.dataset import Dataset
from polish.model.findings import FindingRecord
from polish.model.overlay import Overlay
from polish.model.states import StageStatus
from polish.stages.contract import StageContract

# Statuses a stage function may report. NOT_RUN and STALE belong to the runner.
_STAGE_REPORTED = frozenset({StageStatus.COMPLETE, StageStatus.REFUSED_STRUCTURAL})


class StageContractViolation(Exception):
    """A stage broke its own contract or wrote a source array. Contracts §6, C5, D24."""


@dataclass
class RunContext:
    """What a stage is given. The dataset's source arrays stay read-only. C5."""

    dataset: Dataset
    overlay: Overlay


@dataclass(frozen=True)
class StageOutcome:
    """What a stage returns when it actually runs.

    ``status`` is COMPLETE or REFUSED_STRUCTURAL. The runner sets NOT_RUN
    and STALE itself.
    """

    status: StageStatus
    findings: tuple[FindingRecord, ...] = ()

    def __post_init__(self) -> None:
        if self.status not in _STAGE_REPORTED:
            raise ValueError("A stage reports COMPLETE or REFUSED_STRUCTURAL.")
        if type(self.findings) is not tuple or not all(isinstance(item, FindingRecord) for item in self.findings):
            raise TypeError("findings must be a tuple of FindingRecord")


@dataclass(frozen=True)
class RegisteredStage:
    """A contract plus the two callables the runner needs.

    ``unmet_requirements`` returns human-readable reasons the declared
    requirements do not hold. An empty tuple means they hold. The runner
    also checks ``depends_on_stages`` itself.
    """

    contract: StageContract
    run: Callable[[RunContext], StageOutcome]
    unmet_requirements: Callable[[RunContext], tuple[str, ...]]

    def __post_init__(self) -> None:
        if not isinstance(self.contract, StageContract):
            raise TypeError("contract must be a StageContract")
        if not callable(self.run):
            raise TypeError("run must be callable")
        if not callable(self.unmet_requirements):
            raise TypeError("unmet_requirements must be callable")


@dataclass(frozen=True)
class StageResult:
    """One stage's recorded result. D80, D86, contracts §6.

    ``status`` is a ``states.StageStatus``. ``detail`` is why the stage did
    not run, or empty when it completed with nothing further to say.
    ``duration`` is seconds spent inside ``run``. It is 0 when the stage
    was not called.
    """

    stage_number: int
    status: StageStatus
    findings: tuple[FindingRecord, ...]
    duration: float
    detail: str

    def __post_init__(self) -> None:
        if not isinstance(self.status, StageStatus):
            raise TypeError("status must be a StageStatus")
        if type(self.findings) is not tuple:
            raise TypeError("findings must be a tuple")
        if isinstance(self.duration, bool) or not isinstance(self.duration, (int, float)):
            raise TypeError("duration must be a number of seconds")
        if self.duration < 0:
            raise ValueError("duration cannot be negative")
        if not isinstance(self.detail, str):
            raise TypeError("detail must be a str")


def _array_token(values: object) -> tuple[bytes, bool]:
    array = values
    return (array.tobytes(), bool(array.flags.writeable))


def _source_tokens(dataset: Dataset) -> tuple[tuple[bytes, bool], ...]:
    tokens = [_array_token(curve.values) for curve in dataset.curves]
    tokens.append(_array_token(dataset.grid.values))
    return tuple(tokens)


class Runner:
    """Run registered stages in stage-number order. Contracts §6, D24, C5.

    Parameters
    ----------
    stages:
        Each stage number appears once.
    """

    def __init__(self, stages: tuple[RegisteredStage, ...] | list[RegisteredStage]) -> None:
        if not isinstance(stages, (tuple, list)):
            raise TypeError("stages must be a tuple or list of RegisteredStage")
        registered: list[RegisteredStage] = []
        seen: set[int] = set()
        for stage in stages:
            if not isinstance(stage, RegisteredStage):
                raise TypeError("stages must contain RegisteredStage instances")
            number = stage.contract.stage_number
            if number in seen:
                raise ValueError(f"Stage {number} is registered twice.")
            seen.add(number)
            registered.append(stage)
        self._stages = tuple(sorted(registered, key=lambda stage: stage.contract.stage_number))

    def run(
        self,
        dataset: Dataset,
        overlay: Overlay | None = None,
        upstream_stale: frozenset[int] = frozenset(),
    ) -> tuple[StageResult, ...]:
        """Run every registered stage whose requirements hold.

        ``overlay`` defaults to an empty store when the caller has no overlay
        yet. ``upstream_stale`` lists stage numbers the caller has marked
        stale. An empty set means none were marked. A stale stage is not
        called, and a stage that depends on it is STALE as well (D80).

        Source arrays are compared before and after each call. A change is a
        ``StageContractViolation`` (C5, D17). A change to the overlay or the
        dataset object is a violation when ``alters_data`` is False (D24).
        """

        if not isinstance(dataset, Dataset):
            raise TypeError("dataset must be a Dataset")
        if overlay is None:
            overlay = Overlay()
        elif not isinstance(overlay, Overlay):
            raise TypeError("overlay must be an Overlay")
        if type(upstream_stale) is not frozenset or not all(type(number) is int for number in upstream_stale):
            raise TypeError("upstream_stale must be a frozenset of stage numbers")

        context = RunContext(dataset=dataset, overlay=overlay)
        results: dict[int, StageResult] = {}
        ordered: list[StageResult] = []
        for stage in self._stages:
            result = self._run_one(stage, context, results, upstream_stale)
            results[result.stage_number] = result
            ordered.append(result)
        return tuple(ordered)

    def _run_one(
        self,
        stage: RegisteredStage,
        context: RunContext,
        results: dict[int, StageResult],
        upstream_stale: frozenset[int],
    ) -> StageResult:
        number = stage.contract.stage_number
        reasons = self._blocked_reasons(stage, context, results, upstream_stale)
        if reasons:
            status = StageStatus.STALE if self._is_stale(stage, results, upstream_stale) else StageStatus.NOT_RUN
            return StageResult(number, status, (), 0.0, " ".join(reasons))

        source = context.dataset
        before_source = _source_tokens(source)
        before_overlay = context.overlay.snapshot()
        before_dataset_id = id(context.dataset)
        started = time.perf_counter()
        outcome = stage.run(context)
        duration = time.perf_counter() - started
        if not isinstance(outcome, StageOutcome):
            raise StageContractViolation(f"Stage {number} did not return a StageOutcome.")
        self._enforce_contract(stage, context, source, before_source, before_overlay, before_dataset_id)
        return StageResult(number, outcome.status, outcome.findings, duration, "")

    def _is_stale(
        self,
        stage: RegisteredStage,
        results: dict[int, StageResult],
        upstream_stale: frozenset[int],
    ) -> bool:
        number = stage.contract.stage_number
        if number in upstream_stale:
            return True
        return any(results[dependency].status is StageStatus.STALE for dependency in stage.contract.depends_on_stages)

    def _blocked_reasons(
        self,
        stage: RegisteredStage,
        context: RunContext,
        results: dict[int, StageResult],
        upstream_stale: frozenset[int],
    ) -> tuple[str, ...]:
        number = stage.contract.stage_number
        reasons: list[str] = []
        if number in upstream_stale:
            reasons.append(f"Stage {number} is STALE.")
        for dependency in stage.contract.depends_on_stages:
            if dependency not in results:
                raise StageContractViolation(
                    f"Stage {number} depends on stage {dependency}, which is not registered."
                )
            status = results[dependency].status
            if status is StageStatus.STALE:
                reasons.append(f"Stage {number} depends on stage {dependency}, which is STALE.")
            elif status is not StageStatus.COMPLETE:
                reasons.append(
                    f"Stage {number} depends on stage {dependency}, which is {status.value}."
                )
        reported = stage.unmet_requirements(context)
        if type(reported) is not tuple or not all(isinstance(item, str) and item.strip() for item in reported):
            raise StageContractViolation(
                f"Stage {number} unmet_requirements must return a tuple of non-empty strings."
            )
        reasons.extend(reported)
        return tuple(reasons)

    def _enforce_contract(
        self,
        stage: RegisteredStage,
        context: RunContext,
        source: Dataset,
        before_source: tuple[tuple[bytes, bool], ...],
        before_overlay: tuple[tuple[str, float, str, str], ...],
        before_dataset_id: int,
    ) -> None:
        number = stage.contract.stage_number
        if _source_tokens(source) != before_source:
            raise StageContractViolation(
                f"Stage {number} wrote a source array. The original frame is never overwritten (C5, D17)."
            )
        overlay_changed = context.overlay.snapshot() != before_overlay
        dataset_replaced = id(context.dataset) != before_dataset_id
        if not stage.contract.alters_data and (overlay_changed or dataset_replaced):
            raise StageContractViolation(
                f"Stage {number} declares alters_data=False and modified data (D24)."
            )
