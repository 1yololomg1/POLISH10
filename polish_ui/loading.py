"""Run the engine for the interface, and record one decision. D98, D47, D48, D79.

The interface keeps no decision state of its own. A decision goes through
polish.decisions, is saved beside the well (D47), and the well is run again.
Only this-well decisions (THIS_ITEM) are recorded here. "All wells" needs the
scope preview of UXC4, which is not built.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from polish.decisions import (
    Session,
    SessionNotWritten,
    apply_decision,
    choose_changed_file,
    save_session,
)
from polish.model.findings import FindingRecord
from polish.model.states import UserAction
from polish.pipeline.run import WellRun, run_well
from polish.stages.s00_ingest import IngestResult, UnclassifiedIngest

__all__ = ["LoadResult", "SessionNotWritten", "load_well", "record_decision"]


@dataclass(frozen=True)
class LoadResult:
    """What one load produced. Exactly one of ``run``, ``unclassified``, ``error`` is set.

    ``partial`` is the Stage 0 result carried by an unclassified halt.
    """

    path: Path
    run: WellRun | None
    unclassified: str | None
    partial: IngestResult | None
    error: str | None


def load_well(
    path: Path,
    *,
    app_data: Path | None = None,
    local_tables_directory: Path | None = None,
) -> LoadResult:
    """Run Stages 0-4 with the saved answers. Writes nothing (D98)."""

    file_path = Path(path)
    try:
        run = run_well(file_path, app_data=app_data, local_tables_directory=local_tables_directory)
    except UnclassifiedIngest as exc:
        return LoadResult(file_path, None, str(exc), exc.partial, None)
    except (OSError, ValueError, TypeError) as exc:
        return LoadResult(file_path, None, None, None, f"{type(exc).__name__}: {exc}")
    return LoadResult(file_path, run, None, None, None)


def record_decision(
    session: Session,
    finding: FindingRecord,
    action: UserAction,
    *,
    reason: str = "",
    chosen_option: str | None = None,
    family_id: str | None = None,
    user_label: str | None = None,
    role: str | None = None,
    app_data: Path | None = None,
) -> Path:
    """Record one this-well decision and save the session beside the well. D47, D48.

    The changed-file question is answered through ``choose_changed_file``,
    which keeps the old session file before the new one is written.

    Returns
    -------
    Path
        The session file that was written.

    Raises
    ------
    ValueError
        The engine refused the action, option, family, or role.
    SessionNotWritten
        The session could not be written. Its finding says why.
    """

    if finding.check == "session_file_changed" and action is UserAction.CHOOSE_OPTION:
        if chosen_option is None:
            raise ValueError("CHOOSE_OPTION requires a chosen option.")
        loaded = choose_changed_file(session, chosen_option)
        return save_session(loaded.session)
    updated = apply_decision(
        session,
        finding,
        action,
        reason=reason,
        chosen_option=chosen_option,
        family_id=family_id,
        user_label=user_label,
        role=role,
        app_data=app_data,
    )
    return save_session(updated)
