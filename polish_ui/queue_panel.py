"""Findings queue: a side panel, never a dialog. D79, D81, D86, UXC2.

Every item is shown the same way, in the same place: what POLISH observed,
what it means, what you may do, what happens next. Only the actions the
engine permits and can carry out for that finding are offered.
"""

from __future__ import annotations

import html

from PySide6 import QtCore, QtWidgets

from polish.decisions import Session
from polish.decisions.apply import current_decisions, listed_options
from polish.model.dataset import Curve
from polish.model.findings import FindingRecord
from polish.model.mapping import roles_stated_by_family
from polish.model.states import BlockType, IssueDisposition, UserAction, permitted_actions
from polish.tables.loader import load_shipped_tables
from polish_ui import wording
from polish_ui.evidence_text import render_evidence

# The family box starts here for every finding and every well. Record is
# refused until the user moves off it (P18.6).
CHOOSE_FAMILY = "Choose a family"

_FINDING_ROLE = QtCore.Qt.ItemDataRole.UserRole


def usable_actions(finding: FindingRecord) -> tuple[tuple[UserAction, ...], tuple[str, ...]]:
    """Actions to offer for one finding, and the reason any permitted one is left out.

    The permitted set is ``permitted_actions`` (the glossary matrix).
    CHOOSE_OPTION is left out when the finding lists no options, and
    MAP_EXPLICITLY when the finding carries no mnemonic, because the engine
    refuses both. The reason is shown to the user. Nothing is hidden silently.
    """

    permitted = permitted_actions(finding.block_type)
    evidence = finding.evidence.as_dict()
    offered: list[UserAction] = []
    notes: list[str] = []
    for action in wording.ACTION_ORDER:
        if action not in permitted:
            continue
        if action is UserAction.CHOOSE_OPTION and not listed_options(finding):
            notes.append(wording.NO_OPTIONS)
            continue
        if action is UserAction.MAP_EXPLICITLY and not isinstance(evidence.get("mnemonic"), str):
            notes.append(wording.NO_MNEMONIC)
            continue
        offered.append(action)
    return tuple(offered), tuple(notes)


class QueuePanel(QtWidgets.QWidget):
    """The queue list, the item detail, and the inline action form."""

    decision_requested = QtCore.Signal(object, object, dict)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._session: Session | None = None
        self._mnemonics: dict[int, str] = {}
        self._shipped = load_shipped_tables()
        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.currentItemChanged.connect(self._show_current)
        self.detail = QtWidgets.QTextBrowser()
        self.detail.setOpenLinks(False)
        self.action_box = QtWidgets.QComboBox()
        self.action_box.currentIndexChanged.connect(self._show_fields)
        self.option_box = QtWidgets.QComboBox()
        self.family_box = QtWidgets.QComboBox()
        self.family_box.addItem(CHOOSE_FAMILY)
        self.family_box.addItems(sorted(record.id for record in self._shipped.families))
        self.family_box.currentIndexChanged.connect(self._show_roles)
        self.role_box = QtWidgets.QComboBox()
        self.label_edit = QtWidgets.QLineEdit()
        self.label_edit.setPlaceholderText("Your own name for the curve (optional)")
        self.reason_edit = QtWidgets.QLineEdit()
        self.reason_edit.setPlaceholderText("Reason (required)")
        self.record_button = QtWidgets.QPushButton("Record for this well")
        self.record_button.clicked.connect(self._request)
        self.message = QtWidgets.QLabel("")
        self.message.setWordWrap(True)

        self.form = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(self.form)
        form.setContentsMargins(0, 0, 0, 0)
        form.addRow("Action", self.action_box)
        form.addRow("Option", self.option_box)
        form.addRow("Family", self.family_box)
        form.addRow("Role", self.role_box)
        form.addRow("Label", self.label_edit)
        form.addRow("Reason", self.reason_edit)
        form.addRow(self.record_button)
        self._form_layout = form

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QtWidgets.QLabel("Findings"))
        layout.addWidget(self.tree, 2)
        layout.addWidget(self.detail, 3)
        layout.addWidget(self.form)
        layout.addWidget(self.message)
        self._show_current(None, None)

    def set_items(
        self,
        findings: tuple[FindingRecord, ...],
        session: Session | None,
        curves: tuple[Curve, ...] = (),
    ) -> None:
        """Show these findings. ``session`` is None when no decision can be recorded.

        ``curves`` are the loaded dataset's curves, used to name a curve
        reference by column and mnemonic. The family, role, option, label and
        reason fields are cleared for the new file (P18.6).
        """

        self._session = session
        self._mnemonics = {curve.ref.column_index: curve.original_mnemonic for curve in curves}
        self.form.setEnabled(True)
        current = {} if session is None else current_decisions(session.decisions, session.held_ids)
        self.tree.clear()
        self.message.setText("")
        for block in BlockType:
            members = [item for item in findings if item.block_type is block]
            if not members:
                continue
            names = {wording.STAGE_NAMES[item.stage] for item in members}
            stage_name = names.pop() if len(names) == 1 else wording.STAGE_NAMES[members[0].stage]
            heading = wording.block_sentence(block, stage_name)
            group = QtWidgets.QTreeWidgetItem([f"{block.value} — {heading} ({len(members)})"])
            self.tree.addTopLevelItem(group)
            for finding in members:
                decision = current.get(finding.id)
                state = IssueDisposition.UNDISPOSITIONED if decision is None else decision.disposition.issue_disposition
                child = QtWidgets.QTreeWidgetItem([f"[{state.value}] Stage {finding.stage}: {finding.message}"])
                child.setData(0, _FINDING_ROLE, (finding, state))
                group.addChild(child)
            group.setExpanded(True)
        self._show_current(None, None)

    def findings_shown(self) -> list[FindingRecord]:
        """The findings in the list, in display order."""

        found: list[FindingRecord] = []
        for index in range(self.tree.topLevelItemCount()):
            group = self.tree.topLevelItem(index)
            for row in range(group.childCount()):
                found.append(group.child(row).data(0, _FINDING_ROLE)[0])
        return found

    def select_finding(self, finding_id: str) -> bool:
        """Select the item with this finding id. False when it is not listed."""

        for index in range(self.tree.topLevelItemCount()):
            group = self.tree.topLevelItem(index)
            for row in range(group.childCount()):
                child = group.child(row)
                if child.data(0, _FINDING_ROLE)[0].id == finding_id:
                    self.tree.setCurrentItem(child)
                    return True
        return False

    def offered_actions(self) -> list[str]:
        """The action tokens the form currently offers."""

        return [self.action_box.itemData(index).value for index in range(self.action_box.count())]

    def show_message(self, text: str) -> None:
        """Show a refusal or a save failure inline. No dialog is opened."""

        self.message.setText(text)

    def set_form_enabled(self, enabled: bool, reason: str = "") -> None:
        """Let the form take input, or refuse it while a load runs. P21.7, D47.

        ``reason`` is one line, shown under the form when input is refused.
        """

        self.form.setEnabled(enabled)
        if not enabled:
            self.message.setText(reason)

    def _current(self) -> tuple[FindingRecord, IssueDisposition] | None:
        item = self.tree.currentItem()
        if item is None:
            return None
        return item.data(0, _FINDING_ROLE)

    def _show_current(self, _now: object, _before: object) -> None:
        selected = self._current()
        self.action_box.blockSignals(True)
        self.action_box.clear()
        self.action_box.blockSignals(False)
        self.message.setText("")
        self._reset_choices()
        if selected is None:
            self.detail.setHtml("<p>Select a finding.</p>")
            self.form.setVisible(False)
            return
        finding, state = selected
        offered, notes = usable_actions(finding)
        if self._session is None:
            offered, notes = (), ("No session is open for this file, so no decision can be recorded.",)
        may_do: list[str] = [action.value for action in offered] + list(notes)
        if not permitted_actions(finding.block_type):
            may_do = [wording.VIEW_ONLY]
        evidence = render_evidence(finding.evidence.fields, self._mnemonics)
        evidence_html = "".join(f"<br>{html.escape(line)}" for line in evidence.split("\n") if line)
        results = "; ".join(f"{action.value} → {wording.ACTION_RESULT[action].value}" for action in offered)
        meaning = wording.block_sentence(finding.block_type, wording.STAGE_NAMES[finding.stage])
        self.detail.setHtml(
            "<p><b>WHAT POLISH OBSERVED</b><br>"
            f"{html.escape(finding.message)}{evidence_html}</p>"
            "<p><b>WHAT IT MEANS</b><br>"
            f"{finding.block_type.value}: {html.escape(meaning)}</p>"
            "<p><b>WHAT YOU MAY DO</b><br>"
            f"{'<br>'.join(html.escape(line) for line in may_do)}</p>"
            "<p><b>WHAT HAPPENS NEXT</b><br>"
            f"Current disposition: {state.value}.<br>"
            f"{html.escape(wording.EXPORT_EFFECT[finding.block_type])}."
            f"{'<br>' + html.escape(results) if results else ''}</p>"
        )
        self.form.setVisible(bool(offered))
        if not offered:
            return
        self.action_box.blockSignals(True)
        for action in offered:
            self.action_box.addItem(action.value, action)
        self.action_box.blockSignals(False)
        self.option_box.addItems(list(listed_options(finding)))
        self._show_roles()
        self._show_fields()

    def _reset_choices(self) -> None:
        """Clear the entry fields. The family box returns to "Choose a family". P18.6."""

        self.family_box.blockSignals(True)
        self.family_box.setCurrentIndex(0)
        self.family_box.blockSignals(False)
        self.option_box.clear()
        self.role_box.clear()
        self.label_edit.clear()
        self.reason_edit.clear()

    def _set_row_visible(self, widget: QtWidgets.QWidget, visible: bool) -> None:
        self._form_layout.setRowVisible(widget, visible)

    def _show_fields(self) -> None:
        action = self.action_box.currentData()
        if action is None:
            return
        mapping = action is UserAction.MAP_EXPLICITLY
        self._set_row_visible(self.option_box, action is UserAction.CHOOSE_OPTION)
        self._set_row_visible(self.family_box, mapping)
        self._set_row_visible(self.role_box, mapping and self.role_box.count() > 1)
        self._set_row_visible(self.label_edit, mapping)
        self._set_row_visible(self.reason_edit, action is not UserAction.ACKNOWLEDGE)

    def _show_roles(self) -> None:
        family = self.family_box.currentText()
        self.role_box.clear()
        if family == CHOOSE_FAMILY:
            if self.action_box.currentData() is UserAction.MAP_EXPLICITLY:
                self._set_row_visible(self.role_box, False)
            return
        roles = roles_stated_by_family(self._shipped, family)
        self.role_box.addItems([role.value for role in roles])
        if self.action_box.currentData() is UserAction.MAP_EXPLICITLY:
            self._set_row_visible(self.role_box, self.role_box.count() > 1)

    def _request(self) -> None:
        selected = self._current()
        action = self.action_box.currentData()
        if selected is None or action is None:
            return
        fields: dict[str, str | None] = {}
        if action is not UserAction.ACKNOWLEDGE:
            fields["reason"] = self.reason_edit.text().strip()
        if action is UserAction.CHOOSE_OPTION:
            fields["chosen_option"] = self.option_box.currentText()
        if action is UserAction.MAP_EXPLICITLY:
            if self.family_box.currentText() == CHOOSE_FAMILY:
                self.show_message("Choose a family before recording.")
                return
            fields["family_id"] = self.family_box.currentText()
            label = self.label_edit.text().strip()
            fields["user_label"] = label if label else None
            if self.role_box.count() > 1:
                fields["role"] = self.role_box.currentText()
        self.decision_requested.emit(selected[0], action, fields)
