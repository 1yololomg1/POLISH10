Suggested by the external review (15 Sept) and adopted. Every line must be true before P05 is sent. Commit this list with P02 as `docs/governing/ready_for_P05.md`.

- [x] Unit conversion semantics approved (OD-05)
- [x] Child-dataset semantics approved (OD-03) — **resampling only**; fills, derivations and smoothing are overlays on the active dataset
- [x] State schema approved (OD-04, Appendix A)
- [x] Clean test baseline recorded (P01: 56 passed, ruff clean)
- [x] **Precedence rule explicitly approved by you** (OD-01)
- [x] **UXC adoption explicitly approved by you** (OD-02)
- [x] Backup made of the four untracked fixtures and the governing documents, outside the repo, before P02 moves anything. **Copied 24 Sept 2026** to `C:\porfolio\POLISH10_backup_2026-09-24` (`docs/governing/` and the four wells: `34_10-19.las`, `30_3-5 S.las`, `159-19A_LFP.las`, `WLC_PETRO_COMPUTED_INPUT_1_15_9-F-1B.las`). SHA-256 matched.
- [x] Fixture licences checked; fixtures committed with LFS (P02). FORCE 2020 wells (`34/10-19`, `30/3-5S`) are clear under NLOD 2.0 / CC-BY-4.0. The two Volve wells' commercial-use status is unresolved.
- [x] Legacy agent rules archived; only `polish-rebuild.mdc` and `aikido_rules.mdc` active (P02)
- [x] Aikido decision made (OD-09): keep it
- [x] P03 reconciliation reviewed with Claude
- [x] **Document fixes committed:** brief §3 (competitor claim), §4.1 (overlay vs D43), §4.2 (unit conversion is not "changed"; no "refused" sample state), §4.8 (D86 block types, no override for structural or conformance blocks), §5 ("Stages 0–10"), §7.1 (domain expert's track order); register: D21 citation, source tally, reviewer-notes count, "147" in D82, D62 "forensic"; contracts: "Stages 0–10", Phase numbering retired
- [x] Needs explicit yes/no confirmation from Alvaro before D12's second half can be closed. **Resolved 23 Sept: YES — processed STOP is rewritten to STRT+(n−1)×S_corrected when the well is corrected (D12 Part Two).**
