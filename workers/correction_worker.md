# Correction worker instructions (for Codex)

You are processing one manager correction for the AisleWatch captioning system. Use only the `aw` CLI. Do not read or write the SQLite database directly, and do not edit prompts, schemas, configs or code.

Input: a correction ID, given in the task.

## Steps

1. **Load it.** `aw correction show <id>`. Set status: `aw correction set-status <id> processing`.

2. **Find affected windows.** `aw windows list --camera <cam> --from <t0 - 60s> --to <t1 + 60s>`. If the correction references an alert, use `aw alert show <alert_id>` to get the camera and span.

3. **Compare current captions.** `aw captions show --windows <ids> --models qwen-current,qwen-base`.

4. **Re-label with the teachers.** `aw teacher label --windows <ids> --hint "<corrected_fact>"`. This calls Gemini 3.8 Flash and GPT-6.1 Sol and stores both votes.

5. **Decide which layer failed** and write it down:
   - **caption:** the current model's `door_state` (or another field) disagrees with the teacher-agreed label in the span.
   - **reasoning:** captions match the teacher-agreed labels, but Hermes's claim was still wrong.
   - **both**, or **unclear**.

6. **Find similar windows** (only if a caption error was found):
   `aw search similar --window <a mislabelled window> --limit 40` and `aw search disagreements --since 24h --state <wrong state> --limit 40`.
   Then `aw teacher label --windows <those ids>` (no hint: the hint applies only to the corrected span).

7. **Write labels.** For every labelled window:
   - Both teachers agree, and in the corrected span the label is consistent with the manager's statement → `aw label accept --window <id> --from-teacher --correction <id>`.
   - Teachers disagree, or the teachers contradict the manager → `aw review enqueue --window <id> --correction <id> --reason "<short reason>"`.
   - Never accept a label that contradicts the manager. Never label a window whose split is `test` (`aw` will refuse; do not retry).

8. **Report.** Write `data/reports/<correction_id>.md` with:
   - the correction, verbatim
   - layer that failed, with the evidence (window IDs, captions vs. teacher labels)
   - counts: windows labelled, accepted, queued for review
   - suggested caption-field changes, if any, clearly marked as **suggestion for human approval**
   - anything surprising (e.g. teachers disagreeing with each other often on one camera)

   Then: `aw correction set-status <id> done --report data/reports/<correction_id>.md`.

## If something fails

Set `aw correction set-status <id> failed --report <path>` with what happened. Do not start training, change the current adapter or modify code to work around a failure.
