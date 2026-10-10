---
name: corrections
description: When the site manager corrects something you said about the site or states a site rule, record it so the captioning model is retrained, and save rules as a site-rules skill.
---

# Handling manager corrections

Trigger: the manager says you were wrong about what happened, gives a fact about an event, or states a rule ("doors must never be propped", "flag anything over two minutes").

1. Restate the corrected fact in one line, with camera and times where known, e.g. "Camera B door propped by a box from 14:02 to 14:31."
2. Call `record_correction` with:
   - `refs`: the alert ID, window IDs or camera + time span you were discussing
   - `manager_text`: the manager's words, verbatim
   - `hermes_claim`: what you said that was wrong
   - `corrected_fact`: your one-line restatement
   - `is_rule`: true if a rule was stated
3. If a rule was stated, create or update the `site-rules` skill with the rule in one plain sentence, plus the threshold if one was given. Keep one rule per line; don't duplicate existing rules.
4. Reply in one or two sentences: what you recorded, and (if relevant) the rule you saved. Mention that the captioning model will learn from it at the next training run; `loop_status` shows progress.

Do not argue with a correction about what happened. If it conflicts with what `inspect_clip` showed, record it anyway and mention the conflict in `corrected_fact`; the review queue will resolve it.
