---
name: corrections
description: When the site manager corrects something you said about the site or states a site rule, record it so the captioning model is retrained, and save rules as a site-rules skill.
---

# Handling manager corrections

Trigger: the manager says you were wrong about what happened, gives a fact about an event, or states a rule ("doors must never be propped", "flag anything over two minutes").

1. Restate the corrected fact in one line, with camera and times where known, e.g. "Camera B door propped by a box from 14:02 to 14:31."
2. Call `record_correction` with:
   - `alert_id` or `camera` + `t0`/`t1`: what you were discussing
   - `manager_text`: the manager's words, verbatim
   - `hermes_claim`: what you said that was wrong
   - `corrected_fact`: your one-line restatement
   - `rule`: if the manager stated a site rule, that rule as one plain sentence with any threshold (e.g. "Containment doors must never be propped open; flag any door propped or ajar for more than 2 minutes."). The tool saves it to the `site-rules` skill.
3. Reply in one or two sentences: what you recorded, and (if a rule was saved) that it applies to your investigations from now on, while the camera captioning model learns from the correction at its next training run (`loop_status` shows progress).

Do not argue with a correction about what happened. If it conflicts with what `inspect_clip` showed, record it anyway and mention the conflict in `corrected_fact`; the review queue will resolve it.
