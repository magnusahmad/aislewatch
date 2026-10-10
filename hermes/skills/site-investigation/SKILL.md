---
name: site-investigation
description: Answer what happened at a data-centre site and investigate what led up to an alert or sensor alarm, using AisleWatch captions, alerts, sensor data and the frontier VLM on original footage.
---

# Site investigation

Use the `aislewatch` MCP tools. Always cite window IDs and frame times.

## "What happened today / in this period?"

1. `get_alerts` for the period.
2. For each camera, `get_timeline` for the period.
3. Summarise as a short timeline: time, camera, what happened, window IDs. Group routine passing traffic into one line per camera; list every propped/ajar span and every alert individually.

## "What led up to X?" / "What caused this alarm?"

1. Identify the event: `get_alerts` or `get_sensor` for the aisle and time.
2. Pull the timeline for the 15 minutes before the event on every camera covering that aisle.
3. Note what the captions say, and what they do *not* say (e.g. no door state reported).
4. Call `inspect_clip` on the most suspicious span (or the 10 minutes before the event if nothing stands out) with a specific question, e.g. "Is the containment door open or held open by anything between these times? Who opened it and did they leave?"
5. Answer with:
   - **What the captions show** (window IDs)
   - **What the footage check confirmed** (frame times from `inspect_clip`)
   - **Most likely cause**, and what is still uncertain.

Do not state a cause as fact unless `inspect_clip` confirmed it or the manager did. Check the `site-rules` skill, if it exists, before concluding whether something is a violation.
