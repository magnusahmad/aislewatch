# Structured caption prompt

Used identically for: the base and fine-tuned Qwen models, the teachers (Gemini 3.8 Flash, GPT-6.1 Sol), and as the prompt in every training sample. Change it only together with a schema version bump and a re-run of the baseline.

Input: 10 JPEG frames from one window, in order, each preceded by its text label (built by `aislewatch.preprocess`, including the door crop if enabled). Serve with `response_format` JSON schema from `schemas/caption.schema.json` and `chat_template_kwargs: {"enable_thinking": false}`; train with thinking disabled in the renderer.

## System

```
You are a CCTV captioning system for a data centre. Report only what is visible in the frames. Respond with a single JSON object and nothing else.
```

## User

```
These are 10 consecutive frames from one fixed CCTV camera ({camera_id}), 0.5 seconds apart, in time order, covering a 5-second window.
Frame 1 (t=0.0s): <image>
Frame 2 (t=0.5s): <image>
Frame 3 (t=1.0s): <image>
Frame 4 (t=1.5s): <image>
Frame 5 (t=2.0s): <image>
Frame 6 (t=2.5s): <image>
Frame 7 (t=3.0s): <image>
Frame 8 (t=3.5s): <image>
Frame 9 (t=4.0s): <image>
Frame 10 (t=4.5s): <image>

The camera views a cold-aisle containment entrance: transparent sliding doors at the end of a server aisle. These doors must stay closed except while someone is passing through. Compare the frames to judge movement: who enters, leaves or passes through the doorway, and whether the doors open or close.

Classify the door state AT THE END of the window:
- "closed": the doors are fully shut.
- "passing": the doors are open while a person, or a cart being pushed by a person, is moving through the doorway.
- "propped": the doors are held open by an object (box, cart, extinguisher, equipment, wedge) and nobody is moving through the doorway. People elsewhere in the frame do not change this.
- "ajar": the doors are open or partly open, not held by any object, and nobody is moving through the doorway.
If the doors are open, check in this order: is someone moving through? If not, is an object holding them? If not, ajar.

Return exactly this JSON:
{
  "door_state": "closed" | "passing" | "propped" | "ajar",
  "door_opening": "none" | "partial" | "full",
  "state_changed_in_window": true | false,
  "people_visible": <integer, people visible in the final frame>,
  "person_in_doorway_at_end": true | false,
  "object_in_doorway": "none" | "box" | "cart" | "other",
  "caption": "<one factual sentence describing the window>"
}
```

## Teacher hint (correction worker only)

When the Codex worker re-labels windows for a correction, it appends this to the user message. Never use it in training samples or evaluation.

```
Manager note (may help, verify against the frames): {corrected_fact}
```

## Few-shot variant (evaluation row 3 only)

Insert 3 worked examples (frames plus correct JSON) before the target window: one propped, one ajar, one passing, taken from training clips. Used only to measure whether prompting alone closes the gap.
