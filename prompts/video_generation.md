# Video generation prompts

Method: one base still per camera → edited stills per door state → first/last-frame image-to-video between them (Veo 3.1 main; Kling fallback; Wan for cheap tests). Chain 8–10 s generations into 30–60 s clips with state changes. Add timestamp overlays afterwards with ffmpeg. Make 2–3 takes per segment; keep only takes where a person can tell the door state.

## Base stills

**Camera A (training): high corner, along the aisle**
```
CCTV security camera still. Fixed high-angle wide shot from a ceiling corner, looking along a cold aisle in a modern colocation data centre. Two rows of black server racks with small green and blue status LEDs on both sides, perforated grey raised-floor tiles, overhead yellow cable trays, white ceiling with LED panel lights. At the near end of the aisle, a cold-aisle containment entrance: a pair of transparent polycarbonate sliding doors in a grey aluminium frame, with a small sign reading "COLD AISLE – KEEP DOORS CLOSED". The doors are fully closed. Slightly desaturated colour, mild wide-angle distortion, soft surveillance-camera image quality, 16:9. No people.
```

**Camera C (training): side view across the doorway, warmer and dimmer**
```
CCTV security camera still. Fixed camera mounted high on a wall in a data centre corridor, looking side-on across the entrance of a cold-aisle containment: a pair of transparent polycarbonate sliding doors in a grey aluminium frame set into a containment wall, server racks visible through the doors. Dimmer, slightly warm lighting, mild noise, surveillance-camera quality, 16:9. Doors fully closed. No people.
```

**Camera B (test and demo only): inside the aisle, looking back at the doors**
```
CCTV security camera still. Fixed camera mounted high inside a data centre cold aisle, looking back toward the aisle's entrance: a pair of transparent polycarbonate sliding containment doors in a grey aluminium frame, with the white-floored corridor visible beyond them. Server racks with status LEDs line both sides of the frame. Desaturated surveillance look, slight noise, 16:9. Doors fully closed. No people.
```

## State edits (apply to each base still)

- `Edit: the sliding doors are fully open. Nothing else changes.`
- `Edit: the left sliding door is held fully open by a brown cardboard box on the floor in its track. No people. Nothing else changes.`
- `Edit: a grey equipment cart is parked in the doorway, holding the doors open. No people. Nothing else changes.`
- `Edit: the left sliding door is open about 30 cm, not fully closed. No people. Nothing else changes.`
- Variants for hard cases: smaller wedge or fire extinguisher as the prop; box partly hidden by the door frame; dimmer lighting.

## Video prompt template

```
Keep the scene exactly as in the first frame. Static, locked-off CCTV camera: no camera movement, no zoom, no cuts. [ACTION]. Realistic security-camera footage, natural walking speed, [8|10] seconds.
```

## Segment actions (combine into clips)

| Segment | First → last still | ACTION |
|---|---|---|
| idle-closed | closed → closed | The aisle is empty; only the rack LEDs blink. |
| walk-past | closed → closed | A technician in a dark polo walks past outside the closed doors without stopping. |
| work-inside | closed → closed | A technician works at a rack inside the aisle; the doors stay closed. |
| pass-in | closed → closed | A technician slides the door open, walks into the aisle, and slides it shut behind them. |
| cart-through | closed → open | A technician pushes a grey equipment cart through the doorway into the aisle. |
| hold-for-other | closed → open | One technician holds the door open while a second walks through. |
| prop-box | closed → box-propped | A technician slides the door open, sets a cardboard box in the track so it holds the door open, and walks away. |
| leave-cart | closed → cart-propped | A technician pushes a cart into the doorway, leaves it there and walks off. |
| propped-idle | box-propped → box-propped | Empty scene; the door stays held open by the box. |
| propped-person-near | box-propped → box-propped | A technician works at a rack inside the aisle, not moving toward the door; the door stays propped. |
| drift-ajar | closed → ajar | A technician walks out through the door and leaves; the door slides back but stops about 30 cm short of closing. |
| ajar-idle | ajar → ajar | Empty scene; the door stays slightly open. |
| ajar-guard-misses | ajar → ajar | A guard in a high-visibility vest walks past outside without noticing the gap. |
| close-up | box-propped → closed | A technician returns, picks up the box and slides the door shut. |

## Demo story clips (camera B, not in the test set)

1. **Box incident:** idle-closed → prop-box → propped-idle (≥ 40 s) → close-up.
2. **Ajar incident:** pass-in → drift-ajar → ajar-idle (≥ 40 s) → ajar-guard-misses.
3. **Normal traffic:** walk-past → cart-through → pass-in → idle-closed (no alert expected).
