# Generic caption prompt ("before" baseline)

Used only for evaluation row 1 and the demo's opening. Measures how often a model mentions the door's state at all when it isn't asked to.

Same 10 labelled frames as the structured prompt, then:

```
These are 10 consecutive frames from one fixed CCTV camera ({camera_id}) in a data centre, 0.5 seconds apart, covering 5 seconds. Describe what happens in one or two sentences.
```

Scoring: a caption "mentions the door state" if it says the door is left open, propped, held, ajar or not closed (or clearly says it is closed). Score with a simple keyword check, then spot-check by hand.
