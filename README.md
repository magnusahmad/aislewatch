# AisleWatch

A small open vision-language model watches data-centre aisle cameras and writes structured captions. A Hermes agent investigates alerts with a frontier VLM. Manager corrections are turned into training data, and periodic LoRA runs on Prime Intellect bring the small model up to frontier accuracy at a fraction of the cost.

Built for Paradigm Frontiers 2026. Training footage is AI-generated, with a small real-footage test slice; the client is anonymized.

- **Build spec:** [docs/SPEC.md](docs/SPEC.md). Start here.
- Caption contract: [schemas/caption.schema.json](schemas/caption.schema.json) and [prompts/caption_structured.md](prompts/caption_structured.md)
- Footage generation: [prompts/video_generation.md](prompts/video_generation.md)
- Correction loop: [workers/correction_worker.md](workers/correction_worker.md)
- Hermes skills: [hermes/skills/](hermes/skills/)

Status: spec only; no code yet. Build order is in SPEC §11, starting with the three smoke tests and the perception probe.
