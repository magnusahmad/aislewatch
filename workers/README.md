# Workers

## Codex correction worker

`aw worker run` polls the `corrections` table every 30 s and, for each new correction, runs:

```
codex exec -C <repo-root> "Process correction <id>. Follow workers/correction_worker.md exactly."
```

Before first use, confirm on the installed Codex version and record here:

- the non-interactive sandbox/approval flags that allow writes only under `data/` plus network access for the teacher APIs (via `aw`);
- that it runs on the Codex subscription login (`codex login`), not an API key;
- typical run time and subscription usage per correction.

Run one correction at a time.

Flags confirmed: _(fill in after smoke test S3)_
