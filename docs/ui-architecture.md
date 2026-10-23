# Dashboard architecture

```text
Browser ↔ app.py (Streamlit navigation + per-session playback)
                ├─ ui/playback.py → existing checkpoint, agent, environment, metrics
                ├─ ui/jobs.py → scripts/train.py worker → train.py
                └─ ui/data.py → local CSV/JSON/checkpoint discovery → Plotly

scripts/evaluate.py worker → evaluate.py → results/*.json + results/*.csv
```

Streamlit session state owns one RGB-array environment per visitor; its fragment
advances **one real Gymnasium step** per rerun. Stop/reset or leaving the page
closes the old environment; returning starts a fresh episode on demand.

`TrainingJobs` owns a subprocess and logs stdout/errors. One job per local
Streamlit server can be active at a time. The worker emits one CSV row after each
completed episode and writes periodic evaluation and atomic model checkpoints.
An on-disk stop request is checked after an episode, then the worker saves the
final checkpoint and run status. Periodic evaluation rows are flushed when
available. A process that exits nonzero is reported failed, including a failure
before environment creation, with its configuration and error preserved.
Worker handles live in server memory; after a server restart, historical artifacts
remain visible, but the old process is not reattached. This is not distributed
job scheduling. A `final.pt` from a stopped job does not provide exact resume:
the replay buffer, environment state and full RNG progression are not restored.

The CLI remains independent: `scripts/train.py`, `scripts/evaluate.py`,
`scripts/play.py` and `scripts/plot_results.py` still work without Streamlit.
