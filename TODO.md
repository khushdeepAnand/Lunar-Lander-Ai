# Project checklist

Updated 2026-10-24. Checked boxes have real tests or experiment records in
`docs/experiments.md`. Unchecked boxes are **not** claimed complete.

## Completed

- [x] Set up Python 3.12 virtual environment on Windows; verify Box2D and GPU.
- [x] Implement discrete LunarLander environment, MLP, replay and vanilla DQN.
- [x] Handle termination and time-limit truncation separately in Bellman target.
- [x] Train native-reward baseline; save latest, best and final checkpoints.
- [x] Run 100 held-out evaluation episodes for best native checkpoint:
  **80/100 safe pad landings**, mean native return **252.91** (one training seed).
- [x] Render actual successful agent actions as committed GIF; generate charts
  from real CSV logs. Include small, provenance-documented checkpoint.
- [x] Test optional shaping with matched 300-episode native baseline; document
  **regression** rather than claim improvement.
- [x] Add automated tests, Ruff, setup instructions, portfolio and interview Q&A.
- [x] Initialize local Git repository with logical setup and implementation commits.
- [x] Run final test suite (**37 passed**), Ruff (**clean**) and whitespace check.
- [x] Commit documentation, measured experiments and small demo assets.
- [x] Create GitHub repository and push; choose public visibility.
- [x] Implement local Streamlit dashboard with live simulation, training lab,
  performance analytics, experiment comparison, and checkpoint inspection.
- [x] Implement reproducible multi-seed experiment runner with resume capability.

## Agent tasks remaining

- [ ] Run ≥5 separate training seeds (900 episodes each) and report distribution
  and confidence intervals. One seed cannot establish repeatability of learning.
  The multi-seed runner is implemented; the experiment has not been executed.
- [ ] Optionally profile long-run CPU vs CUDA speed and validate CPU-only install
  in a clean environment. CPU wheel exists but was not installed on this GPU PC.
- [ ] Optional exact resume: persist replay memory and Gymnasium RNG state.
  Existing checkpoints restore networks/optimizer/counters but **not** replay
  history; loading for evaluation works, exact training resume is not offered.

## Your personal tasks

- [x] Playback launched for you: `scripts/play.py --checkpoint
  examples/best_native.pt --seed 40400` opened a live pygame window and logged
  **3/3 safe landings** (seeds 40400-40402, mean native return 268.75).
- [x] Demo GIF regenerated at higher quality: 900x600, 179 frames, 20 fps,
  every second simulated frame, still 100% real environment frames.
- [ ] Eyeball the live window yourself once to confirm the picture looks right
  on your display.
- [ ] Review `docs/interview.md` and practice explaining Q-target truncation,
  checkpoint selection and why 16 rewarded endings are **not** safe landings.
- [ ] Decide whether to run multi-seed experiments before claiming reliable
  *training* (not just one good saved policy). They require more local compute.
- [x] Add project link to personal portfolio/resume. Use measured, qualified
  bullet points in `docs/portfolio.md`; do not claim guaranteed 80% on all
  retraining runs.