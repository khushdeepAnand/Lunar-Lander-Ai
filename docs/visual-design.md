# Visual design

Mission-control palette defined in `.streamlit/config.toml`: near-black navy
`#091522` background, `#142638` panels, ice cyan `#55D6E9` primary, near-white
`#EAF3F7` text. Charts reserve green for measured safe outcomes, red for crashes,
amber for exploration, and cyan for returns and loss. Streamlit native controls,
Material Symbols, bordered KPI cards and a fixed left navigation keep keyboard
interaction and responsive layouts consistent. Text and labels accompany color.

The simulation viewport renders Gymnasium frames; no overlay claims invented
physics or predicted trajectories. Charts use explicit units and axes. Where
there is no evidence, an empty state replaces a zero or fabricated result.

`docs/assets/landing_40400.gif` predates the dashboard and is identified as a
recorded episode. `docs/assets/dashboard-overview.png` and
`docs/assets/dashboard-simulation.png` and
`docs/assets/dashboard-failure-gallery.png` were captured from the running
local Streamlit application in an actual browser. The simulation screenshot
shows a real environment frame at step 189, native telemetry and the four
network Q-values for the preceding decision state. Failure strips show actual
frames captured during a separate seeded evaluation, not model predictions.

## Stage 3 visual layer (mission styling, HUD, arena)

- `ui/theme.py` holds the single CSS block (palette tokens, compact type scale, status pills,
  telemetry tiles) plus the animated mission banner. Motion is CSS only: a pulsing LIVE dot, a
  slow starfield drift, flowing arrows in the learning-loop diagram, smooth dial and Q-bar
  transitions. Nothing animates simulation state; physics frames come from Gymnasium.
- `ui/hud.py` draws overlays on the real RGB frame: landing-pad span, observed trajectory
  trail, last-command badge. Pixel mapping follows LunarLander-v3's own constants and is checked
  against Box2D in `tests/test_hud.py`. The "Mission styling" toggle recolors only exact-black
  sky and exact-white terrain (cosmetic); lander, flags and particles are untouched.
- The attitude dial shows the observed angle (exact radians-to-degrees conversion) and leg
  contacts. The Q-bars show each action's gap to the network's best action for the state before
  the last step. They are estimated returns, not confidence, and bar length is relative to the
  spread with a floor of 1.0 so tiny gaps are not exaggerated.
- The Policy arena replays up to three trained `best.pt` checkpoints on one seed. One seed is
  an anecdote; statistics live on the analytics and experiment pages.
- Screenshots in `docs/assets/dashboard-*.png` predate this layer and must be retaken from the
  running app before publishing.
