"""HUD overlays must line up with the real simulator and never alter the frame size."""

import numpy as np
from ui import hud

from lunar_lander_ai.environment import make_env

ENV = {"id": "LunarLander-v3", "continuous": False, "max_episode_steps": 1000}


def test_pixel_mapping_matches_box2d_lander():
    """Observation -> pixel mapping agrees with the simulator's own hull position."""
    env = make_env(ENV, "rgb_array")
    try:
        obs, _ = env.reset(seed=7)
        for _ in range(40):
            obs, *_ = env.step(0)
        x_px, y_px = hud.to_pixels(obs)
        body = env.unwrapped.lander.position
        assert abs(x_px - body.x * hud.SCALE) < 1.0
        assert abs(y_px - (hud.VIEW_H - body.y * hud.SCALE)) < 1.0
    finally:
        env.close()


def test_compose_keeps_shape_and_marks_trail():
    """Overlay is cosmetic: same shape, trail pixels added only when requested."""
    frame = np.zeros((400, 600, 3), dtype=np.uint8)
    trail = [np.array([0.0, 1.4 - 0.01 * i, 0, 0, 0, 0, 0, 0], dtype=np.float32)
             for i in range(30)]
    plain = hud.compose(frame, trail, 2, show_trail=False, show_pad=False, style=False)
    with_trail = hud.compose(frame, trail, 2, show_trail=True, show_pad=False, style=False)
    assert plain.shape == with_trail.shape == frame.shape
    assert (with_trail != plain).any()
