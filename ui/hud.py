"""Cosmetic overlays drawn on top of real Gymnasium LunarLander frames.

Nothing here changes physics or actions. The trail and pad marker are derived from
the actual observations; the pixel mapping follows LunarLander-v3's own rendering
constants and is checked against the simulator in ``tests/test_hud.py``.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# LunarLander-v3 geometry: 600x400 viewport, SCALE 30, pad at H/4, legs 18 px below hull.
VIEW_W, VIEW_H, SCALE, LEG_DOWN = 600, 400, 30.0, 18.0
W_UNITS, H_UNITS = VIEW_W / SCALE, VIEW_H / SCALE
PAD_Y_UNITS = H_UNITS / 4
PAD_HALF_WIDTH = 0.2  # normalized x; matches ``metrics.classify``

NAVY_TOP, NAVY_BOTTOM = np.array([5, 12, 24]), np.array([16, 34, 56])
ACTION_SHORT = ("IDLE", "LEFT", "MAIN", "RIGHT")
CYAN, AMBER = (85, 214, 233), (244, 191, 105)


def to_pixels(obs) -> tuple[float, float]:
    """Map the normalized (x, y) observation to lander pixel coordinates."""
    x_units = float(obs[0]) * (W_UNITS / 2) + W_UNITS / 2
    y_units = float(obs[1]) * (H_UNITS / 2) + PAD_Y_UNITS + LEG_DOWN / SCALE
    return x_units * SCALE, VIEW_H - y_units * SCALE


def restyle(frame: np.ndarray) -> np.ndarray:
    """Cosmetic recolor: black sky -> navy gradient, white moon -> cool gray.

    Only exact background/terrain colors are replaced; lander, flags and engine
    particles keep Gymnasium's colors.
    """
    out = frame.copy()
    sky = (frame == 0).all(axis=2)
    ramp = np.linspace(0, 1, frame.shape[0])[:, None]
    gradient = (NAVY_TOP + (NAVY_BOTTOM - NAVY_TOP) * ramp).astype(np.uint8)
    out[sky] = np.broadcast_to(gradient[:, None, :], frame.shape)[sky]
    out[(frame == 255).all(axis=2)] = (196, 210, 224)
    return out


def compose(frame, trail, action=None, show_trail=True, show_pad=True, style=True):
    """Return an RGB array: real frame + optional trail, pad marker and command badge."""
    base = restyle(frame) if style else frame
    image = Image.fromarray(base).convert("RGBA")
    layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    height = image.size[1] / VIEW_H
    if show_pad:
        y = (VIEW_H - PAD_Y_UNITS * SCALE) * height
        x1 = (W_UNITS / 2 - PAD_HALF_WIDTH * W_UNITS / 2) * SCALE * height
        x2 = (W_UNITS / 2 + PAD_HALF_WIDTH * W_UNITS / 2) * SCALE * height
        draw.line([(x1, y), (x2, y)], fill=(*CYAN, 255), width=3)
        for x in (x1, x2):
            draw.line([(x, y - 7), (x, y + 7)], fill=(*CYAN, 255), width=2)
    points = [tuple(v * height for v in to_pixels(obs)) for obs in trail]
    if show_trail and len(points) > 1:
        total = len(points) - 1
        for index in range(total):
            alpha = int(40 + 215 * (index + 1) / total)
            draw.line([points[index], points[index + 1]], fill=(*AMBER, alpha), width=2)
    if action is not None:
        font = ImageFont.load_default(size=14)
        label = f"CMD {ACTION_SHORT[action]}"
        draw.rounded_rectangle([8, 8, 8 + 9 * len(label) + 10, 32], radius=5,
                               fill=(7, 17, 31, 200), outline=(*CYAN, 160))
        draw.text((14, 12), label, fill=(234, 243, 247, 255), font=font)
    return np.asarray(Image.alpha_composite(image, layer).convert("RGB"))


def attitude_svg(angle_rad: float, left_leg: bool, right_leg: bool) -> str:
    """Small attitude dial: horizon line rotated by the observed angle (derived, exact)."""
    deg = float(np.degrees(angle_rad))
    leg = lambda on: "#77D7A6" if on else "#284058"  # noqa: E731
    return (
        '<svg viewBox="0 0 120 120" width="100%" style="max-width:150px" role="img" '
        f'aria-label="Tilt {deg:+.1f} degrees"><circle cx="60" cy="60" r="52" fill="#0D1B2A" '
        'stroke="#284058" stroke-width="2"/><g class="dial" style="transform-origin:60px 60px;'
        f'transform:rotate({-deg:.1f}deg)"><rect x="12" y="60" width="96" height="1.6" '
        'fill="#55D6E9"/><path d="M60 38 L72 66 H48 Z" fill="#8F7BFF" opacity=".9"/></g>'
        '<path d="M60 4V14M4 60H14M106 60H116" stroke="#8FA6B8" stroke-width="2"/>'
        f'<circle cx="46" cy="96" r="5" fill="{leg(left_leg)}"/>'
        f'<circle cx="74" cy="96" r="5" fill="{leg(right_leg)}"/>'
        f'<text x="60" y="112" text-anchor="middle" font-size="9" fill="#8FA6B8" '
        f'font-family="Consolas,monospace">{deg:+.1f}°</text></svg>')


def q_bars(q_values, names) -> str:
    """Animated gap-to-best bars. Length is relative to the spread, never exaggerated."""
    best = max(q_values)
    span = max(best - min(q_values), 1.0)
    rows = []
    for name, value in zip(names, q_values, strict=True):
        chosen = value == best
        width = max(4.0, 100.0 * (1 - (best - value) / span))
        color = "#F4BF69" if chosen else "#55D6E9"
        rows.append(
            f'<div class="qrow"><span class="qn">{name}</span><span class="qbar">'
            f'<i style="width:{width:.0f}%;background:{color}"></i></span>'
            f'<span class="qv">{value - best:+.2f}</span></div>')
    return "".join(rows)
