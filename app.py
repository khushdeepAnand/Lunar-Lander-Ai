"""Local mission-control dashboard for real LunarLander DQN artifacts."""

import json
import subprocess
import sys
from io import BytesIO
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from ui import hud, theme  # noqa: E402
from ui.data import (  # noqa: E402
    artifacts,
    checkpoint_in_record,
    checkpoints,
    comparable_reward_runs,
    configuration_differences,
    read_json,
    source_run,
    table,
    wilson,
)
from ui.jobs import TrainingJobs  # noqa: E402
from ui.playback import Playback  # noqa: E402

from lunar_lander_ai.checkpoint import load  # noqa: E402
from lunar_lander_ai.config import load_config  # noqa: E402

CYAN, GREEN, RED, AMBER = "#55D6E9", "#77D7A6", "#F08080", "#F4BF69"


@st.cache_resource
def jobs():
    """Share the one local training worker across dashboard sessions."""
    return TrainingJobs()


@st.cache_data(max_entries=40)
def _checkpoint_metadata(content: bytes) -> dict:
    """Cache by content, never path or timestamp; only caller-approved local bytes."""
    try:
        _, payload = load(BytesIO(content), torch.device("cpu"))
        return {"readable": True, "episode": payload["episode"],
                "created_utc": payload.get("created_utc"),
                "architecture": payload["config"]["agent"]["hidden_sizes"],
                "steps": payload["steps"], "config": payload["config"]}
    except Exception as exc:
        return {"readable": False, "error": str(exc)}


def checkpoint_info(path: Path) -> dict:
    """Only project-owned checkpoint directories cross the trusted-pickle boundary."""
    if path not in checkpoints():
        return {"readable": False, "error": "Not a project-local checkpoint"}
    return _checkpoint_metadata(path.read_bytes())


def heading(title, description):
    """Consistent mission page heading."""
    st.title(title)
    st.caption(description)


def selected_checkpoint():
    """Return the selected trusted local checkpoint if it still exists."""
    path = st.session_state.get("checkpoint")
    return Path(path) if path and Path(path) in checkpoints() else None


def evaluation_records(checkpoint=None):
    """Find only independent evaluations matching this checkpoint path."""
    for path in artifacts("results", "eval_*.json"):
        try:
            record = read_json(path)
            recorded = checkpoint_in_record(record)
            if checkpoint is None or (recorded and recorded.resolve() == checkpoint.resolve()):
                yield path, record
        except (OSError, ValueError, json.JSONDecodeError):
            continue


def chart(fig, title, x, y):
    """Apply a shared, accessible chart style."""
    fig.update_layout(title=title, xaxis_title=x, yaxis_title=y, height=330,
                      margin={"l": 24, "r": 20, "t": 48, "b": 35},
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font_color="#EAF3F7", hovermode="x unified")
    fig.update_xaxes(showgrid=True, gridcolor="#284058")
    fig.update_yaxes(showgrid=True, gridcolor="#284058")
    st.plotly_chart(fig, alt=f"{title}: {y} by {x}", width="stretch")


def outcome_bar(record):
    """Stacked bar of recorded outcomes; counts come from the artifact's own rates."""
    n = int(record["episodes"])
    parts = (("Safe landing", "landed_rate", GREEN),
             ("Off-pad / unsafe rest", "off_pad_or_unsafe_rate", AMBER),
             ("Timeout", "truncated_rate", "#8FA6B8"), ("Crash", "crashed_rate", RED))
    fig = go.Figure()
    for label, key, color in parts:
        count = round(n * record.get(key, 0))
        fig.add_bar(y=["Held-out"], x=[count], name=f"{label}: {count}", orientation="h",
                    marker_color=color, hovertemplate=f"{label}: {count}/{n}<extra></extra>")
    fig.update_layout(barmode="stack", height=120, margin={"l": 0, "r": 0, "t": 6, "b": 0},
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font_color="#EAF3F7", legend={"orientation": "h", "y": -0.35},
                      xaxis={"range": [0, n], "showgrid": False},
                      yaxis={"showticklabels": False})
    st.plotly_chart(fig, alt=f"Held-out outcomes over {n} episodes", width="stretch")


def overview():
    """Orient visitors using only saved observations and experiments."""
    cp = selected_checkpoint()
    run = source_run(cp) if cp else None
    log = ROOT / "logs" / f"train_{run}.csv" if run else None
    frame = table(log) if log and log.exists() else pd.DataFrame()
    evaluation = next(evaluation_records(cp), None) if cp else None
    info = checkpoint_info(cp) if cp else None
    if info and not info["readable"]:
        st.error(f"Selected checkpoint could not be validated: {info['error']}")
        info = None
    record = evaluation[1] if evaluation and info else None
    st.title("Mission overview")
    chips = (theme.pill(f"MODEL {cp.name}", "info") if info else theme.pill("NO MODEL", "warn"))
    if info:
        chips += theme.pill(f"CHECKPOINT EPISODE {info['episode']}")
    if record:
        chips += theme.pill(f"HELD-OUT n={int(record['episodes'])}", "ok")
    st.html(theme.hero(
        "An autonomous lander learns to land itself",
        "It observes eight numbers, chooses one of four engine commands, and improves "
        "through native environment rewards and experience replay. Everything below comes "
        "from recorded runs; nothing is scripted.", chips, live=info is not None))
    with st.container(border=True):
        model_card, episode_card, landing_card, return_card = st.columns(4)
        model_card.metric("Selected model", cp.name if info else "None")
        episode_card.metric("Checkpoint episode", str(info["episode"]) if info else "—")
        if record:
            n = int(record["episodes"])
            wins = round(n * record["landed_rate"])
            low, high = wilson(wins, n)
            landing_card.metric("Held-out safe landings", f"{wins}/{n}",
                                f"95% CI {low:.0%}–{high:.0%}", delta_color="off")
            return_card.metric("Held-out mean return", f"{record['mean_return']:.1f}",
                               "greedy policy", delta_color="off")
        else:
            landing_card.metric("Held-out safe landings", "—")
            return_card.metric("Held-out mean return", "—")
    if record:
        outcome_bar(record)
        timestamp = pd.Timestamp(evaluation[0].stat().st_mtime, unit="s", tz="UTC")
        st.caption(f"Independent evaluation {evaluation[0].stem} · "
                   f"{timestamp.strftime('%Y-%m-%d %H:%M UTC')} · Wilson 95% interval. "
                   "“Off-pad / unsafe rest” earned the rest bonus but failed the pad-and-both-legs "
                   "safe-landing test.")
    elif cp and info:
        st.info("No matching independent evaluation for this exact checkpoint path. "
                "Evaluate it in Model checkpoints.")
    else:
        st.info("Select a trusted local checkpoint in the sidebar to begin.")
    if not frame.empty:
        last = int(frame.episode.iloc[-1])
        st.caption(f"Source training run {run}: {last} episodes logged. Its last-100-episode "
                   f"mean native return ({frame.rolling_return.iloc[-1]:.1f}) includes ε-greedy "
                   "exploration and is not comparable with the greedy held-out return above."
                   + (" It also covers later episodes than this checkpoint."
                      if info and last != info["episode"] else ""))
    play, arena_link = st.columns(2)
    if play.button("Open live simulation", icon=":material/play_arrow:", disabled=info is None,
                   type="primary", width="stretch"):
        st.switch_page(PAGES["simulation"])
    if arena_link.button("Open policy arena", icon=":material/swords:", width="stretch"):
        st.switch_page(PAGES["arena"])
    with st.container(border=True):
        st.subheader("Learning loop")
        st.html(theme.PIPELINE_SVG)
        st.caption("The online network chooses actions. The target network is a frozen copy used "
                   "only for Bellman targets. Evaluation never updates weights.")
    if (ROOT / "docs/assets/landing_40400.gif").exists():
        st.image(str(ROOT / "docs/assets/landing_40400.gif"), width=520,
                 alt="Previously recorded real model episode, seed 40400")
        st.caption("Recorded example (seed 40400), not live telemetry. Play a fresh episode on "
                   "Live simulation.")


def command_chips(action):
    """Highlight the actual last applied command."""
    names = ("IDLE", "LEFT", "MAIN", "RIGHT")
    return "".join(theme.pill(name, "on" if action == i else "") for i, name in enumerate(names))


def simulation():
    """Advance the true environment one step per timed fragment rerun."""
    heading("Live simulation", "Real Gymnasium physics · greedy policy · normalized observations")
    cp = selected_checkpoint()
    if cp is None:
        st.warning("Select a checkpoint in the sidebar first.")
        return
    c1, c2, c3, c4, c5 = st.columns([1.2, 1, 1.4, 1, 1], vertical_alignment="bottom")
    seed = c1.number_input("Episode seed", min_value=0, value=40400, step=1)
    count = c2.number_input("Episodes", min_value=1, max_value=100, value=1)
    fps = c3.slider("Steps / second", 2, 30, 15)
    trail = c4.toggle("Trajectory trail", value=True)
    styled = c5.toggle("Mission styling", value=True, help="Cosmetic recolor of sky and "
                       "terrain only. Physics, lander and actions are untouched.")
    left, middle, right = st.columns(3)
    if left.button("Start / reset", icon=":material/restart_alt:", width="stretch"):
        old = st.session_state.pop("playback", None)
        if old:
            old.close()
        try:
            st.session_state.playback = Playback(cp, int(seed))
            st.session_state.sim_index = 1
            st.session_state.playing = True
        except Exception as exc:
            st.error(f"Could not start playback: {exc}")
    if middle.button("Pause / resume", disabled="playback" not in st.session_state,
                     width="stretch"):
        st.session_state.playing = not st.session_state.get("playing", False)
    if right.button("Stop", disabled="playback" not in st.session_state, width="stretch"):
        st.session_state.playback.close()
        del st.session_state["playback"]
        st.session_state.playing = False

    @st.fragment(run_every=1 / fps)
    def viewport():
        playback = st.session_state.get("playback")
        if playback is None:
            st.info("Press Start / reset to play the selected policy in the real environment.")
            return
        if st.session_state.get("playing") and playback.outcome is None:
            try:
                playback.step()
            except Exception as exc:
                playback.close()
                st.session_state.playing = False
                st.error(f"Simulation stopped: {exc}")
                return
        picture, telemetry = st.columns([5, 3])
        with picture:
            st.image(hud.compose(playback.frame, playback.trail, playback.action, trail, True,
                                 styled), alt="Current Gymnasium LunarLander frame with overlays",
                     width="stretch")
            st.caption("Cyan bar: landing-pad span. Amber trail: observed past positions of "
                       "this episode, not a prediction.")
            if playback.outcome:
                label = playback.outcome.replace("_", " ")
                (st.success if playback.outcome == "landed" else st.warning)(
                    f"Episode ended: {label} · {playback.steps} steps · "
                    f"native return {playback.total:+.1f}")
                if st.session_state.get("sim_index", 1) < count and st.button(
                    "Next episode", icon=":material/skip_next:"
                ):
                    try:
                        st.session_state.sim_index += 1
                        st.session_state.playback = Playback(
                            cp, int(seed) + st.session_state.sim_index - 1)
                        st.session_state.playing = True
                        st.rerun(scope="fragment")
                    except Exception as exc:
                        st.error(f"Could not start next episode: {exc}")
        with telemetry:
            obs = playback.obs
            if playback.outcome:
                state = theme.pill(playback.outcome.replace("_", " ").upper(),
                                   "ok" if playback.outcome == "landed" else "bad")
            else:
                state = (theme.pill("FLYING", "info") if st.session_state.get("playing")
                         else theme.pill("PAUSED", "warn"))
            st.markdown(state + theme.pill(f"EP {st.session_state.get('sim_index', 1)}/{count}")
                        + theme.pill(f"STEP {playback.steps}"), unsafe_allow_html=True)
            st.markdown("**Last applied command**")
            st.markdown(command_chips(playback.action), unsafe_allow_html=True)
            dial, readout = st.columns([1, 1])
            dial.html(hud.attitude_svg(obs[4], obs[6] > 0.5, obs[7] > 0.5))
            readout.html(theme.tile("Return", f"{playback.total:+.1f}",
                                    f"last reward {playback.reward:+.2f}"))
            grid = st.columns(2)
            grid[0].html(theme.tile("Height (norm.)", f"{obs[1]:+.3f}", "above pad reference")
                         + theme.tile("Horiz. offset (norm.)", f"{obs[0]:+.3f}",
                                      "pad is |x| ≤ 0.2"))
            grid[1].html(theme.tile("Vertical vel.", f"{obs[3]:+.3f}", "observation units")
                         + theme.tile("Horizontal vel.", f"{obs[2]:+.3f}", "observation units"))
            if playback.q_values is not None:
                st.caption("Network Q(s,a) gap to its best action, for the state before the last "
                           "step. 0 = chosen. Estimated return, not confidence.")
                st.html(hud.q_bars(playback.q_values, ("None", "Left", "Main", "Right")))
            with st.expander("Raw 8-value observation"):
                labels = ("x", "y", "vx", "vy", "angle (rad)", "angular vel.", "left leg",
                          "right leg")
                st.dataframe(pd.DataFrame({"Signal": labels, "Value": [f"{v:+.4f}" for v in obs]}),
                             hide_index=True, alt="Eight raw state observations")
            st.caption("Normalized Gymnasium coordinates, not meters. Tilt degrees are an exact "
                       "conversion of the observed angle.")

    viewport()


def arena_label(path):
    """Label from recorded checkpoint metadata, never from the filename alone."""
    info = checkpoint_info(path)
    if not info["readable"]:
        return f"{path.parent.name} (unreadable)"
    shaped = info["config"].get("reward_shaping", {}).get("enabled")
    return (f"episode {info['episode']} · {'shaped' if shaped else 'native'} reward · "
            f"{path.parent.name[-6:]}")


def arena():
    """Replay several real checkpoints on the identical seed, side by side."""
    heading("Policy arena", "Same seed · same physics · different learned policies")
    options = [p for p in checkpoints() if p.parent.name != "examples" and p.stem == "best"]
    if len(options) < 2:
        st.info("Need at least two trained run checkpoints to compare. Train another run first.")
        return
    chosen = st.multiselect("Checkpoints (max 3)", options, default=options[:3],
                            max_selections=3, format_func=arena_label)
    c1, c2 = st.columns([1, 2], vertical_alignment="bottom")
    seed = c1.number_input("Shared episode seed", min_value=0, value=40400, step=1)
    fps = c2.slider("Steps / second", 2, 30, 20)
    if st.button("Launch all", icon=":material/play_arrow:", type="primary",
                 disabled=not chosen):
        for _, old in st.session_state.pop("arena", []):
            old.close()
        try:
            st.session_state.arena = [(arena_label(p), Playback(p, int(seed))) for p in chosen]
        except Exception as exc:
            st.error(f"Could not start the arena: {exc}")

    @st.fragment(run_every=1 / fps)
    def stage():
        runs = st.session_state.get("arena")
        if not runs:
            st.info("Pick checkpoints and launch. Each panel runs the real simulator with that "
                    "checkpoint's own greedy actions.")
            return
        for column, (label, playback) in zip(st.columns(len(runs)), runs, strict=True):
            if playback.outcome is None:
                try:
                    playback.step()
                except Exception as exc:
                    playback.close()
                    column.error(f"Stopped: {exc}")
                    continue
            with column:
                st.markdown(f"**{label}**")
                st.image(hud.compose(playback.frame, playback.trail, playback.action),
                         alt=f"Live frame for {label}", width="stretch")
                done = playback.outcome
                st.markdown(
                    theme.pill(done.replace("_", " ").upper() if done else "FLYING",
                               "info" if not done else "ok" if done == "landed" else "bad")
                    + theme.pill(f"STEP {playback.steps}")
                    + theme.pill(f"RETURN {playback.total:+.0f}"), unsafe_allow_html=True)
        if all(p.outcome for _, p in runs):
            st.caption("One seed is one anecdote. Statistics over many seeds are on "
                       "Performance analytics and Experiment comparison.")

    stage()


def training():
    """Configure, launch and inspect a separate training process."""
    heading("Training lab", "Isolated worker · one local run · episode-boundary stop")
    defaults = load_config(ROOT / "configs/default.yaml").values
    a, t = defaults["agent"], defaults["training"]
    with st.form("train_config", border=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            episodes = st.number_input("Episodes", 1, 100000, t["episodes"])
            seed = st.number_input("Training seed", 0, 2_147_483_647, defaults["seed"])
            lr = st.number_input("Learning rate", 0.000001, 1.0, a["learning_rate"],
                                 format="%.6f")
            gamma = st.number_input("Discount factor", 0.001, 1.0, a["gamma"])
            device = st.selectbox("Device", ["auto", "cpu", "cuda"])
        with c2:
            batch = st.number_input("Batch size", 1, 100000, a["batch_size"])
            capacity = st.number_input("Replay capacity", 1, 10_000_000, a["buffer_capacity"])
            start = st.number_input("Epsilon start", 0.0, 1.0, a["epsilon"]["start"])
            end = st.number_input("Epsilon minimum", 0.0, 1.0, a["epsilon"]["end"])
            decay = st.number_input("Epsilon decay steps", 1, 100_000_000,
                                    a["epsilon"]["decay_steps"])
        with c3:
            target = st.number_input("Target update frequency", 1, 100000,
                                     a["target_update_freq"])
            hidden = st.text_input("Hidden dimensions (comma separated)",
                                   ",".join(map(str, a["hidden_sizes"])))
            evaluation = st.number_input("Evaluation interval", 1, 100000,
                                         t["eval_interval"])
            checkpoint = st.number_input("Checkpoint interval", 1, 100000,
                                         t["checkpoint_interval"])
            shaping = st.checkbox("Additional reward shaping", value=False)
            weight = st.number_input("Shaping weight", 0.0, 100.0, 2.0,
                                     disabled=not shaping)
        launched = st.form_submit_button("Start training", icon=":material/play_arrow:")
    if launched:
        try:
            sizes = [int(part.strip()) for part in hidden.split(",")]
            overrides = {"training.episodes": int(episodes), "seed": int(seed),
                         "device": device, "agent.learning_rate": float(lr),
                         "agent.gamma": float(gamma), "agent.batch_size": int(batch),
                         "agent.buffer_capacity": int(capacity),
                         "agent.epsilon.start": float(start), "agent.epsilon.end": float(end),
                         "agent.epsilon.decay_steps": int(decay),
                         "agent.target_update_freq": int(target), "agent.hidden_sizes": sizes,
                         "training.eval_interval": int(evaluation),
                         "training.checkpoint_interval": int(checkpoint),
                         "reward_shaping.enabled": bool(shaping),
                         "reward_shaping.angle_penalty_weight": float(weight) if shaping else 0.0}
            st.success(f"Started {jobs().start(overrides)}")
        except (ValueError, RuntimeError, OSError) as exc:
            st.error(f"Training not started: {exc}")

    @st.fragment(run_every=2)
    def monitor():
        state = jobs().status()
        st.subheader("Worker status")
        st.caption(f"{state['status'].upper()} · {state.get('run_id', 'No run')} · "
                   f"PID {state.get('pid', '—')}")
        st.button("Refresh status", icon=":material/refresh:")
        if state["status"] == "running" and st.button("Stop after current episode"):
            jobs().stop()
            st.warning("Stop requested; waiting for the current episode to finish.")
        run = state.get("run_id")
        if run:
            log = ROOT / "logs" / f"train_{run}.csv"
            if log.exists():
                frame = table(log)
                if not frame.empty:
                    row = frame.iloc[-1]
                    job_config = read_json(ROOT / "logs" / f"config_{run}.json")
                    budget = job_config["training"]["episodes"]
                    st.progress(min(row.episode / budget, 1.0),
                                text=f"Episode {int(row.episode)} / {budget}")
                    with st.container(horizontal=True):
                        for label, value in (("Native return", f"{row.native_return:.1f}"),
                                             ("Rolling return", f"{row.rolling_return:.1f}"),
                                             ("Epsilon", f"{row.epsilon:.3f}"),
                                             ("Mean loss", str(row.mean_loss)),
                                             ("Episode seconds", f"{row.duration_seconds:.1f}")):
                            st.metric(label, value, border=True)
                    st.download_button("Download training CSV", log.read_bytes(),
                                       file_name=log.name, mime="text/csv")
            latest = ROOT / "checkpoints" / run / "latest.pt"
            if latest.exists():
                st.caption(f"Checkpoint saved: {latest.relative_to(ROOT)} · "
                           f"{pd.Timestamp(latest.stat().st_mtime, unit='s', tz='UTC')}")
            periodic = ROOT / "logs" / f"eval_{run}.csv"
            if periodic.exists():
                evaluations = table(periodic)
                if not evaluations.empty:
                    result = evaluations.iloc[-1]
                    st.caption(f"Latest periodic evaluation: episode {int(result.episode)}, "
                               f"mean native return {result.mean_return:.1f}, "
                               f"safe landings {result.landed_rate:.0%} "
                               f"of {int(result.episodes)} episodes")
            output = ROOT / "logs" / f"output_{run}.txt"
            if output.exists():
                with st.expander("Worker stdout / errors"):
                    st.code(output.read_text(encoding="utf-8", errors="replace")[-12000:])
            if state.get("error"):
                st.error(state["error"])
            if state["status"] != "running" and log.exists() and st.button(
                "Open this run's analytics", icon=":material/query_stats:"
            ):
                st.session_state.analytics_run = run
                st.switch_page(PAGES["analytics"])
    monitor()
    st.caption("Stop is checked after each completed episode; final checkpoint is saved. "
               "Exact resumption is unsupported: the replay buffer and environment state "
               "are not persisted. The dashboard does not offer pause/resume.")


def analytics():
    """Chart recorded training and independent evaluation without inferred success."""
    heading("Performance analytics", "Training trajectories and independent evaluation")
    runs = artifacts("logs", "train_*.csv")
    if not runs:
        st.info("No training CSV yet. Start training to generate analytics.")
        return
    selected = st.session_state.get("analytics_run")
    default = next((i for i, p in enumerate(runs)
                    if p.stem.removeprefix("train_") == selected), 0)
    path = st.selectbox("Training run", runs, index=default,
                        format_func=lambda p: p.stem.removeprefix("train_"))
    st.session_state.analytics_run = path.stem.removeprefix("train_")
    window = st.slider("Rolling window (episodes)", 2, 200, 50)
    show_raw = st.checkbox("Show individual episode returns", value=True)
    frame = table(path)
    if frame.empty:
        st.info("This training log has no completed episodes yet.")
        return
    if "native_return" in frame:
        fig = go.Figure()
        if show_raw:
            fig.add_scatter(x=frame.episode, y=frame.native_return, name="Native return (raw)",
                            line={"color": "#547485", "width": 1})
        fig.add_scatter(x=frame.episode,
                        y=frame.native_return.rolling(window, min_periods=1).mean(),
                        name=f"Native return ({window}-episode mean)", line={"color": CYAN})
        chart(fig, "Training native return", "Training episode", "Native return")
    for column, title, color, ylabel in (("epsilon", "Exploration schedule", AMBER,
                                          "Epsilon"), ("mean_loss", "DQN training loss", CYAN,
                                                       "Mean Huber loss")):
        if column in frame and frame[column].notna().any():
            fig = go.Figure(go.Scatter(x=frame.episode, y=frame[column],
                                       name=title, line={"color": color}))
            chart(fig, title, "Training episode", ylabel)
    if "length" in frame:
        chart(go.Figure(go.Histogram(x=frame.length, marker_color=CYAN,
                                      name="Training episode lengths")),
              "Training episode lengths", "Environment steps", "Episode count")
    run = path.stem.removeprefix("train_")
    periodic = ROOT / "logs" / f"eval_{run}.csv"
    if periodic.exists():
        ev = table(periodic)
        if not ev.empty:
            for field, title, color in (("landed_rate", "Periodic safe landing rate", GREEN),
                                        ("crashed_rate", "Periodic crash rate", RED)):
                if field in ev:
                    fig = go.Figure(go.Scatter(x=ev.episode, y=ev[field] * 100,
                                               mode="lines+markers", name=title,
                                               line={"color": color},
                                               customdata=ev["episodes"],
                                               hovertemplate=("%{y:.1f}% · n=%{customdata}"
                                                              "<extra></extra>")))
                    chart(fig, title, "Training episode at evaluation", "Independent episodes (%)")
            st.caption("Periodic evaluation uses greedy action selection; sample size is "
                       "shown on hover. Training returns are not landing counts.")
    configs = []
    for training_log in runs:
        config_path = ROOT / "logs" / training_log.name.replace("train_", "config_")
        config_path = config_path.with_suffix(".json")
        try:
            if config_path.exists():
                configs.append((training_log, read_json(config_path)))
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    pairs = [(native, shaped) for native in configs for shaped in configs
             if comparable_reward_runs(native[1], shaped[1])]
    if pairs:
        native, shaped = st.selectbox(
            "Matched native vs shaped training", pairs,
            format_func=lambda pair: f"{pair[0][0].stem} vs {pair[1][0].stem}",
        )
        comparison = go.Figure()
        comparison_data = {}
        for label, (log_file, _) in (("Native reward", native), ("Additional shaping", shaped)):
            observations = table(log_file)
            if observations.empty or "native_return" not in observations:
                continue
            average = observations.native_return.rolling(window, min_periods=1).mean()
            comparison.add_scatter(x=observations.episode, y=average, name=label,
                                   line={"color": CYAN if label == "Native reward" else AMBER})
            comparison_data[label] = pd.Series(average.to_numpy(), index=observations.episode)
        if comparison.data:
            chart(comparison, "Matched reward-setting comparison", "Training episode",
                  f"{window}-episode mean native return")
            st.caption("Only the optional shaping configuration differs. These are "
                       "training native returns, not independent landing rates.")
            st.download_button("Download comparison CSV", pd.DataFrame(comparison_data)
                               .rename_axis("episode").to_csv().encode(),
                               file_name="reward_comparison.csv", mime="text/csv")
    evaluation = list(evaluation_records())
    if evaluation:
        choice = st.selectbox("Independent evaluation artifact", evaluation,
                              format_func=lambda item: item[0].stem)
        ep_csv = choice[0].with_suffix(".csv")
        summary = choice[1]
        n = int(summary.get("episodes", 0))
        if n:
            successes = round(n * summary.get("landed_rate", 0))
            lo, hi = wilson(successes, n)
            st.info(f"Safe landings: {successes}/{n} ({successes/n:.0%}). "
                    f"95% Wilson interval: {lo:.0%}–{hi:.0%}. "
                    "Small samples are only preliminary evidence.")
        if ep_csv.exists():
            samples = table(ep_csv)
            if "native_return" in samples:
                chart(go.Figure(go.Histogram(x=samples.native_return, marker_color=CYAN,
                                              name="Independent returns")),
                      "Independent evaluation returns", "Native episode return", "Episodes")
            st.download_button("Download evaluation CSV", ep_csv.read_bytes(),
                               file_name=ep_csv.name, mime="text/csv")
    st.download_button("Download training CSV", path.read_bytes(),
                       file_name=path.name, mime="text/csv")


def experiments():
    """Browse historic runs and flag confounding differences."""
    heading("Experiment comparison", "Compare native returns; inspect changes before interpreting")
    ids = sorted({p.stem.removeprefix("config_") for p in artifacts("logs", "config_*.json")},
                 reverse=True)
    if not ids:
        st.info("No experiments recorded yet.")
        return
    selected = st.multiselect("Runs to compare", ids, default=ids[:2])
    configs = {run: read_json(ROOT / "logs" / f"config_{run}.json") for run in selected}
    differences = configuration_differences(configs)
    st.subheader("Configuration differences")
    if differences.empty:
        st.caption("No recorded setting differences (or fewer than two runs selected).")
    else:
        st.dataframe(differences, hide_index=True, alt="Only settings that differ between runs")
    if len(selected) == 2:
        first, second = (configs[run] for run in selected)
        matched = comparable_reward_runs(first, second) or comparable_reward_runs(second, first)
        protocols = []
        for run in selected:
            records = [r for _, r in evaluation_records() if checkpoint_in_record(r)
                       and checkpoint_in_record(r).parent.name == run]
            protocols.append({(r.get("seed"), r.get("episodes")) for r in records})
        same_protocol = bool(protocols[0] & protocols[1])
        adequate = first["training"]["episodes"] >= 900 and all(
            any(n >= 100 and seed == 40400 for seed, n in protocol
                if isinstance(n, int)) for protocol in protocols
        )
        label = ("Controlled configuration, exploratory evidence"
                 if matched and same_protocol and not adequate else
                 "Controlled full-budget comparison" if matched and same_protocol else
                 "Exploratory / confounded")
        st.info(f"{label} · training budget: {first['training']['episodes']} / "
                f"{second['training']['episodes']} · independent eval protocol match: "
                f"{'yes' if same_protocol else 'no'}. "
                + ("Full-budget evidence available." if adequate and matched and same_protocol else
                   "Not a replicated full-budget shaping result."))
    fig = go.Figure()
    entries = []
    for run in selected:
        try:
            cfg = read_json(ROOT / "logs" / f"config_{run}.json")
            meta_path = ROOT / "logs" / f"run_{run}.json"
            meta = read_json(meta_path) if meta_path.exists() else {}
            row = {"Run": run, "Status": meta.get("status", "legacy / unknown"),
                   "Seed": cfg["seed"], "Episode budget": cfg["training"]["episodes"],
                   "Shaping": cfg["reward_shaping"]["enabled"],
                   "Epsilon decay": cfg["agent"]["epsilon"]["decay_steps"],
                   "Started": meta.get("started_utc", "—")}
            entry = ROOT / "logs" / f"train_{run}.csv"
            if entry.exists():
                frame = table(entry)
                if not frame.empty:
                    row["Completed"] = int(frame.episode.iloc[-1])
                    fig.add_scatter(x=frame.episode,
                                    y=frame.native_return.rolling(50, min_periods=1).mean(),
                                    name=run)
                    if "duration_seconds" in frame:
                        row["Training seconds"] = round(frame.duration_seconds.sum(), 1)
            matched = [record for _, record in evaluation_records()
                       if (checkpoint_in_record(record) is not None and
                           checkpoint_in_record(record).parent.name == run)]
            if matched:
                latest = matched[0]
                row["Evaluation n"] = latest.get("episodes")
                row["Mean eval return"] = latest.get("mean_return")
                for outcome in ("landed", "crashed", "truncated", "off_pad_or_unsafe"):
                    row[f"{outcome} %"] = round(latest.get(f"{outcome}_rate", 0) * 100, 1)
                row["Evaluation seed"] = latest.get("seed")
            entries.append(row)
        except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
            st.warning(f"Could not read {run}: {exc}")
    if entries:
        st.dataframe(pd.DataFrame(entries), hide_index=True, alt="Run metadata and differences")
        selected_run = st.selectbox("Inspect run artifacts", [item["Run"] for item in entries])
        st.json(read_json(ROOT / "logs" / f"config_{selected_run}.json"))
        st.caption(f"Training log: logs/train_{selected_run}.csv · "
                   f"Checkpoints: checkpoints/{selected_run}/ · "
                   f"Periodic evaluation: logs/eval_{selected_run}.csv")
    if fig.data:
        chart(fig, "Rolling native return by experiment", "Training episode",
              "50-episode mean native return")
    if len(entries) > 1 and any(len({row.get(k) for row in entries}) > 1 for k in
                                ("Seed", "Episode budget", "Shaping", "Epsilon decay",
                                 "Evaluation seed", "Evaluation n")):
        st.warning("These runs differ in seed, budget, reward setting or exploration schedule. "
                   "Their curves are descriptive, not a controlled A/B result.")
    st.caption("Independent evaluation artifacts and sample sizes are on Performance analytics. "
               "Missing/failed runs remain visible as evidence.")


def models():
    """Inspect local checkpoint metadata and run a genuine independent evaluation."""
    heading("Model checkpoints", "Best = highest periodic mean native return")
    cp = selected_checkpoint()
    st.warning("Checkpoints use PyTorch deserialization. Only repository-created, trusted "
               "local files are listed. Never place untrusted models in the model directories.")
    available = checkpoints()
    if not available:
        st.info("No local checkpoints found. Train an agent or add the trusted example model.")
        return
    validate_all = st.button("Validate all listed checkpoints",
                             help="Opt in: loads every trusted local model and may take a while")
    listing = []
    for file in available:
        info = checkpoint_info(file) if validate_all or file == cp else None
        kind = "Example" if file.parent.name == "examples" else file.stem.capitalize()
        listing.append({"Checkpoint": str(file.relative_to(ROOT)), "Type": kind,
                        "Run ID": file.parent.name if kind != "Example" else "Showcase copy",
                        "Training episode": info.get("episode") if info else None,
                        "Created": info.get("created_utc", "—") if info else "—",
                        "Architecture": str(info.get("architecture", "—")) if info else "—",
                        "Size (KiB)": round(file.stat().st_size / 1024, 1),
                        "Metadata valid": ("Yes" if info["readable"] else "No") if info
                        else "Not checked",
                        "Evaluations": len(list(evaluation_records(file)))})
    st.dataframe(pd.DataFrame(listing), hide_index=True,
                 alt="Trusted local models, type, training episode and validation status")
    if cp is None:
        st.info("Select a checkpoint in the sidebar to inspect or evaluate it.")
        return
    info = checkpoint_info(cp)
    if not info["readable"]:
        st.error(f"Checkpoint cannot be loaded: {info['error']}")
        return
    st.subheader(f"Selected: {cp.relative_to(ROOT)}")
    with st.expander("Checkpoint metadata and configuration"):
        st.json({key: value for key, value in info.items() if key != "readable"})
    matching = list(evaluation_records(cp))
    st.caption(f"{len(matching)} independent evaluation record(s) for this exact path.")
    for path, summary in matching[:3]:
        st.write(f"{path.name} · {summary.get('episodes')} episodes · "
                  f"safe landing {summary.get('landed_rate', 0):.0%}")
    run = source_run(cp)
    journey = sorted((ROOT / "checkpoints" / run).glob("episode_*.pt")) if run else []
    if journey:
        st.subheader("Learning journey · same training run")
        desired = {100, 300, 500, 750, 900}
        journey = [file for file in journey if int(file.stem.split("_")[1]) in desired]
        for col, file in zip(st.columns(max(1, len(journey))), journey, strict=False):
            with col:
                st.caption(f"Episode {int(file.stem.split('_')[1])}")
                if st.button("Replay", key=f"journey_{file}"):
                    st.session_state.checkpoint = str(file)
                    st.switch_page(PAGES["simulation"])
    else:
        st.caption("Learning journey requires periodic episode checkpoints from a new run.")
    galleries = [directory for directory in (ROOT / "results/gallery").glob("*/manifest.json")
                 if directory.is_file()]
    for manifest in galleries:
        try:
            data = read_json(manifest)
            if checkpoint_in_record(data) != cp:
                continue
            st.subheader("Recorded failure gallery")
            for item in data.get("episodes", []):
                image = manifest.parent / Path(item["image"]).name
                if image.exists():
                    st.image(str(image), alt=f"Actual frames: seed {item['seed']}, "
                             f"{item['outcome']}, {item['steps']} steps", width="stretch")
                    st.caption(f"Seed {item['seed']} · {item['outcome']} · {item['steps']} steps")
        except (OSError, KeyError, ValueError, json.JSONDecodeError):
            continue
    episodes = st.number_input("Independent evaluation episodes", 1, 1000, 30)
    seed = st.number_input("Evaluation seed", 0, 2_147_483_647, 40400)
    process = st.session_state.get("evaluation_process")
    running = process is not None and process.poll() is None
    if st.button("Evaluate selected checkpoint", disabled=running):
        output = ROOT / "logs" / "dashboard_evaluation.txt"
        with output.open("w", encoding="utf-8") as file:
            st.session_state.evaluation_process = subprocess.Popen(
                [sys.executable, str(ROOT / "scripts/evaluate.py"), "--checkpoint", str(cp),
                 "--episodes", str(int(episodes)), "--seed", str(int(seed)),
                 "--device", "cpu"], cwd=ROOT, stdout=file, stderr=subprocess.STDOUT)
        st.success("Evaluation started in a separate process. Refresh to view the result.")
    if process is not None:
        code = process.poll()
        st.caption("Evaluation running" if code is None else f"Evaluation exited with code {code}")
        output = ROOT / "logs" / "dashboard_evaluation.txt"
        if output.exists() and code is not None:
            st.code(output.read_text(encoding="utf-8", errors="replace")[-5000:])
    if st.button("Refresh evaluations"):
        st.rerun()


def about():
    """Document what this platform does and does not prove."""
    heading("About the mission", "Implementation and evidence")
    st.markdown("**Engine:** Gymnasium LunarLander-v3, discrete actions (0 none, 1 left, "
                "2 main, 3 right). **Policy:** PyTorch vanilla DQN, uniform replay, "
                "frozen target network, Huber loss. **Evidence:** native returns, explicit "
                "termination and landing-pad checks; no training reward is called a landing.")
    st.markdown("**Limits:** one local training worker; no exact checkpoint resume or "
                "multi-user isolation. Recorded runs may differ in methodology. "
                "95% landing intervals use Wilson's binomial score method (z=1.96).")
    st.caption("Open Performance analytics from the left navigation for measured results.")


st.set_page_config(page_title="Lunar AI | Mission control", page_icon=":material/rocket_launch:",
                   layout="wide")
st.session_state.setdefault("playing", False)
theme.apply()
with st.sidebar:
    theme.brand()
    files = checkpoints()
    choices = [None, *files]
    if "checkpoint" not in st.session_state:
        demo = ROOT / "examples" / "best_native.pt"
        st.session_state.checkpoint = str(demo if demo in files else files[0]) if files else None
    current = selected_checkpoint()
    picked = st.selectbox("Active checkpoint", choices,
                          index=choices.index(current) if current in choices else 0,
                          format_func=lambda p: str(p.relative_to(ROOT)) if p else "None")
    if str(picked) != st.session_state.checkpoint:
        old = st.session_state.pop("playback", None)
        if old:
            old.close()
        st.session_state.checkpoint = str(picked) if picked else None
        st.session_state.playing = False
    st.caption(f"Environment: LunarLander-v3 · model: {picked.name if picked else 'none'}")
PAGES = {
    "overview": st.Page(overview, title="Mission overview", icon=":material/dashboard:",
                        url_path="overview"),
    "simulation": st.Page(simulation, title="Live simulation", icon=":material/play_circle:",
                          url_path="simulation"),
    "arena": st.Page(arena, title="Policy arena", icon=":material/swords:", url_path="arena"),
    "training": st.Page(training, title="Training lab", icon=":material/model_training:",
                        url_path="training"),
    "analytics": st.Page(analytics, title="Performance analytics", icon=":material/query_stats:",
                         url_path="analytics"),
    "experiments": st.Page(experiments, title="Experiment comparison",
                           icon=":material/compare_arrows:", url_path="experiments"),
    "models": st.Page(models, title="Model checkpoints", icon=":material/folder_open:",
                      url_path="models"),
    "about": st.Page(about, title="About and technical details", icon=":material/info:",
                     url_path="about"),
}
page = st.navigation(list(PAGES.values()), position="sidebar")
if page.title != "Live simulation" and "playback" in st.session_state:
    st.session_state.pop("playback").close()
    st.session_state.playing = False
if page.title != "Policy arena":
    for _, running in st.session_state.pop("arena", []):
        running.close()
page.run()
