# ruff: noqa: E501
"""Mission-control design system: one CSS block and a few reusable HTML helpers."""

import streamlit as st

CSS = """
<style>
:root{--bg:#07111F;--panel:#0D1B2A;--panel2:#132538;--line:#284058;--cyan:#55D6E9;
--green:#77D7A6;--amber:#F4BF69;--red:#F08080;--text:#EAF3F7;--muted:#8FA6B8;
--mono:'JetBrains Mono','Cascadia Mono',Consolas,monospace;}
.stAppDeployButton,[data-testid="stToolbar"]{display:none!important}
.block-container{padding-top:1.4rem;max-width:1280px}
h1{font-size:1.55rem!important;letter-spacing:.02em;font-weight:650!important;
   padding:0 0 .1rem!important}
h2,h3{font-size:1.05rem!important;letter-spacing:.04em;text-transform:uppercase;
   color:var(--muted)!important;font-weight:600!important}
[data-testid="stMetric"]{padding:.15rem .1rem}
[data-testid="stMetricValue"]{font-family:var(--mono);font-size:1.45rem!important}
[data-testid="stMetricLabel"]{color:var(--muted);text-transform:uppercase;
   font-size:.68rem!important;letter-spacing:.08em}
[data-testid="stVerticalBlockBorderWrapper"]{background:var(--panel);border-color:var(--line)}
.pill{display:inline-block;padding:.12rem .65rem;border-radius:999px;font-size:.72rem;
   font-weight:700;letter-spacing:.08em;border:1px solid var(--line);margin:0 .3rem .3rem 0;
   color:var(--muted);font-family:var(--mono)}
.pill.on{color:#07111F;background:var(--amber);border-color:var(--amber)}
.pill.ok{color:#07111F;background:var(--green);border-color:var(--green)}
.pill.warn{color:#07111F;background:var(--amber);border-color:var(--amber)}
.pill.bad{color:#07111F;background:var(--red);border-color:var(--red)}
.pill.info{color:var(--cyan);border-color:var(--cyan)}
.tile{background:var(--panel2);border:1px solid var(--line);border-radius:10px;
   padding:.55rem .8rem;margin-bottom:.5rem}
.tile .k{color:var(--muted);font-size:.64rem;letter-spacing:.1em;text-transform:uppercase}
.tile .v{font-family:var(--mono);font-size:1.3rem;color:var(--text)}
.tile .u{color:var(--muted);font-size:.7rem}
.brand{font-weight:800;letter-spacing:.18em;font-size:1.05rem;color:var(--text)}
.brand b{color:var(--cyan)}
.sub{color:var(--muted);font-size:.78rem}
@keyframes rise{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:none}}
@keyframes pulse{0%,100%{box-shadow:0 0 0 0 rgba(119,215,166,.6)}70%{box-shadow:0 0 0 8px rgba(119,215,166,0)}}
@keyframes drift{from{background-position:0 0,0 0,0 0}to{background-position:260px 130px,180px 90px,0 0}}
@keyframes flow{to{stroke-dashoffset:-24}}
.hero{position:relative;overflow:hidden;border:1px solid var(--line);border-radius:14px;
 padding:1.1rem 1.4rem;margin-bottom:.9rem;animation:rise .6s ease both;
 background:radial-gradient(1.2px 1.2px at 20px 30px,#cfe9f2 60%,transparent),
 radial-gradient(1px 1px at 90px 80px,#8FA6B8 60%,transparent),
 linear-gradient(135deg,#0D1B2A 0%,#102a44 100%);
 background-size:260px 130px,180px 90px,100% 100%;animation:rise .6s ease both,drift 90s linear infinite}
.hero h2{margin:0!important;color:var(--text)!important;font-size:1.5rem!important;
 text-transform:none!important;letter-spacing:.01em}
.hero p{margin:.35rem 0 .6rem;color:var(--muted);max-width:60ch;font-size:.9rem}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;background:var(--green);
 margin-right:.45rem;animation:pulse 1.8s infinite}
.dot.idle{background:var(--muted);animation:none}
.flow{stroke-dasharray:6 6;animation:flow 1s linear infinite}
.dial{transition:transform .12s linear}
.qrow{display:flex;align-items:center;gap:.5rem;margin:.28rem 0;font-family:var(--mono);
 font-size:.74rem}
.qn{width:3.2rem;color:var(--muted)}.qv{width:3.4rem;text-align:right;color:var(--text)}
.qbar{flex:1;height:10px;background:var(--panel2);border-radius:6px;overflow:hidden}
.qbar i{display:block;height:100%;border-radius:6px;transition:width .15s ease}
section[data-testid="stSidebar"]{border-right:1px solid var(--line)}
.stButton>button{transition:transform .12s ease,border-color .12s ease}
.stButton>button:hover{transform:translateY(-1px);border-color:var(--cyan)}
</style>
"""


def apply() -> None:
    """Inject the shared design system once per script run."""
    st.markdown(CSS, unsafe_allow_html=True)


def pill(text: str, kind: str = "") -> str:
    """Status chip; kind in {'', 'on', 'ok', 'warn', 'bad', 'info'}."""
    return f'<span class="pill {kind}">{text}</span>'


def tile(label: str, value: str, unit: str = "") -> str:
    """Compact telemetry readout."""
    return (f'<div class="tile"><div class="k">{label}</div>'
            f'<div class="v">{value}</div><div class="u">{unit}&nbsp;</div></div>')


def brand() -> None:
    """Sidebar wordmark."""
    st.markdown('<div class="brand">LUNAR<b>/</b>AI</div>'
                '<div class="sub">AUTONOMOUS LANDING LAB · LOCAL</div>', unsafe_allow_html=True)


PIPELINE_SVG = """
<svg viewBox="0 0 900 190" width="100%" role="img"
 aria-label="DQN loop: observation, online Q-network, action, environment, reward, replay buffer, learning update; target network feeds the update">
<defs><marker id="a" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7"
 orient="auto"><path d="M0 0L8 4L0 8z" fill="#55D6E9"/></marker></defs>
<g font-family="Consolas,monospace" font-size="12" fill="#EAF3F7" text-anchor="middle">
<g stroke="#55D6E9" stroke-width="1.5" fill="#132538">
<rect x="10" y="20" width="120" height="50" rx="8"/><rect x="170" y="20" width="130" height="50" rx="8"/>
<rect x="340" y="20" width="110" height="50" rx="8"/><rect x="490" y="20" width="130" height="50" rx="8"/>
<rect x="660" y="20" width="110" height="50" rx="8"/>
<rect x="490" y="115" width="140" height="50" rx="8"/><rect x="250" y="115" width="150" height="50" rx="8"/></g>
<rect x="30" y="115" width="170" height="50" rx="8" fill="#0D1B2A" stroke="#F4BF69"
 stroke-width="1.5" stroke-dasharray="5 4"/>
<text x="70" y="42">Observation</text><text x="70" y="58" fill="#8FA6B8">8 numbers</text>
<text x="235" y="42">Online Q-network</text><text x="235" y="58" fill="#8FA6B8">Q(s,a) x4</text>
<text x="395" y="42">Action</text><text x="395" y="58" fill="#8FA6B8">ε-greedy</text>
<text x="555" y="42">LunarLander-v3</text><text x="555" y="58" fill="#8FA6B8">Box2D physics</text>
<text x="715" y="42">Reward</text><text x="715" y="58" fill="#8FA6B8">native</text>
<text x="560" y="137">Replay buffer</text><text x="560" y="153" fill="#8FA6B8">uniform sample</text>
<text x="325" y="137">Learning update</text><text x="325" y="153" fill="#8FA6B8">Huber loss, Adam</text>
<text x="115" y="137" fill="#F4BF69">Target network</text><text x="115" y="153" fill="#8FA6B8">frozen, periodic copy</text>
</g>
<g class="flow" stroke="#55D6E9" stroke-width="1.5" marker-end="url(#a)" fill="none">
<path d="M130 45H168"/><path d="M300 45H338"/><path d="M450 45H488"/><path d="M620 45H658"/>
<path d="M715 70V90H630V115" /><path d="M490 140H402"/></g>
<path d="M200 140H248" stroke="#F4BF69" stroke-width="1.5" stroke-dasharray="5 4"
 marker-end="url(#a)" fill="none"/>
<path d="M325 115V90H235V72" stroke="#55D6E9" stroke-width="1.5" fill="none"
 marker-end="url(#a)"/>
</svg>
"""


def hero(title: str, text: str, chips: str, live: bool = True) -> str:
    """Animated mission banner (static markup; CSS supplies the motion)."""
    return (f'<div class="hero"><span class="sub"><span class="dot{"" if live else " idle"}">'
            f'</span>REAL BOX2D SIMULATION · LOCAL</span><h2>{title}</h2><p>{text}</p>{chips}</div>')
