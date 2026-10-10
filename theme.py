"""theme.py - Colour palette, global CSS and small HTML snippets for the dashboard."""
from __future__ import annotations

import html

BG = "#0B1220"
CARD = "#111A2E"
CARD_HI = "#16223B"
BORDER = "#22304D"
TEXT = "#E6EDF7"
MUTED = "#8FA0BF"
CYAN = "#22D3EE"
BLUE = "#3B82F6"
VIOLET = "#8B5CF6"
GREEN = "#34D399"
AMBER = "#FBBF24"
RED = "#F87171"

ACCENTS = {"cyan": CYAN, "blue": BLUE, "violet": VIOLET, "green": GREEN, "amber": AMBER, "red": RED}
COLORWAY = [CYAN, BLUE, VIOLET, "#38BDF8", "#A78BFA", "#2DD4BF", "#818CF8", "#67E8F9"]

LOGO_SVG = f"""
<svg width="44" height="44" viewBox="0 0 44 44" xmlns="http://www.w3.org/2000/svg">
  <defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
    <stop offset="0" stop-color="{CYAN}"/><stop offset="1" stop-color="{VIOLET}"/></linearGradient></defs>
  <rect x="2" y="2" width="40" height="40" rx="11" fill="{CARD}" stroke="url(#g)" stroke-width="2"/>
  <path d="M10 16v-4h4M34 16v-4h-4M10 28v4h4M34 28v4h-4" stroke="url(#g)" stroke-width="2.4"
        stroke-linecap="round" fill="none"/>
  <circle cx="22" cy="22" r="5.5" fill="url(#g)"/>
</svg>"""

CSS = f"""
<style>
.stApp {{ background: radial-gradient(1200px 600px at 85% -10%, rgba(139,92,246,.14), transparent 60%),
                     radial-gradient(900px 500px at -10% 0%, rgba(34,211,238,.10), transparent 55%), {BG}; }}
section[data-testid="stSidebar"] {{ background: {CARD}; border-right: 1px solid {BORDER}; }}
.block-container {{ padding-top: 4.5rem; padding-bottom: 3rem; max-width: 1400px; }}
h1, h2, h3, h4 {{ letter-spacing: -0.01em; }}
.vt-hero {{ display:flex; align-items:center; gap:14px; margin-bottom: 6px; }}
.vt-title {{ font-size: 1.7rem; font-weight: 750; line-height: 1.1;
            background: linear-gradient(90deg, {CYAN}, {BLUE} 55%, {VIOLET});
            -webkit-background-clip: text; background-clip: text; color: transparent; }}
.vt-sub {{ color: {MUTED}; font-size: .9rem; }}
.vt-card {{ background: linear-gradient(180deg, {CARD_HI}, {CARD}); border: 1px solid {BORDER};
           border-radius: 14px; padding: 14px 16px; height: 100%; }}
.vt-card .lbl {{ color: {MUTED}; font-size: .74rem; text-transform: uppercase; letter-spacing: .08em; }}
.vt-card .val {{ font-size: 1.65rem; font-weight: 700; margin-top: 4px; line-height: 1.15; word-break: break-word; }}
.vt-card .sub {{ color: {MUTED}; font-size: .78rem; margin-top: 2px; }}
.vt-pill {{ display:inline-block; padding: 2px 10px; border-radius: 999px; font-size: .75rem;
           font-weight: 600; border: 1px solid; }}
.vt-empty {{ border: 1px dashed {BORDER}; border-radius: 14px; padding: 28px; text-align: center;
            color: {MUTED}; background: rgba(17,26,46,.5); }}
.vt-section {{ color: {MUTED}; font-size: .78rem; text-transform: uppercase; letter-spacing: .1em;
              margin: 18px 0 8px; }}
.vt-video {{ border: 1px solid {BORDER}; border-radius: 14px; padding: 8px; background: {CARD}; }}
div.stButton > button {{ border-radius: 10px; border: 1px solid {BORDER}; font-weight: 600; }}
div.stButton > button[kind="primary"] {{ background: linear-gradient(90deg, {CYAN}, {BLUE}); color: #04121f; border: 0; }}
[data-testid="stDataFrame"] {{ border: 1px solid {BORDER}; border-radius: 12px; }}
@keyframes vtpulse {{ 0% {{ opacity: 1; }} 50% {{ opacity: .45; }} 100% {{ opacity: 1; }} }}
.vt-live {{ animation: vtpulse 1.6s ease-in-out infinite; }}
</style>
"""


def esc(text) -> str:
    return html.escape(str(text))


def card(label: str, value, sub: str = "", accent: str = "cyan") -> str:
    color = ACCENTS.get(accent, CYAN)
    return (f'<div class="vt-card"><div class="lbl">{esc(label)}</div>'
            f'<div class="val" style="color:{color}">{esc(value)}</div>'
            f'<div class="sub">{esc(sub)}</div></div>')


def pill(text: str, kind: str = "cyan", live: bool = False) -> str:
    color = ACCENTS.get(kind, CYAN)
    cls = "vt-pill vt-live" if live else "vt-pill"
    return (f'<span class="{cls}" style="color:{color};border-color:{color};'
            f'background:{color}1A">{esc(text)}</span>')


def empty_state(title: str, hint: str = "") -> str:
    return f'<div class="vt-empty"><b>{esc(title)}</b><br>{esc(hint)}</div>'


def style_figure(fig, height: int = 330):
    """Apply the dark transparent look to a Plotly figure."""
    fig.update_layout(
        template="plotly_dark", height=height, paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", colorway=COLORWAY, font=dict(color=TEXT),
        margin=dict(l=10, r=10, t=30, b=10), legend=dict(bgcolor="rgba(0,0,0,0)"))
    fig.update_xaxes(gridcolor=BORDER, zerolinecolor=BORDER)
    fig.update_yaxes(gridcolor=BORDER, zerolinecolor=BORDER)
    return fig
