"""
Visual identity for OceanChat: a light, editorial layout in the style of Cohere's chat.

Native widget colours (sliders, inputs, tables, popovers) come from
.streamlit/config.toml. This module covers what the theme config cannot: the page
chrome (top nav, banner, floating panels), the custom components, and Plotly styling.

Every rule is scoped to a keyed container - st.container(key="x") renders with the CSS
class `st-key-x` - so styles for the nav, the chat list and the composer cannot leak
into one another the way global button selectors did in the previous design.

Streamlit caches imported modules: after editing only this file, restart the server,
or the old CSS keeps being served.
"""

from string import Template

import streamlit as st

CANVAS = "#F1EFEA"
PANEL = "#FFFFFF"
SUBTLE = "#F6F4F0"
INK = "#1C1B19"
MUTED = "#76726C"
FAINT = "#A29E97"
LINE = "#E5E2DC"
BANNER = "#E8DFD2"
CORAL = "#FF7759"
TEAL = "#0E7C86"
SAND = "#D9A441"

PLOT_SEQUENCE = [TEAL, CORAL, SAND, "#6C63B5", "#3FA37A", "#C2577E"]
MONO = "'IBM Plex Mono', ui-monospace, Consolas, monospace"

_TOKENS = dict(CANVAS=CANVAS, PANEL=PANEL, SUBTLE=SUBTLE, INK=INK, MUTED=MUTED,
               FAINT=FAINT, LINE=LINE, BANNER=BANNER, CORAL=CORAL, TEAL=TEAL, MONO=MONO)

# string.Template, not an f-string: CSS is full of braces and percentages.
_CSS = Template("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&display=swap');

/* ------------------------------------------------------------------ chrome */
header[data-testid="stHeader"] { background: transparent !important; pointer-events: none; }
header[data-testid="stHeader"] button { pointer-events: auto; }
[data-testid="stToolbarActions"], [data-testid="stStatusWidget"],
[data-testid="stDecoration"], #MainMenu, footer { display: none !important; }
.block-container { padding: .85rem 1.5rem 1rem !important; max-width: 100% !important; }

.oc-label {
  font-family: $MONO; font-size: .68rem; font-weight: 500; letter-spacing: .09em;
  text-transform: uppercase; color: $MUTED;
}

/* ----------------------------------------------------------------- top bar */
.st-key-topbar { border-bottom: 1px solid $LINE; padding-bottom: .5rem; margin-bottom: .15rem; }
.oc-logo { display: flex; align-items: center; gap: .5rem; font-weight: 600;
           font-size: 1.05rem; color: $INK; letter-spacing: -.3px; }
.oc-logo svg { flex: none; }

.st-key-nav button {
  background: transparent !important; border: none !important; box-shadow: none !important;
  border-radius: 0 !important; min-height: 0 !important; padding: .4rem .55rem !important;
  color: $MUTED !important;
}
.st-key-nav button p {
  font-family: $MONO; font-size: .72rem; font-weight: 500; letter-spacing: .09em;
  text-transform: uppercase; color: inherit !important;
}
.st-key-nav button:hover { color: $INK !important; }
.st-key-nav button[kind="primary"] { color: $INK !important; box-shadow: inset 0 -2px 0 $CORAL !important; }

/* ------------------------------------------------------------------ banner */
.st-key-banner {
  position: relative; background: $BANNER; border-radius: 8px;
  padding: .3rem 2.6rem; margin: .55rem 0 .8rem;
}
.st-key-banner p { margin: 0 !important; font-size: .82rem; color: #4C473F; }
.st-key-banner_link button {
  background: transparent !important; border: none !important; padding: 0 !important;
  min-height: 0 !important; color: $INK !important;
}
.st-key-banner_link button p { font-size: .82rem; text-decoration: underline;
                               text-underline-offset: 2px; }
.st-key-banner_close { position: absolute !important; right: .35rem; top: 50%;
                       transform: translateY(-50%); width: auto !important; }
.st-key-banner_close button { background: transparent !important; border: none !important;
                              min-height: 0 !important; padding: .15rem .4rem !important;
                              color: #6B655C !important; }

/* ------------------------------------------------------------------- panel */
.st-key-panel {
  background: $PANEL; border: 1px solid $LINE; border-radius: 12px;
  padding: 0 1.3rem 1.1rem;
}
.st-key-panelhead { border-bottom: 1px solid $LINE; padding: .7rem 0 .65rem; }
.oc-title { font-size: .98rem; font-weight: 600; color: $INK; line-height: 1.3; }
.oc-sub { font-size: .78rem; color: $MUTED; margin-top: .1rem; }
.st-key-panelhead button { min-height: 0 !important; padding: .32rem .65rem !important; }
.st-key-panelhead button p { font-size: .8rem; }

/* ------------------------------------------------------- chat: empty state */
.oc-intro { text-align: center; margin: 2rem 0 .4rem; font-size: 1.1rem;
            font-weight: 500; color: $INK; }
.st-key-introbox {
  max-width: 800px; margin: 0 auto !important; border: 1px solid $LINE;
  border-radius: 10px; padding: .35rem 1rem 1rem;
}
.oc-intro-sub { font-size: .82rem; color: #4C473F; margin: .2rem 0 .1rem; }

.st-key-mode [data-testid="stButtonGroup"] { width: 100%; }
.st-key-mode [data-testid="stButtonGroup"] > div { width: 100%; gap: 0 !important; }
.st-key-mode button {
  flex: 1 1 0; background: transparent !important; border: none !important;
  border-bottom: 2px solid $LINE !important; border-radius: 0 !important;
  padding: .55rem 1rem !important; color: $MUTED !important; box-shadow: none !important;
}
.st-key-mode button p { font-family: $MONO; font-size: .7rem; font-weight: 500;
                        letter-spacing: .09em; text-transform: uppercase; color: inherit !important; }
.st-key-mode button[aria-checked="true"], .st-key-mode button[data-selected="true"] {
  color: $INK !important; border-bottom-color: $CORAL !important;
}

.st-key-cards button {
  height: 100%; min-height: 5.8rem; background: $SUBTLE !important;
  border: 1px solid $LINE !important; border-radius: 8px !important;
  padding: .8rem .9rem !important; justify-content: flex-start !important;
  align-items: flex-start !important;
}
.st-key-cards button:hover { background: #EFECE6 !important; border-color: #D6D1C8 !important; }
.st-key-cards > div { flex: 1 1 0 !important; min-width: 0; }
.st-key-cards button > div,
.st-key-cards button [data-testid="stMarkdownContainer"] {
  width: 100%; justify-content: flex-start; text-align: left;
}
.st-key-cards button p { text-align: left; white-space: pre-line; font-size: .84rem;
                         line-height: 1.45; color: $INK; }
.st-key-cards button p::first-line {
  font-family: $MONO; font-size: .66rem; letter-spacing: .09em; text-transform: uppercase;
  color: $MUTED;
}

/* ---------------------------------------------------------- chat: messages */
/* Fill the viewport so the composer stays on screen, like a real chat panel.
   The fixed size comes from the wrapper around the pane, not the pane itself: the
   pane is a flex child and is shrunk to fit it, whatever its own height says. */
[data-testid="stLayoutWrapper"]:has(> .st-key-msgs) {
  height: calc(100vh - 385px) !important; min-height: 260px; flex: 0 0 auto !important;
}
.st-key-msgs { border: none !important; height: 100% !important; }
[data-testid="stChatMessage"] { background: transparent; padding: .55rem .25rem; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
  background: $SUBTLE; border-radius: 10px; padding: .6rem .85rem;
}
[data-testid="stChatMessageAvatarAssistant"] { background: $TEAL !important; }
[data-testid="stChatMessageAvatarAssistant"] svg { fill: #FFFFFF !important; }

/* ---------------------------------------------------------- chat: composer */
.st-key-composer { padding-top: .35rem; }
.st-key-composer [data-testid="stChatInput"] {
  border: 1px solid $LINE !important; border-radius: 10px !important; background: $PANEL !important;
}
.st-key-composer [data-testid="stButtonGroup"] button {
  min-height: 0 !important; padding: .18rem .6rem !important; border-radius: 6px !important;
}
.st-key-composer [data-testid="stButtonGroup"] button p { font-size: .76rem; }

/* ----------------------------------------------------------------- sidebar */
section[data-testid="stSidebar"] { background: $CANVAS !important; border-right: none !important;
                                   box-shadow: none !important; }
[data-testid="stSidebarContent"] {
  background: $PANEL; border: 1px solid $LINE; border-radius: 12px;
  margin: .85rem 0 .85rem .85rem; height: calc(100vh - 1.7rem);
}
[data-testid="stSidebarHeader"] { padding: .35rem .6rem 0 !important; height: 2.2rem !important; }
[data-testid="stSidebarUserContent"] { padding: 0 .95rem 1rem !important; }
section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: .35rem; }

.st-key-sidehead { border-bottom: 1px solid $LINE; padding-bottom: .5rem; }
.oc-side-title { font-weight: 600; font-size: .92rem; color: $INK; }
.st-key-sidehead button { background: transparent !important; border: none !important;
                          min-height: 0 !important; padding: .2rem .4rem !important;
                          color: $MUTED !important; }
.st-key-sidehead button:hover { color: $INK !important; }

.st-key-chatlist [data-testid="stVerticalBlock"] { gap: .1rem; }
.st-key-chatlist button {
  justify-content: flex-start !important; background: transparent !important;
  border: 1px solid transparent !important; border-radius: 7px !important;
  min-height: 0 !important; padding: .38rem .55rem !important; color: $INK !important;
}
.st-key-chatlist button > div,
.st-key-chatlist button [data-testid="stMarkdownContainer"] {
  width: 100%; min-width: 0; justify-content: flex-start; text-align: left;
}
.st-key-chatlist button p { text-align: left; font-size: .85rem; white-space: nowrap;
                            overflow: hidden; text-overflow: ellipsis; }
.st-key-chatlist button:hover { background: $SUBTLE !important; }
.st-key-chatlist button[kind="secondary"] { background: #EEEBE5 !important; }
.st-key-chatlist button[kind="secondary"] p { font-weight: 500; }
.oc-group { font-family: $MONO; font-size: .64rem; letter-spacing: .09em;
            text-transform: uppercase; color: $FAINT; margin: .55rem 0 .1rem .55rem; }
.oc-quiet { text-align: center; color: $FAINT; font-size: .8rem; padding: 4rem 0 3rem; }
.oc-side-foot { font-family: $MONO; font-size: .68rem; color: $MUTED; line-height: 1.7;
                border-top: 1px solid $LINE; padding-top: .6rem; margin-top: .4rem; }

/* -------------------------------------------------------------- components */
div[data-testid="stMetric"] {
  background: $SUBTLE; border: 1px solid $LINE; border-radius: 10px; padding: .7rem .95rem;
}
div[data-testid="stMetricLabel"] p {
  font-family: $MONO; font-size: .66rem !important; letter-spacing: .09em;
  text-transform: uppercase; color: $MUTED !important;
}
div[data-testid="stMetricValue"] { font-size: 1.35rem; font-weight: 600; color: $INK; }
div[data-testid="stMetricDelta"] svg { display: none !important; }
div[data-testid="stMetricDelta"] { color: $MUTED !important; font-size: .76rem; }

.oc-note {
  background: $SUBTLE; border: 1px solid $LINE; border-left: 3px solid $TEAL;
  border-radius: 0 8px 8px 0; padding: .6rem .85rem; font-size: .83rem; color: #3E3A34;
}
.oc-empty { text-align: center; padding: 2.4rem 1rem; color: $MUTED; font-size: .86rem; }
.oc-empty .h { color: $INK; font-weight: 600; font-size: .98rem; margin-bottom: .25rem; }

.oc-mem { border: 1px solid $LINE; border-radius: 8px; padding: .65rem .85rem; background: $PANEL; }
.oc-mem .t { font-family: $MONO; font-size: .64rem; letter-spacing: .09em;
             text-transform: uppercase; color: $CORAL; }
.oc-mem .c { color: $INK; font-size: .9rem; margin-top: .15rem; }
.oc-mem .d { color: $FAINT; font-size: .72rem; margin-top: .2rem; font-family: $MONO; }

.oc-exp-head { display: flex; align-items: center; gap: .6rem; flex-wrap: wrap; }
.oc-exp-head .n { font-family: $MONO; font-size: .7rem; color: $FAINT; }
.oc-exp-head .t { font-weight: 600; font-size: 1rem; color: $INK; }
.oc-exp-desc { color: $MUTED; font-size: .84rem; margin: .15rem 0 .3rem; }
.oc-verdict { font-family: $MONO; font-size: .64rem; font-weight: 600; letter-spacing: .08em;
              text-transform: uppercase; padding: .16rem .5rem; border-radius: 5px; }
.oc-verdict.win { background: #E2F1EA; color: #1E7650; }
.oc-verdict.tie { background: #F4EDDA; color: #85661C; }
.oc-verdict.neg { background: #FBE8E2; color: #AE472D; }

div[data-testid="stExpander"] details { border: 1px solid $LINE !important;
                                        border-radius: 10px !important; }
hr { border-color: $LINE !important; margin: .7rem 0 !important; }

/* Header: only the title flexes; popovers and buttons keep their natural width. */
.st-key-panelhead > div:not(:first-child) { flex: none !important; min-width: max-content; }
.st-key-panelhead button p { white-space: nowrap; }

/* Context chips: filled when on, dashed and faint when switched off. */
.st-key-composer [data-testid="stButtonGroup"] button[aria-checked="true"],
.st-key-composer [data-testid="stButtonGroup"] button[aria-pressed="true"] {
  background: $SUBTLE !important; border: 1px solid $LINE !important; color: $INK !important;
}
.st-key-composer [data-testid="stButtonGroup"] button[aria-checked="false"],
.st-key-composer [data-testid="stButtonGroup"] button[aria-pressed="false"] {
  background: transparent !important; border: 1px dashed $LINE !important; color: $FAINT !important;
}
</style>
""").substitute(_TOKENS)


def inject_css():
    # st.html rather than st.markdown: it is Streamlit's API for raw HTML, skips
    # markdown processing, and a block holding only styles takes up no layout space.
    st.html(_CSS)


def logo_html():
    return (
        '<div class="oc-logo">'
        '<svg width="22" height="22" viewBox="0 0 24 24" aria-hidden="true">'
        f'<rect width="24" height="24" rx="6" fill="{TEAL}"/>'
        '<path d="M3.5 10c1.7-1.8 3.4-1.8 5.1 0s3.4 1.8 5.1 0 3.4-1.8 5.1 0" '
        'stroke="#FFFFFF" stroke-width="1.9" fill="none" stroke-linecap="round"/>'
        '<path d="M3.5 15c1.7-1.8 3.4-1.8 5.1 0s3.4 1.8 5.1 0 3.4-1.8 5.1 0" '
        'stroke="#9FE3EA" stroke-width="1.9" fill="none" stroke-linecap="round"/>'
        "</svg><span>oceanchat</span></div>"
    )


def label(text):
    st.markdown(f'<div class="oc-label">{text}</div>', unsafe_allow_html=True)


def note(html):
    st.markdown(f'<div class="oc-note">{html}</div>', unsafe_allow_html=True)


def empty_state(title, body):
    st.markdown(
        f'<div class="oc-empty"><div class="h">{title}</div><div>{body}</div></div>',
        unsafe_allow_html=True,
    )


def verdict_chip(kind, text):
    return f'<span class="oc-verdict {kind}">{text}</span>'


# ------------------------------------------------------------------ Plotly
GEO = dict(
    showland=True, landcolor="#E7E2D8",
    showocean=True, oceancolor="#EDF4F5",
    showcoastlines=True, coastlinecolor="#B7B0A4",
    showframe=False, bgcolor="rgba(0,0,0,0)",
)


def style_fig(fig, height=420):
    """Apply the OceanChat look to any Plotly figure."""
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=PANEL,
        font=dict(color=INK, family="Inter, sans-serif", size=12),
        margin=dict(l=8, r=8, t=48, b=8),
        # text="" matters: styling a title that has no text makes Plotly draw the
        # string "undefined" above the figure (the float map has no title of its own).
        title=dict(text=fig.layout.title.text or "", x=0, xanchor="left",
                   font=dict(size=14, color=INK)),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=MUTED, size=11)),
        colorway=PLOT_SEQUENCE,
        hoverlabel=dict(bgcolor=PANEL, bordercolor=LINE,
                        font=dict(color=INK, family="Inter, sans-serif")),
    )
    axis = dict(gridcolor="#EEEBE5", zeroline=False, linecolor=LINE,
                tickfont=dict(color=MUTED), title_font=dict(color=MUTED))
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)
    return fig


def style_geo(fig, lat, lon, height=540, pad=5):
    fig.update_geos(
        **GEO,
        lataxis_range=[float(lat.min()) - pad, float(lat.max()) + pad],
        lonaxis_range=[float(lon.min()) - pad, float(lon.max()) + pad],
    )
    return style_fig(fig, height)
