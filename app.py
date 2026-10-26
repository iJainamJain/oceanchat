"""
OceanChat - Streamlit front end.

Layout follows a light two-panel chat design: a floating "Chats" panel on the left (the
Streamlit sidebar), a top navigation bar, and one white working panel. Only the active
page renders on each rerun, so the charts on other pages cost nothing.
"""

import json
import os
import pickle
import re
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import xarray as xr
from dotenv import load_dotenv

import store
import theme

load_dotenv()

st.set_page_config(
    page_title="OceanChat",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)
theme.inject_css()

DATA_CANDIDATES = [
    os.path.join("data", "argo_indian_ocean.nc"),
    os.path.join("data", "sample_argo.nc"),
]
MODEL_PATH = os.path.join("models", "ocean_model.keras")
SCALER_PATH = os.path.join("models", "scaler.pkl")
META_PATH = os.path.join("models", "model_meta.json")
DL_RESULTS_PATH = os.path.join("models", "dl_results.json")

LLM_MODEL = "qwen/qwen3.8-27b"

PAGES = {
    "chat": "Chat",
    "dashboard": "Dashboard",
    "map": "Map",
    "playground": "Playground",
    "bgc": "Biogeochem",
    "experiments": "Experiments",
    "memory": "Memory",
}

MODES = {"analyst": "Use data tools", "chat": "Just chat"}
MODE_NAME = {"analyst": "Analyst", "chat": "Chat"}
MODE_BLURB = {
    "analyst": "Writes and runs pandas code on the selected measurements, "
               "so answers come with real numbers.",
    "chat": "Talks through oceanography and the project without running code. "
            "Faster, and better for concepts.",
}
STARTERS = {
    "analyst": [
        ("Explore the data", "Average temperature below 1000 db"),
        ("Find extremes", "Which float recorded the highest salinity?"),
        ("Oxygen", "Where is dissolved oxygen lowest?"),
    ],
    "chat": [
        ("Learn a topic", "Explain the oxygen minimum zone"),
        ("How Argo works", "How does an Argo float measure a profile?"),
        ("Deep learning", "Why split training data by float, not by row?"),
    ],
}
PLACEHOLDER = {
    "analyst": "Ask about the data, e.g. mean salinity between 200 and 500 db",
    "chat": "Message...",
}
CHIPS = {
    "analyst": {"memory": ":material/memory: Memory",
                "filters": ":material/filter_alt: Filtered data"},
    "chat": {"memory": ":material/memory: Memory",
             "summary": ":material/dataset: Dataset summary"},
}

# Sparse biogeochemical parameters carried through by build_dataset.py.
BGC_META = {
    "DOX2": ("Dissolved oxygen", "umol/kg", "Tealgrn"),
    "CPHL": ("Chlorophyll-a", "mg/m3", "Emrld"),
    "BBP700": ("Backscattering (700 nm)", "m-1", "Purp"),
    "NTAW": ("Nitrate", "umol/kg", "Sunset"),
    "PHPH": ("pH", "total scale", "Peach"),
}


# ============================== LOADERS ==============================
def mtime(path):
    return os.path.getmtime(path) if os.path.exists(path) else 0


@st.cache_data(show_spinner="Loading ARGO dataset...")
def load_data(filepath, _mtime):
    """_mtime is a cache key: rebuilding the file invalidates the cache."""
    ds = xr.open_dataset(filepath)
    df = ds.to_dataframe().reset_index()
    df = df.drop(columns=["OBS"], errors="ignore")
    df = df.dropna(subset=["TEMP", "PSAL", "LATITUDE", "LONGITUDE"])
    if "TIME" in df.columns:
        df["TIME"] = pd.to_datetime(df["TIME"])
    return df


@st.cache_data
def load_json(path, _mtime):
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except json.JSONDecodeError:
        return None


@st.cache_resource(show_spinner="Loading neural network...")
def load_dl():
    # Imported lazily: TensorFlow adds seconds to every cold start otherwise.
    from tensorflow.keras.models import load_model

    model = load_model(MODEL_PATH)
    with open(SCALER_PATH, "rb") as fh:
        scaler = pickle.load(fh)
    return model, scaler


@st.cache_data
def dataset_summary(filepath, _mtime):
    """A few lines of context for 'Just chat' mode, which cannot run code."""
    d = load_data(filepath, _mtime)
    lines = [f"Dataset: {len(d):,} Argo measurements"]
    if "PLATFORM" in d.columns:
        lines[0] += f" from {d.PLATFORM.nunique()} floats"
    lines.append(f"Latitude {d.LATITUDE.min():.1f} to {d.LATITUDE.max():.1f}, "
                 f"longitude {d.LONGITUDE.min():.1f} to {d.LONGITUDE.max():.1f} (Indian Ocean)")
    if "TIME" in d.columns:
        lines.append(f"Time span {d.TIME.min():%Y-%m-%d} to {d.TIME.max():%Y-%m-%d}")
    lines.append(f"Temperature mean {d.TEMP.mean():.2f} C (range {d.TEMP.min():.1f} to "
                 f"{d.TEMP.max():.1f}); salinity mean {d.PSAL.mean():.2f} psu; "
                 f"maximum pressure {d.PRES.max():.0f} db")
    return "\n".join(lines)


def resolve_data_path():
    return next((p for p in DATA_CANDIDATES if os.path.exists(p)), None)


def sample_for_plot(frame, n=25000):
    return frame.sample(n, random_state=42) if len(frame) > n else frame


# ============================== LLM HELPERS ==============================
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def strip_reasoning(text):
    """Remove the <think> block that reasoning models such as Qwen emit."""
    text = THINK_RE.sub("", text)
    if "</think>" in text:
        text = text.split("</think>")[-1]
    elif "<think>" in text:
        text = text.split("<think>")[0]
    return text.strip()


def parse_json_array(raw):
    """Pull a JSON array out of an LLM reply, tolerating prose and code fences."""
    text = strip_reasoning(raw)
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()

    # Collect every balanced [...] span, then try the most promising ones.
    spans, depth, start = [], 0, None
    for i, ch in enumerate(text):
        if ch == "[":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "]" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                spans.append(text[start:i + 1])

    for span in sorted(spans, key=len, reverse=True):
        try:
            parsed = json.loads(span)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, list):
            return parsed
    return []


def llm():
    from langchain_groq import ChatGroq

    # The key's on-demand tier caps output at 1000 tokens/minute; Groq's default 2048
    # request is rejected outright with a 429. Qwen's hidden reasoning also counts toward
    # max_tokens and can exhaust it mid-thought, leaving an empty answer, so turn it off.
    return ChatGroq(model=LLM_MODEL, temperature=0, max_tokens=900, reasoning_effort="none")


def build_agent(frame, use_memory):
    from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

    floats = frame.PLATFORM.nunique() if "PLATFORM" in frame.columns else "many"
    prefix = (
        "You are an oceanographer analysing Argo float data held in the pandas dataframe "
        f"`df` ({len(frame):,} measurements from {floats} floats). Source: Euro-Argo / "
        "Copernicus Marine in-situ profiles from Argo profiling floats in the Indian Ocean, "
        "quality-controlled (QC flags 1-2 only). "
        "Columns: TEMP (deg C), PSAL (salinity, psu), PRES (pressure in decibar, "
        "approximately depth in metres), LATITUDE, LONGITUDE, TIME (datetime), "
        "PLATFORM (float WMO id). Sparse biogeochemical columns may also be present: "
        "DOX2 (dissolved oxygen, umol/kg), CPHL (chlorophyll-a, mg/m3), BBP700 "
        "(backscatter, m-1), NTAW (nitrate, umol/kg), PHPH (pH, total scale) - these are "
        "mostly NaN, so always dropna before using them. "
        "Each row is one depth level of one profile; a profile is one unique "
        "(PLATFORM, TIME) pair, so count profiles with df.groupby(['PLATFORM', 'TIME']).ngroups. "
        "Always compute answers by running code on the full `df`; never guess. "
        "Answer concisely and always state the units. "
        "Your only tool is named exactly `python_repl_ast`; never call any other tool name."
    )
    if use_memory:
        prefix += store.memories_block(st.session_state.memories)
    # No df.head() preview in the prompt: with reasoning off, Qwen otherwise answers
    # from the five preview rows ("data only goes to 12 db") instead of running code.
    # suffix="" with include_df_in_prompt=None is how the library allows that: passing
    # include_df_in_prompt=False on the tool-calling path crashes on prefix + None.
    return create_pandas_dataframe_agent(
        llm(), frame, verbose=True, allow_dangerous_code=True,
        agent_type="tool-calling", prefix=prefix, suffix="", include_df_in_prompt=None,
    )


def run_agent(frame, prompt, use_memory, attempts=3):
    """Qwen occasionally calls a tool name that does not exist (e.g. 'python_repl') and
    Groq rejects the whole request. The slip is random, so a fresh attempt usually works."""
    for attempt in range(attempts):
        try:
            return build_agent(frame, use_memory).invoke({"input": prompt})
        except Exception as exc:
            if "tool call validation failed" not in str(exc) or attempt == attempts - 1:
                raise


def plain_chat(prompt, history, use_memory, summary):
    system = (
        "You are OceanChat, an oceanography assistant for a student deep-learning "
        "project on Argo float data from the Indian Ocean. Explain clearly and "
        "concisely, use SI units, and say when you are unsure. You cannot run code in "
        "this mode: if a question needs exact numbers from the dataset, suggest "
        "switching to 'Use data tools' mode."
    )
    if summary:
        system += "\n\n" + summary
    if use_memory:
        system += store.memories_block(st.session_state.memories)

    messages = [("system", system)]
    messages += [(m["role"], m["content"]) for m in history[-10:]]
    messages.append(("user", prompt))
    return llm().invoke(messages).content


def answer_prompt(mode, prompt, history, frame, chips, summary):
    """Returns the reply text. Errors become a visible reply, not a vanishing toast."""
    use_memory = "memory" in chips
    try:
        if mode == "analyst":
            result = run_agent(frame, prompt, use_memory)
            text = result.get("output", result) if isinstance(result, dict) else result
        else:
            text = plain_chat(prompt, history, use_memory,
                              summary if "summary" in chips else None)
        return strip_reasoning(str(text)) or "_(The model returned an empty answer.)_"
    except Exception as exc:
        return f"⚠️ Something went wrong: `{exc}`"


def extract_memories(chat):
    """Ask the LLM for durable facts worth keeping. Returns a list of dicts."""
    transcript = "\n".join(
        f"{m['role'].upper()}: {m['content'][:600]}" for m in chat["messages"][-12:]
    )
    prompt = (
        "From the conversation below, extract any durable facts worth remembering "
        "for future sessions - the user's role or expertise, stated preferences, "
        "ongoing goals, or project constraints. Ignore one-off data questions and "
        "anything already obvious from the dataset itself.\n\n"
        "Reply with ONLY a JSON array. Each item: "
        '{"name": "short-kebab-slug", "description": "one line", '
        '"content": "the fact, one sentence", '
        '"type": "user" | "preference" | "project" | "reference"}. '
        "Return [] if there is nothing worth keeping.\n\n"
        f"CONVERSATION:\n{transcript}"
    )
    items = parse_json_array(llm().invoke(prompt).content)
    return [i for i in items if isinstance(i, dict) and i.get("content")]


# ============================== STATE ==============================
def filter_defaults(df_all):
    return {
        "f_depth": (float(df_all.PRES.min()), float(df_all.PRES.max())),
        "f_lat": (float(df_all.LATITUDE.min()), float(df_all.LATITUDE.max())),
        "f_lon": (float(df_all.LONGITUDE.min()), float(df_all.LONGITUDE.max())),
        "f_floats": [],
    }


def init_state(df_all):
    ss = st.session_state

    if "chats" not in ss:
        # Empty threads are not worth listing; drop any left from earlier sessions.
        ss.chats = [c for c in store.load_chats() if c["messages"]]
    if "memories" not in ss:
        ss.memories = store.load_memories()
    if "active_chat" not in ss or store.get_chat(ss.chats, ss.active_chat) is None:
        ss.active_chat = store.new_chat(ss.chats)["id"]
    if "pending" not in ss:
        ss.pending = None
    if "show_banner" not in ss:
        ss.show_banner = True
    if "page" not in ss:
        requested = st.query_params.get("page", "chat")
        ss.page = requested if requested in PAGES else "chat"
    if "last_mode" not in ss:
        ss.last_mode = "analyst"

    # Widgets that are not drawn on the current page are normally wiped from session
    # state. Re-assigning each key to itself keeps filters, mode and chips intact while
    # the user visits other pages. Those widgets are then rendered with a key only, no
    # default value, so Streamlit never sees two competing sources for one value.
    persistent = filter_defaults(df_all) | {
        "mode": "analyst",
        "chips_analyst": list(CHIPS["analyst"]),
        "chips_chat": list(CHIPS["chat"]),
    }
    for key, default in persistent.items():
        ss[key] = ss[key] if key in ss else default


def apply_filters(df_all):
    ss = st.session_state
    mask = (
        df_all.PRES.between(*ss.f_depth)
        & df_all.LATITUDE.between(*ss.f_lat)
        & df_all.LONGITUDE.between(*ss.f_lon)
    )
    if ss.f_floats and "PLATFORM" in df_all.columns:
        mask &= df_all.PLATFORM.isin(ss.f_floats)
    df = df_all[mask]
    return df, len(df) < len(df_all)


def active_filter_count(df_all):
    ss, d = st.session_state, filter_defaults(df_all)
    return sum(1 for k in ("f_depth", "f_lat", "f_lon") if tuple(ss[k]) != d[k]) + (
        1 if ss.f_floats else 0
    )


# ---------- callbacks: run before the rerun, so the UI reflects them immediately ----------
def set_page(page):
    st.session_state.page = page
    st.query_params["page"] = page


def select_chat(chat_id):
    st.session_state.active_chat = chat_id
    set_page("chat")


def new_chat():
    ss = st.session_state
    current = store.get_chat(ss.chats, ss.active_chat)
    if current is None or current["messages"]:
        ss.active_chat = store.new_chat(ss.chats)["id"]
    set_page("chat")


def delete_active_chat():
    ss = st.session_state
    ss.chats = store.delete_chat(ss.chats, ss.active_chat)
    ss.active_chat = store.new_chat(ss.chats)["id"]


def dismiss_banner():
    st.session_state.show_banner = False


def reset_filters(defaults):
    for key, value in defaults.items():
        st.session_state[key] = value


def keep_mode():
    """Clicking the active segment deselects it; treat that as 'stay put'."""
    ss = st.session_state
    if ss.mode is None:
        ss.mode = ss.last_mode
    ss.last_mode = ss.mode


# ============================== CHROME ==============================
def escape_md(text):
    return re.sub(r"([\\`*_\[\]#<>|~])", r"\\\1", text)


def recency_group(iso):
    try:
        when = datetime.fromisoformat(iso)
    except (TypeError, ValueError):
        return "Earlier"
    now = datetime.now(timezone.utc)
    if when.date() == now.date():
        return "Today"
    if now - when < timedelta(days=7):
        return "Previous 7 days"
    return "Earlier"


def render_sidebar(df_all, data_path):
    ss = st.session_state
    with st.sidebar:
        with st.container(key="sidehead", horizontal=True, vertical_alignment="center"):
            st.markdown('<div class="oc-side-title">Chats</div>',
                        unsafe_allow_html=True, width="stretch")
            st.button("", icon=":material/edit_square:", key="new_chat",
                      type="tertiary", help="New chat", on_click=new_chat)

        query = st.text_input("Search chats", key="chat_query",
                              placeholder="Search chats", label_visibility="collapsed")

        listed = [c for c in ss.chats if c["messages"]]
        if query:
            q = query.lower()
            listed = [c for c in listed
                      if q in c["title"].lower()
                      or any(q in m["content"].lower() for m in c["messages"])]

        with st.container(key="chatlist"):
            if not listed:
                st.markdown(
                    '<div class="oc-quiet">'
                    + ("No chats match that search." if query
                       else "It's quiet here… for now")
                    + "</div>",
                    unsafe_allow_html=True,
                )
            last_group = None
            for chat in listed:
                group = recency_group(chat.get("updated_at"))
                if group != last_group:
                    st.markdown(f'<div class="oc-group">{group}</div>', unsafe_allow_html=True)
                    last_group = group
                active = chat["id"] == ss.active_chat and ss.page == "chat"
                st.button(
                    escape_md(chat["title"]), key=f"chat_{chat['id']}", width="stretch",
                    type="secondary" if active else "tertiary",
                    on_click=select_chat, args=(chat["id"],),
                )

        floats = df_all.PLATFORM.nunique() if "PLATFORM" in df_all.columns else 0
        st.markdown(
            f'<div class="oc-side-foot">{os.path.basename(data_path)}<br>'
            f"{len(df_all):,} rows · {floats} floats<br>"
            f"{len(ss.memories)} memories · {LLM_MODEL}</div>",
            unsafe_allow_html=True,
        )


def render_top_bar():
    with st.container(key="topbar", horizontal=True, vertical_alignment="center"):
        st.markdown(theme.logo_html(), unsafe_allow_html=True, width="stretch")
        with st.container(key="nav", horizontal=True, horizontal_alignment="right",
                          gap=None, width="content"):
            for key, label in PAGES.items():
                st.button(
                    label, key=f"nav_{key}",
                    type="primary" if key == st.session_state.page else "tertiary",
                    on_click=set_page, args=(key,),
                )


def render_banner(df_all):
    if not st.session_state.show_banner:
        return
    floats = df_all.PLATFORM.nunique() if "PLATFORM" in df_all.columns else 0
    with st.container(key="banner", horizontal=True, horizontal_alignment="center",
                      vertical_alignment="center", gap="small"):
        st.markdown(
            f"Exploring **{len(df_all):,}** Argo measurements from **{floats}** floats "
            "across the Indian Ocean — see the"
        )
        st.button("deep-learning experiments", key="banner_link", type="tertiary",
                  on_click=set_page, args=("experiments",))
        st.button("", icon=":material/close:", key="banner_close", type="tertiary",
                  help="Dismiss", on_click=dismiss_banner)


def filters_popover(df_all):
    n = active_filter_count(df_all)
    defaults = filter_defaults(df_all)
    with st.popover(f"Filters · {n}" if n else "Filters", icon=":material/tune:",
                    type="tertiary", width=132,
                    help="Filter the measurements every page uses"):
        st.markdown('<div class="oc-label">Filter measurements</div>', unsafe_allow_html=True)
        d, la, lo = defaults["f_depth"], defaults["f_lat"], defaults["f_lon"]
        st.slider("Depth / pressure (db)", d[0], d[1], key="f_depth", step=10.0)
        st.slider("Latitude", la[0], la[1], key="f_lat")
        st.slider("Longitude", lo[0], lo[1], key="f_lon")
        if "PLATFORM" in df_all.columns:
            st.multiselect("Floats", sorted(df_all.PLATFORM.unique().tolist()),
                           key="f_floats", placeholder="All floats")
        st.button("Reset filters", key="f_reset", width="stretch",
                  on_click=reset_filters, args=(defaults,), disabled=n == 0)


def panel_header(title, subtitle, df_all, filters=True, extras=None):
    with st.container(key="panelhead", horizontal=True, vertical_alignment="center"):
        st.markdown(f'<div class="oc-title">{title}</div><div class="oc-sub">{subtitle}</div>',
                    unsafe_allow_html=True, width="stretch")
        if extras:
            extras()
        if filters:
            filters_popover(df_all)


def selection_text(df, filtered):
    return f"{len(df):,} measurements" + (" (filtered)" if filtered else "")


def no_data_state():
    theme.empty_state("No measurements match these filters",
                      "Open <b>Filters</b> above and widen the range, or reset them.")


# ============================== PAGES ==============================
def page_chat(df, filtered, df_all, ctx):
    ss = st.session_state
    chat = store.get_chat(ss.chats, ss.active_chat)
    mode = ss.mode or "analyst"
    has_messages = bool(chat["messages"])

    def mode_control():
        st.segmented_control("Mode", list(MODES), format_func=MODES.get, key="mode",
                             label_visibility="collapsed", on_change=keep_mode,
                             width="stretch")

    def header_extras():
        if has_messages:
            if st.button("Remember", icon=":material/bookmark_add:", key="remember",
                         type="tertiary", help="Save durable facts from this chat to memory",
                         disabled=not ctx["has_key"]):
                save_memories_from(chat)
            with st.popover("Options", icon=":material/more_horiz:", type="tertiary",
                            width=124):
                title = st.text_input("Title", value=chat["title"], key=f"title_{chat['id']}")
                if st.button("Rename", key="rename", width="stretch"):
                    store.rename_chat(ss.chats, chat["id"], title)
                    st.rerun()
                st.download_button("Export as Markdown", store.chat_to_markdown(chat),
                                   file_name=f"{chat['title'][:40]}.md",
                                   mime="text/markdown", width="stretch")
                st.button("Delete conversation", key="delete_chat", width="stretch",
                          on_click=delete_active_chat)

    title = chat["title"] if has_messages else "Chat with OceanChat"
    panel_header(title, f"{MODES[mode]} · {selection_text(df, filtered)}", df_all,
                 extras=header_extras)

    msgs = st.container(key="msgs", height=470, border=False, autoscroll=True)
    with msgs:
        if not has_messages:
            st.markdown(f'<div class="oc-intro">Try a prompt in <b>{MODE_NAME[mode]}</b> mode'
                        "</div>", unsafe_allow_html=True)
            with st.container(key="introbox"):
                mode_control()
                st.markdown(f'<div class="oc-intro-sub">{MODE_BLURB[mode]}</div>',
                            unsafe_allow_html=True)
                with st.container(key="cards", horizontal=True, gap="small", wrap=False):
                    for i, (label, prompt) in enumerate(STARTERS[mode]):
                        if st.button(f"{label}  \n{prompt}", key=f"card_{mode}_{i}",
                                     width="stretch"):
                            ss.pending = prompt
        else:
            for msg in chat["messages"]:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

    with st.container(key="composer"):
        if has_messages:
            mode_control()
        typed = st.chat_input(PLACEHOLDER[mode], key="chat_box",
                              disabled=not ctx["has_key"])
        chips = st.pills("Context", list(CHIPS[mode]), format_func=CHIPS[mode].get,
                         selection_mode="multi", key=f"chips_{mode}",
                         label_visibility="collapsed")
        if not ctx["has_key"]:
            st.caption("GROQ_API_KEY is missing from `.env`, so chat is disabled.")

    prompt = typed or ss.pending
    ss.pending = None
    if not prompt or not ctx["has_key"]:
        return

    history = list(chat["messages"])
    store.append_message(ss.chats, chat["id"], "user", prompt)
    with msgs:
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Running analysis on the data..." if mode == "analyst"
                            else "Thinking..."):
                frame = df if "filters" in chips else df_all
                answer = answer_prompt(mode, prompt, history, frame, chips, ctx["summary"])
    store.append_message(ss.chats, chat["id"], "assistant", answer)
    st.rerun()


def save_memories_from(chat):
    ss = st.session_state
    with st.spinner("Reviewing conversation..."):
        try:
            added = 0
            for item in extract_memories(chat):
                ss.memories, created = store.add_memory(
                    ss.memories, item.get("name", ""), item.get("description", ""),
                    item["content"], item.get("type", "project"), save=False,
                )
                added += int(created)
            store.save_memories(ss.memories)
            st.toast(f"Saved {added} new memories." if added
                     else "Nothing new worth remembering.")
        except Exception as exc:
            st.toast(f"Could not extract memories: {exc}")


def page_dashboard(df, filtered, df_all, ctx):
    panel_header("Dashboard", f"{selection_text(df, filtered)} · Argo profiles, Indian Ocean",
                 df_all)
    if df.empty:
        return no_data_state()

    k = st.columns(5)
    k[0].metric("Measurements", f"{len(df):,}")
    k[1].metric("Floats", f"{df.PLATFORM.nunique():,}" if "PLATFORM" in df.columns else "—")
    k[2].metric("Avg temperature", f"{df.TEMP.mean():.2f} °C")
    k[3].metric("Avg salinity", f"{df.PSAL.mean():.2f} psu")
    k[4].metric("Max depth", f"{df.PRES.max():,.0f} db")

    plot_df = sample_for_plot(df)
    if len(plot_df) < len(df):
        st.caption(f"Scatter plots drawn from a {len(plot_df):,}-point random sample.")

    c1, c2 = st.columns(2)
    with c1:
        fig = px.scatter(
            plot_df, x="TEMP", y="PRES", color="LATITUDE",
            title="Temperature profile by depth",
            labels={"TEMP": "Temperature (°C)", "PRES": "Pressure (db)", "LATITUDE": "Lat"},
            color_continuous_scale="RdYlBu_r", opacity=0.5,
        )
        fig.update_yaxes(autorange="reversed")
        fig.update_traces(marker=dict(size=3))
        st.plotly_chart(theme.style_fig(fig), width="stretch")
    with c2:
        # The T-S diagram is the standard way oceanographers identify water masses.
        fig = px.scatter(
            plot_df, x="PSAL", y="TEMP", color="PRES",
            title="Temperature–salinity diagram (water masses)",
            labels={"PSAL": "Salinity (psu)", "TEMP": "Temperature (°C)", "PRES": "Depth (db)"},
            color_continuous_scale="Viridis", opacity=0.5,
        )
        fig.update_traces(marker=dict(size=3))
        st.plotly_chart(theme.style_fig(fig), width="stretch")

    c3, c4 = st.columns(2)
    with c3:
        fig = px.histogram(df, x="PSAL", nbins=60, title="Salinity distribution",
                           labels={"PSAL": "Salinity (psu)"},
                           color_discrete_sequence=[theme.TEAL])
        st.plotly_chart(theme.style_fig(fig, 360), width="stretch")
    with c4:
        st.plotly_chart(binned_profile(df, "TEMP", "Temperature", "°C", 360), width="stretch")

    with st.expander("Raw data"):
        st.dataframe(df.head(200), width="stretch", hide_index=True)
        st.download_button("Download current selection (CSV)",
                           df.to_csv(index=False).encode("utf-8"),
                           file_name="argo_selection.csv", mime="text/csv")


def binned_profile(frame, column, label, units, height=420, min_count=1):
    """Mean profile in 50 db bins with a +/- 1 standard deviation envelope."""
    binned = (frame.assign(bin=(frame.PRES // 50) * 50)
              .groupby("bin")[column].agg(["mean", "std", "size"]))
    binned = binned[binned["size"] >= min_count].dropna(subset=["mean"])
    fig = go.Figure()
    if not binned.empty:
        std = binned["std"].fillna(0)
        fig.add_trace(go.Scatter(x=binned["mean"] + std, y=binned.index,
                                 line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=binned["mean"] - std, y=binned.index,
                                 line=dict(width=0), fill="tonextx",
                                 fillcolor="rgba(14,124,134,.14)", name="±1 std dev",
                                 hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=binned["mean"], y=binned.index,
                                 line=dict(color=theme.CORAL, width=2.5),
                                 name=f"Mean {label.lower()}"))
    fig.update_layout(title=f"Mean {label.lower()} profile (50 db bins)",
                      xaxis_title=f"{label} ({units})", yaxis_title="Pressure (db)")
    fig.update_yaxes(autorange="reversed")
    return theme.style_fig(fig, height)


def page_map(df, filtered, df_all, ctx):
    group_cols = ["LATITUDE", "LONGITUDE"] + (["PLATFORM"] if "PLATFORM" in df.columns else [])
    panel_header("Float map", f"{selection_text(df, filtered)} · colour shows near-surface "
                 "temperature (top 50 db)", df_all)
    if df.empty:
        return no_data_state()

    surface = df[df.PRES <= 50]
    prof = (surface if len(surface) else df).groupby(group_cols, as_index=False).agg(
        SST=("TEMP", "mean"), SSS=("PSAL", "mean"), n=("TEMP", "size"))
    st.caption(f"{len(prof):,} unique profile locations")

    fig = px.scatter_geo(
        prof, lat="LATITUDE", lon="LONGITUDE", color="SST",
        hover_name="PLATFORM" if "PLATFORM" in prof.columns else None,
        hover_data={"SST": ":.2f", "SSS": ":.2f", "n": True},
        color_continuous_scale="RdYlBu_r", projection="natural earth",
        labels={"SST": "Temp (°C)", "SSS": "Salinity (psu)", "n": "Samples"},
    )
    fig.update_traces(marker=dict(size=7, line=dict(width=0.5, color="#FFFFFF")))
    st.plotly_chart(theme.style_geo(fig, prof.LATITUDE, prof.LONGITUDE, 600), width="stretch")


def page_playground(df, filtered, df_all, ctx):
    meta = ctx["meta"]
    panel_header("Model playground",
                 "Predict temperature with the neural network and see the full profile it implies",
                 df_all, filters=False)

    if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH)):
        return theme.empty_state("No model yet", "Run <code>python train_model.py</code>.")

    try:
        model, scaler = load_dl()
    except Exception as exc:
        return st.error(f"Could not load the model: {exc}")

    features = (meta or {}).get("features", ["PRES", "PSAL"])
    left, right = st.columns([2, 3], gap="large")

    with left:
        theme.label("Inputs")
        inputs = {}
        for feat in features:
            lo, hi = float(df_all[feat].min()), float(df_all[feat].max())
            inputs[feat] = st.slider(feat, lo, hi, float(df_all[feat].median()),
                                     key=f"dl_{feat}")
        row = np.array([[inputs[f] for f in features]])
        pred = float(model.predict(scaler.transform(row), verbose=0)[0][0])
        st.metric("Predicted temperature", f"{pred:.2f} °C")

    with right:
        if "PRES" in features:
            depths = np.linspace(float(df_all.PRES.min()), float(df_all.PRES.max()), 200)
            grid = np.tile(row, (len(depths), 1))
            grid[:, features.index("PRES")] = depths
            curve = model.predict(scaler.transform(grid), verbose=0).ravel()
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=curve, y=depths, name="Predicted profile",
                                     line=dict(color=theme.TEAL, width=3)))
            fig.add_trace(go.Scatter(x=[pred], y=[inputs["PRES"]], mode="markers",
                                     name="Your selection",
                                     marker=dict(color=theme.CORAL, size=12, symbol="diamond")))
            fig.update_layout(title="Predicted temperature against depth (other inputs fixed)",
                              xaxis_title="Temperature (°C)", yaxis_title="Pressure (db)")
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(theme.style_fig(fig, 430), width="stretch")

    if not meta:
        return

    st.divider()
    theme.label("How well does it generalise?")
    m = st.columns(3)
    m[0].metric("Test MAE", f"{meta['mae']:.3f} °C")
    m[1].metric("Test RMSE", f"{meta['rmse']:.3f} °C")
    m[2].metric("R²", f"{meta['r2']:.4f}")

    floats = meta.get("n_floats", {})
    theme.note(
        f"<b>Evaluation protocol:</b> {meta.get('split', 'unspecified split')}. "
        f"Tested on {floats.get('test', '?')} floats the model never saw. Depth levels "
        "within one profile are near-identical, so a random split would leak "
        "near-duplicates into the test set and inflate the score."
    )
    if meta.get("baselines"):
        rows = [{"Model": k, "MAE (°C)": v["mae"], "RMSE (°C)": v["rmse"], "R²": v["r2"]}
                for k, v in meta["baselines"].items()]
        rows.append({"Model": "Neural network (this model)", "MAE (°C)": meta["mae"],
                     "RMSE (°C)": meta["rmse"], "R²": meta["r2"]})
        st.dataframe(pd.DataFrame(rows).round(4), width="stretch", hide_index=True)


def page_bgc(df, filtered, df_all, ctx):
    panel_header("Biogeochemistry",
                 "Oxygen, chlorophyll, backscatter, nitrate and pH from BGC-equipped floats",
                 df_all)
    available = [b for b in BGC_META if b in df.columns and bool(df[b].notna().any())]
    if not available:
        return theme.empty_state(
            "No biogeochemical data in this selection",
            "Only a minority of floats carry BGC sensors. Widen the filters.")

    cover = st.columns(len(BGC_META))
    for col, (name, (label, units, _)) in zip(cover, BGC_META.items()):
        n = int(df[name].notna().sum()) if name in df.columns else 0
        col.metric(label, f"{n:,}", help=f"Measurements in {units}")

    choice = st.segmented_control(
        "Parameter", available, default=available[0], key="bgc_choice",
        format_func=lambda b: BGC_META[b][0],
    ) or available[0]
    label, units, scale = BGC_META[choice]
    sub = df[df[choice].notna()]
    floats_with = sub.PLATFORM.nunique() if "PLATFORM" in sub.columns else 0
    st.caption(f"{len(sub):,} measurements from {floats_with} float(s) · range "
               f"{sub[choice].min():.3g} to {sub[choice].max():.3g} {units}")

    b1, b2 = st.columns(2)
    with b1:
        fig = px.scatter(
            sample_for_plot(sub, 20000), x=choice, y="PRES", color="LATITUDE",
            title=f"{label} against depth",
            labels={choice: f"{label} ({units})", "PRES": "Pressure (db)", "LATITUDE": "Lat"},
            color_continuous_scale="RdYlBu_r", opacity=0.55,
        )
        fig.update_traces(marker=dict(size=4))
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(theme.style_fig(fig), width="stretch")
    with b2:
        # For oxygen this is where the Arabian Sea oxygen minimum zone appears.
        st.plotly_chart(binned_profile(sub, choice, label, units, min_count=5), width="stretch")

    group_cols = ["LATITUDE", "LONGITUDE"] + (["PLATFORM"] if "PLATFORM" in sub.columns else [])
    loc = sub.groupby(group_cols, as_index=False).agg(value=(choice, "mean"), n=(choice, "size"))
    fig = px.scatter_geo(
        loc, lat="LATITUDE", lon="LONGITUDE", color="value", size="n",
        hover_name="PLATFORM" if "PLATFORM" in loc.columns else None,
        color_continuous_scale=scale, projection="natural earth",
        title=f"Where {label.lower()} was measured (profile means)",
        labels={"value": f"{label} ({units})", "n": "Samples"},
    )
    st.plotly_chart(theme.style_geo(fig, loc.LATITUDE, loc.LONGITUDE, 480), width="stretch")
    theme.note("Most BGC parameters here are distributed as <b>adjusted</b> values only and "
               "are far sparser than temperature and salinity, so treat regional averages "
               "with care.")


# ---------- experiments ----------
def compare(nn, base, tol=0.05):
    """Verdict for a lower-is-better metric: win / tie / neg, plus the relative change."""
    rel = (base - nn) / base if base else 0.0
    if rel > tol:
        return "win", rel
    if rel < -tol:
        return "neg", rel
    return "tie", rel


def learning_curves(history, title, height=300):
    fig = go.Figure()
    for name, series in history.items():
        if isinstance(series, dict):  # task 3 stores one history per architecture
            if "val_loss" in series:
                fig.add_trace(go.Scatter(y=series["val_loss"], name=f"{name} (val)",
                                         mode="lines"))
        elif name in ("loss", "val_loss"):
            fig.add_trace(go.Scatter(y=series, mode="lines",
                                     name="train" if name == "loss" else "validation"))
    fig.update_layout(title=title, xaxis_title="Epoch", yaxis_title="MSE loss (log)",
                      yaxis_type="log")
    return theme.style_fig(fig, height)


def metrics_frame(models, unit):
    rows = []
    for name, m in models.items():
        row = {"Model": name.strip(), f"MAE ({unit})": m["mae"], f"RMSE ({unit})": m["rmse"],
               "R²": m["r2"]}
        if "params" in m:
            row["Params"] = f"{m['params']:,}"
        rows.append(row)
    return pd.DataFrame(rows).round(4)


def experiment_block(number, title, desc, verdict_html, table, history, curve_title):
    st.markdown(
        f'<div class="oc-exp-head"><span class="n">{number:02d}</span>'
        f'<span class="t">{title}</span>{verdict_html}</div>'
        f'<div class="oc-exp-desc">{desc}</div>',
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns([3, 2], gap="medium")
    with c1:
        st.dataframe(table, width="stretch", hide_index=True)
    with c2:
        st.plotly_chart(learning_curves(history, curve_title), width="stretch")


def page_experiments(df, filtered, df_all, ctx):
    results = load_json(DL_RESULTS_PATH, mtime(DL_RESULTS_PATH))
    panel_header("Deep-learning experiments",
                 "Four models, each tested on floats it never saw and against a classical "
                 "baseline", df_all, filters=False)
    if not results:
        return theme.empty_state("No experiment results yet",
                                 "Run <code>python run_dl.py</code> to train all four.")

    theme.note(
        "<b>Protocol.</b> Every split is grouped by float, so no float appears in both "
        "training and test data. Neighbouring depth levels in one profile differ by a "
        "median of 0.008 °C, so a random split would leak near-duplicates and inflate "
        "every score below."
    )

    if "oxygen" in results:
        r = results["oxygen"]
        models = r["models"]
        nn = models["Neural network (MLP)"]
        best = min((v for k, v in models.items() if "Neural" not in k), key=lambda v: v["mae"])
        kind, rel = compare(nn["mae"], best["mae"])
        chip = theme.verdict_chip(kind, f"{rel * 100:+.0f}% MAE vs best baseline")
        st.divider()
        experiment_block(
            1, "Dissolved oxygen from physics",
            f"Predict oxygen from depth, temperature, salinity and position. Oxygen sensors "
            f"are rare ({r['n_floats']} floats), so inferring it from what every float "
            f"measures extends coverage. Tested on {r['n_test']:,} measurements from "
            "unseen floats.",
            chip, metrics_frame(models, "µmol/kg"), r["history"], "Training curve")

    if "autoencoder" in results:
        r = results["autoencoder"]
        rows = {}
        verdicts = []
        for var in ("TEMP", "PSAL"):
            rows[f"Autoencoder · {var}"] = r["autoencoder"][var]
            rows[f"PCA ({r['latent_dim']} comp) · {var}"] = r["pca_baseline"][var]
            verdicts.append(compare(r["autoencoder"][var]["mae"],
                                    r["pca_baseline"][var]["mae"])[0])
        kind = "win" if all(v == "win" for v in verdicts) else (
            "neg" if "neg" in verdicts else "tie")
        text = {"win": "beats PCA", "tie": "ties PCA", "neg": "no gain over PCA"}[kind]
        st.divider()
        experiment_block(
            2, "Profile autoencoder",
            f"Compress a 100-level temperature/salinity profile into {r['latent_dim']} numbers "
            f"({r['compression_ratio']}× smaller) and rebuild it. PCA at the same size is the "
            "linear yardstick. Ocean profiles sit close to a linear manifold, which is why a "
            "nonlinear encoder has little left to exploit.",
            theme.verdict_chip(kind, text), metrics_frame(rows, "native"),
            r["history"], "Reconstruction loss")

    if "sequence" in results:
        r = results["sequence"]
        models = r["models"]
        clim = models["Climatology (mean profile)"]
        nets = {k: v for k, v in models.items() if "Climatology" not in k}
        best_name, best = min(nets.items(), key=lambda kv: kv[1]["mae"])
        kind, rel = compare(best["mae"], clim["mae"])
        st.divider()
        experiment_block(
            3, "Full profile from surface data",
            "Given only surface temperature and salinity, position and season, predict the "
            "whole 0–2000 db temperature profile. Dense, 1D-CNN and LSTM decoders compete on "
            f"identical inputs. Best: {best_name.lower()}.",
            theme.verdict_chip(kind, f"{rel * 100:+.0f}% MAE vs climatology"),
            metrics_frame(models, "°C"), r["history"], "Validation loss by architecture")

    if "imputation" in results:
        r = results["imputation"]
        models = r["models"]
        cnn = models.get("Dilated CNN  TEMP")
        interp = models.get("Linear interp TEMP")
        kind, rel = compare(cnn["rmse"], interp["rmse"]) if cnn and interp else ("tie", 0)
        st.divider()
        experiment_block(
            4, "Masked-depth imputation",
            f"Self-supervised: hide {int(r['mask_fraction'] * 100)}% of depth levels in "
            f"blocks of {r['block_size']} and reconstruct them. Linear interpolation is what "
            "operational systems use. The gain shows in RMSE, where interpolation fails badly "
            "across the curved thermocline.",
            theme.verdict_chip(kind, f"{rel * 100:+.0f}% RMSE vs interpolation"),
            metrics_frame(models, "native"), r["history"], "Training curve")


def page_memory(df, filtered, df_all, ctx):
    ss = st.session_state
    panel_header("Memory", "Facts OceanChat keeps across conversations and restarts",
                 df_all, filters=False)

    with st.expander("Add a memory", icon=":material/add:"):
        content = st.text_area("Fact to remember",
                               placeholder="e.g. I'm presenting this at SIH in October.",
                               key="mem_content")
        c1, c2 = st.columns(2)
        mem_type = c1.selectbox("Type", store.MEMORY_TYPES, key="mem_type")
        name = c2.text_input("Short name", placeholder="sih-presentation", key="mem_name")
        if st.button("Save memory", key="mem_save", type="primary"):
            if content.strip():
                ss.memories, created = store.add_memory(
                    ss.memories, name or content[:40], content[:80], content.strip(), mem_type)
                st.toast("Memory saved." if created else "Memory updated.")
                st.rerun()
            else:
                st.warning("Write something to remember first.")

    if not ss.memories:
        return theme.empty_state(
            "No memories yet",
            "Add one above, or press <b>Remember</b> in a conversation to let the "
            "assistant pull out what matters.")

    for mem_type in store.MEMORY_TYPES:
        items = sorted((m for m in ss.memories if m["type"] == mem_type),
                       key=lambda m: m.get("updated_at", ""), reverse=True)
        if not items:
            continue
        theme.label(f"{mem_type} · {len(items)}")
        for mem in items:
            c_text, c_del = st.columns([12, 1], vertical_alignment="center")
            c_text.markdown(
                f'<div class="oc-mem"><div class="t">{mem["type"]}</div>'
                f'<div class="c">{mem["content"]}</div>'
                f'<div class="d">{mem["name"]} · updated {mem.get("updated_at", "")[:10]}</div>'
                "</div>",
                unsafe_allow_html=True,
            )
            if c_del.button("", icon=":material/delete:", key=f"del_{mem['id']}",
                            type="tertiary", help="Forget this"):
                ss.memories = store.delete_memory(ss.memories, mem["id"])
                st.rerun()


PAGE_FUNCS = {
    "chat": page_chat,
    "dashboard": page_dashboard,
    "map": page_map,
    "playground": page_playground,
    "bgc": page_bgc,
    "experiments": page_experiments,
    "memory": page_memory,
}


# ============================== MAIN ==============================
def main():
    data_path = resolve_data_path()
    if data_path is None:
        st.error("No dataset found in `data/`. Run `python build_dataset.py` first.")
        st.stop()

    data_mtime = mtime(data_path)
    df_all = load_data(data_path, data_mtime)
    init_state(df_all)
    df, filtered = apply_filters(df_all)

    ctx = {
        "meta": load_json(META_PATH, mtime(META_PATH)),
        "has_key": "GROQ_API_KEY" in os.environ,
        "summary": dataset_summary(data_path, data_mtime),
    }

    render_sidebar(df_all, data_path)
    render_top_bar()
    render_banner(df_all)
    with st.container(key="panel"):
        PAGE_FUNCS[st.session_state.page](df, filtered, df_all, ctx)


main()
