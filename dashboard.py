"""
KKTC Enforcement Intelligence System — Streamlit Dashboard
Run: streamlit run dashboard.py
"""
import json
import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path

import folium
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from folium.plugins import HeatMap, MarkerCluster
from streamlit_folium import st_folium

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG  (must be first Streamlit call)
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="KKTC-INTEL",
    page_icon="⬛",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL CSS INJECTION
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&display=swap');

/* ── Variables ─────────────────────────────────────────────────────────── */
:root {
  --bg0:    #060a0d;
  --bg1:    #0b1118;
  --bg2:    #0e1820;
  --bg3:    #111c26;
  --cyan:   #00e5ff;
  --cyan2:  #00b8cc;
  --cyan3:  #005f6e;
  --cyan4:  #003040;
  --amber:  #ffaa00;
  --red:    #ff3333;
  --txt:    #E0E0E0;
  --txt2:   #F0F0F0;
  --mute:   #9CA3AF;
  --grid:   #0c1e28;
  --mono:   'Share Tech Mono','Courier New',Courier,monospace;
  --glow:   0 0 8px #00e5ff, 0 0 22px rgba(0,229,255,0.30);
  --glow-s: 0 0 5px #00e5ff, 0 0 12px rgba(0,229,255,0.20);
}

/* ── Base ───────────────────────────────────────────────────────────────── */
html, body, .stApp {
  background: var(--bg0) !important;
  font-family: var(--mono) !important;
  font-size: 14px !important;
}
p, li, td, th, label { font-size: 14px !important; }
.main .block-container {
  background: var(--bg0) !important;
  padding-top: 0.6rem !important;
  max-width: 1440px !important;
}
* { font-family: var(--mono) !important; }

/* ── Scanline overlay ──────────────────────────────────────────────────── */
.stApp::after {
  content: '';
  position: fixed;
  inset: 0;
  background: repeating-linear-gradient(
    0deg,
    transparent 0px,
    transparent 3px,
    rgba(0,229,255,0.012) 3px,
    rgba(0,229,255,0.012) 4px
  );
  pointer-events: none;
  z-index: 99999;
  mix-blend-mode: screen;
}

/* ── Sidebar ────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
  background: var(--bg1) !important;
  border-right: 1px solid var(--cyan3) !important;
}
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown span {
  color: var(--txt2) !important;
}
[data-testid="stSidebar"] hr {
  border-color: var(--cyan3) !important;
  opacity: 0.5 !important;
}
/* Nav radio items */
[data-testid="stSidebar"] [data-baseweb="radio"] label p {
  color: var(--mute) !important;
  letter-spacing: 2px !important;
  text-transform: uppercase !important;
  font-size: 11px !important;
  transition: color 0.15s, text-shadow 0.15s;
}
[data-testid="stSidebar"] [data-baseweb="radio"] label:hover p {
  color: var(--cyan) !important;
  text-shadow: var(--glow-s) !important;
}
/* Selected radio */
[data-testid="stSidebar"] [data-baseweb="radio"] [aria-checked="true"] ~ label p,
[data-testid="stSidebar"] [role="radio"][aria-checked="true"] + div p {
  color: var(--cyan) !important;
  text-shadow: var(--glow-s) !important;
}
/* Sidebar selectbox */
[data-testid="stSidebar"] [data-baseweb="select"] > div {
  background: var(--bg2) !important;
  border: 1px solid var(--cyan3) !important;
  border-radius: 0 !important;
  color: var(--cyan) !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] span { color: var(--cyan) !important; }
/* Selectbox dropdown panel */
[data-baseweb="popover"], [data-baseweb="menu"] {
  background: var(--bg2) !important;
  border: 1px solid var(--cyan3) !important;
  border-radius: 0 !important;
}
[data-baseweb="menu"] li { color: var(--txt) !important; }
[data-baseweb="menu"] li:hover { background: var(--cyan4) !important; color: var(--cyan) !important; }
/* Sidebar date input */
[data-testid="stSidebar"] input {
  background: var(--bg2) !important;
  border: 1px solid var(--cyan3) !important;
  border-radius: 0 !important;
  color: var(--cyan) !important;
}
/* Sidebar checkboxes */
[data-testid="stSidebar"] [data-baseweb="checkbox"] span { color: var(--txt) !important; }
[data-testid="stSidebar"] [data-baseweb="checkbox"] [data-checked] { background: var(--cyan) !important; }
/* Sidebar caption */
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
  color: var(--mute) !important;
  font-size: 10px !important;
  letter-spacing: 1px !important;
}

/* ── Headings ───────────────────────────────────────────────────────────── */
h1 {
  font-family: var(--mono) !important;
  color: var(--cyan) !important;
  text-shadow: var(--glow) !important;
  letter-spacing: 5px !important;
  text-transform: uppercase !important;
  font-size: 1.45rem !important;
  font-weight: normal !important;
  border-bottom: 1px solid var(--cyan3);
  padding-bottom: 8px;
  margin-bottom: 0 !important;
}
h2 {
  font-family: var(--mono) !important;
  color: #9CA3AF !important;
  letter-spacing: 3px !important;
  text-transform: uppercase !important;
  font-size: 0.85rem !important;
  font-weight: normal !important;
  opacity: 0.9;
}
h3 {
  font-family: var(--mono) !important;
  color: #9CA3AF !important;
  letter-spacing: 2px !important;
  text-transform: uppercase !important;
  font-size: 0.75rem !important;
  font-weight: normal !important;
}
/* Caption */
[data-testid="stCaptionContainer"] p, small {
  color: var(--mute) !important;
  font-size: 10px !important;
  letter-spacing: 1px !important;
}

/* ── KPI Metrics ────────────────────────────────────────────────────────── */
[data-testid="metric-container"] {
  background: var(--bg1) !important;
  border: 1px solid var(--cyan3) !important;
  border-top: 2px solid var(--cyan) !important;
  border-radius: 0 !important;
  padding: 14px 16px 10px !important;
}
[data-testid="stMetricValue"],
[data-testid="stMetricValue"] > div,
[data-testid="stMetricValue"] p {
  font-family: var(--mono) !important;
  color: var(--cyan) !important;
  text-shadow: var(--glow) !important;
  font-size: 1.85rem !important;
  font-weight: normal !important;
  letter-spacing: 2px !important;
}
[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] > div,
[data-testid="stMetricLabel"] p {
  font-family: var(--mono) !important;
  color: var(--mute) !important;
  font-size: 9px !important;
  letter-spacing: 2.5px !important;
  text-transform: uppercase !important;
}

/* ── Divider ────────────────────────────────────────────────────────────── */
hr {
  border: none !important;
  border-top: 1px solid var(--cyan3) !important;
  opacity: 0.4 !important;
}

/* ── Info / Warning ─────────────────────────────────────────────────────── */
[data-testid="stAlert"] {
  background: var(--bg1) !important;
  border: 1px solid var(--cyan3) !important;
  border-left: 3px solid var(--cyan) !important;
  border-radius: 0 !important;
  color: var(--txt) !important;
}
[data-testid="stAlert"] p { color: var(--txt) !important; }

/* ── DataFrames ─────────────────────────────────────────────────────────── */
[data-testid="stDataFrame"] {
  border: 1px solid var(--cyan3) !important;
  border-radius: 0 !important;
}
[data-testid="stDataFrame"] iframe {
  border-radius: 0 !important;
  background: var(--bg1) !important;
}

/* ── Expander ───────────────────────────────────────────────────────────── */
[data-testid="stExpander"] {
  border: 1px solid var(--cyan3) !important;
  border-radius: 0 !important;
  background: var(--bg1) !important;
}
[data-testid="stExpander"] summary {
  color: var(--cyan) !important;
  font-size: 11px !important;
  letter-spacing: 2px !important;
  text-transform: uppercase !important;
}

/* ── Columns / block structure ──────────────────────────────────────────── */
[data-testid="column"] { background: transparent !important; }

/* ── Blink animation ────────────────────────────────────────────────────── */
@keyframes blink {
  0%, 100% { opacity: 1; }
  50%       { opacity: 0.15; }
}
@keyframes sweep {
  0%   { opacity: 0.6; }
  50%  { opacity: 1; }
  100% { opacity: 0.6; }
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
BASE_DIR  = Path(__file__).parent
DB_PATH   = BASE_DIR / "data" / "arrests.db"
PRED_JSON = BASE_DIR / "data" / "predictions.json"

NC_CENTER = (35.17, 33.36)
NC_ZOOM   = 10

DAY_ORDER = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

TIME_WINDOWS = {
    "ALL DAY":                (0,  24),
    "EARLY MORNING  05–08":   (5,   8),
    "MORNING        08–12":   (8,  12),
    "AFTERNOON      12–17":   (12, 17),
    "EVENING        17–21":   (17, 21),
    "NIGHT          21–05":   (21, 29),
}

TW_LABELS = {
    "early_morning": "EARLY MORNING 05–08",
    "morning":       "MORNING 08–12",
    "afternoon":     "AFTERNOON 12–17",
    "evening":       "EVENING 17–21",
    "night":         "NIGHT 21–05",
}

# Dark-theme chart colours — cyan spectrum + amber/red alerts
TW_COLORS = {
    "early_morning": "#00607a",
    "morning":       "#008080",
    "afternoon":     "#00b8cc",
    "evening":       "#cc8800",
    "night":         "#6040a0",
}
SEASON_COLORS = {
    "winter": "#005a80",
    "spring": "#006840",
    "summer": "#a06000",
    "autumn": "#8a3000",
}

# Plotly base layout applied to every chart
_FONT = dict(family="'Courier New', Courier, monospace")
CHART_BASE = dict(
    paper_bgcolor="#060a0d",
    plot_bgcolor="#060a0d",
    font=dict(**_FONT, color="#9CA3AF", size=11),
    xaxis=dict(
        gridcolor="#0c1e28", linecolor="#0c1e28", zerolinecolor="#0c1e28",
        tickfont=dict(**_FONT, color="#9CA3AF", size=10),
        title_font=dict(**_FONT, color="#9CA3AF"),
        showgrid=True, zeroline=False,
    ),
    yaxis=dict(
        gridcolor="#0c1e28", linecolor="#0c1e28", zerolinecolor="#0c1e28",
        tickfont=dict(**_FONT, color="#9CA3AF", size=10),
        title_font=dict(**_FONT, color="#9CA3AF"),
        showgrid=True, zeroline=False,
    ),
    hoverlabel=dict(
        bgcolor="#0b1118", bordercolor="#005f6e",
        font=dict(**_FONT, color="#00e5ff", size=12),
    ),
    legend=dict(
        bgcolor="rgba(0,0,0,0)", bordercolor="#0c1e28",
        font=dict(**_FONT, color="#9CA3AF", size=10),
        orientation="h", yanchor="bottom", y=1.01, x=0,
    ),
    margin=dict(t=30, b=30, l=10, r=10),
)


def dark_fig(height: int = 380, **overrides) -> go.Figure:
    """Return an empty Figure pre-loaded with the dark military layout."""
    layout = {**CHART_BASE, "height": height, **overrides}
    return go.Figure(layout=go.Layout(**layout))


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADERS
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=300)
def load_arrests() -> pd.DataFrame:
    conn = sqlite3.connect(str(DB_PATH))
    df = pd.read_sql_query("SELECT * FROM arrests ORDER BY date DESC, time DESC", conn)
    conn.close()
    df["date"]         = pd.to_datetime(df["date"], errors="coerce")
    df["count"]        = pd.to_numeric(df["count"],        errors="coerce").fillna(1).astype(int)
    df["hour_decimal"] = pd.to_numeric(df["hour_decimal"], errors="coerce")
    df["latitude"]     = pd.to_numeric(df["latitude"],     errors="coerce")
    df["longitude"]    = pd.to_numeric(df["longitude"],    errors="coerce")
    df["is_weekend"]   = pd.to_numeric(df["is_weekend"],   errors="coerce").fillna(0).astype(int)
    return df


@st.cache_data(ttl=300)
def load_predictions() -> dict:
    if PRED_JSON.exists():
        with open(PRED_JSON, encoding="utf-8") as f:
            return json.load(f)
    return {}


@st.cache_data(ttl=300)
def load_accuracy_data() -> pd.DataFrame:
    """Load all evaluated predictions (was_correct IS NOT NULL)."""
    conn = sqlite3.connect(str(DB_PATH))
    df = pd.read_sql_query(
        """
        SELECT id, target_date, district, specific_area,
               predicted_time_window, predicted_hour, predicted_count,
               confidence, was_correct, predicted_at
        FROM predictions
        WHERE was_correct IS NOT NULL
        ORDER BY target_date DESC
        """,
        conn,
    )
    conn.close()
    if not df.empty:
        df["target_date"] = pd.to_datetime(df["target_date"], errors="coerce")
        df["confidence"]  = pd.to_numeric(df["confidence"],  errors="coerce").fillna(0)
        df["was_correct"] = pd.to_numeric(df["was_correct"], errors="coerce")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS — colours, icons, popups
# ══════════════════════════════════════════════════════════════════════════════
def conf_color(c: float) -> str:
    return "#00e5ff" if c >= 0.60 else "#ffaa00" if c >= 0.40 else "#ff3333"

def conf_label(c: float) -> str:
    return "HIGH" if c >= 0.60 else "MODERATE" if c >= 0.40 else "LOW"

def hex_to_rgb(h: str):
    h = h.lstrip("#")
    return int(h[:2], 16), int(h[2:4], 16), int(h[4:6], 16)

def hour_bar_color(h: int) -> str:
    if 5 <= h < 8:   return TW_COLORS["early_morning"]
    if 8 <= h < 12:  return TW_COLORS["morning"]
    if 12 <= h < 17: return TW_COLORS["afternoon"]
    if 17 <= h < 21: return TW_COLORS["evening"]
    return TW_COLORS["night"]


def pulse_icon(conf: float) -> folium.DivIcon:
    """Pulsing tactical marker; colour and size vary with confidence."""
    color     = conf_color(conf)
    r, g, b   = hex_to_rgb(color)
    sz        = int(11 + conf * 26)
    ring      = int(sz * 2.4)
    off       = sz // 2
    html = f"""
<div style="
  width:{sz}px;height:{sz}px;border-radius:50%;
  background:rgba({r},{g},{b},0.85);
  border:1px solid rgb({r},{g},{b});
  box-shadow:0 0 {sz//2}px rgba({r},{g},{b},0.6);
  animation:kktcPulse 1.9s ease-out infinite;
  position:relative;left:-{off}px;top:-{off}px;
"></div>
<style>
@keyframes kktcPulse{{
  0%  {{box-shadow:0 0 0 0   rgba({r},{g},{b},0.6),0 0 {sz//2}px rgba({r},{g},{b},0.4);}}
  70% {{box-shadow:0 0 0 {ring}px rgba({r},{g},{b},0),0 0 {sz//2}px rgba({r},{g},{b},0.4);}}
  100%{{box-shadow:0 0 0 0   rgba({r},{g},{b},0),0 0 {sz//2}px rgba({r},{g},{b},0.4);}}
}}
</style>"""
    return folium.DivIcon(html=html, icon_size=(sz, sz), icon_anchor=(off, off))


# Folium popup styles (renders inside iframe — must use inline CSS)
_POPUP_BASE = "background:#FFFFFF;color:#1E293B;font-family:'Courier New',monospace;border:1px solid #CBD5E1;padding:10px 12px;font-size:12px;line-height:1.7;min-width:220px"

def prediction_popup(p: dict) -> folium.Popup:
    conf   = p.get("confidence", 0)
    color  = conf_color(conf)
    label  = conf_label(conf)
    dist   = p.get("district", "UNK").upper()
    area   = (p.get("specific_area") or "").upper()
    pdate  = p.get("predicted_date", "?")
    pday   = (p.get("predicted_day") or "")[:3].upper()
    ptw    = TW_LABELS.get(p.get("predicted_time_window",""), p.get("predicted_time_window","?"))
    ph     = p.get("predicted_hour")
    hstr   = f"~{int(ph):02d}:{int((ph%1)*60):02d}" if ph is not None else "?"
    cnt    = p.get("predicted_count", "?")
    repeat = "⚠ REPEAT CYCLE DETECTED" if p.get("has_temporal_pattern") else ""
    intv   = p.get("pattern_interval_days")
    bar_w  = int(conf * 100)
    html = f"""
<div style="{_POPUP_BASE}">
  <div style="color:{color};letter-spacing:2px;font-size:11px;margin-bottom:6px;font-weight:bold">◉ TARGET ACQUIRED</div>
  <div style="border-top:1px solid #CBD5E1;padding-top:6px">
    <span style="color:#1E293B;font-weight:bold">{dist}</span>
    {f'<span style="color:#475569"> / {area}</span>' if area else ''}
  </div>
  <div style="margin-top:4px;color:#1E293B">
    DATE &nbsp;<span style="color:#0F172A;font-weight:bold">{pdate}</span> {pday}<br>
    TIME &nbsp;<span style="color:#0F172A;font-weight:bold">{ptw}</span> {hstr}<br>
    COUNT&nbsp;<span style="color:#0F172A;font-weight:bold">~{cnt}</span>
  </div>
  <div style="margin-top:6px">
    <div style="background:#E2E8F0;height:8px;width:100%;border-radius:2px">
      <div style="background:{color};height:8px;width:{bar_w}%;border-radius:2px;box-shadow:0 0 4px {color}"></div>
    </div>
    <span style="color:{color};letter-spacing:1px;font-weight:bold">CONF {conf:.0%} ◆ {label}</span>
  </div>
  {f'<div style="color:#D97706;margin-top:4px;font-size:11px;font-weight:bold">⚠ {repeat} ({intv:.1f}d)</div>' if repeat and intv else ''}
</div>"""
    return folium.Popup(html, max_width=270)


def arrest_popup(row) -> str:
    dstr  = row["date"].strftime("%Y-%m-%d") if pd.notna(row["date"]) else "UNKNOWN"
    tstr  = (row.get("time") or "----")
    area  = (row.get("specific_area") or "").upper()
    dist  = row["district"].upper()
    rsn   = row.get("reason") or ""
    unit  = row.get("operating_unit") or ""
    return f"""
<div style="{_POPUP_BASE}">
  <div style="color:#0EA5E9;letter-spacing:2px;font-size:11px;margin-bottom:6px;font-weight:bold">▪ ARREST RECORD</div>
  <div style="border-top:1px solid #CBD5E1;padding-top:6px">
    <span style="color:#1E293B;font-weight:bold">{dist}</span>
    {f'<span style="color:#475569"> / {area}</span>' if area else ''}
  </div>
  <div style="margin-top:4px;color:#1E293B">
    DATE &nbsp;<span style="color:#0F172A;font-weight:bold">{dstr}</span><br>
    TIME &nbsp;<span style="color:#0F172A;font-weight:bold">{tstr}</span><br>
    COUNT&nbsp;<span style="color:#0F172A;font-weight:bold">{int(row['count'])}</span> PERSONS
    {f'<br>UNIT &nbsp;<span style="color:#475569">{unit}</span>' if unit else ''}
    {f'<br>CHGE &nbsp;<span style="color:#475569">{rsn}</span>' if rsn else ''}
  </div>
</div>"""


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
<div style="font-family:'Courier New',monospace;padding:4px 0 10px">
  <div style="color:#00e5ff;font-size:13px;letter-spacing:3px;
              text-shadow:0 0 8px #00e5ff,0 0 20px rgba(0,229,255,0.3)">
    KKTC-INTEL
  </div>
  <div style="color:#9CA3AF;font-size:9px;letter-spacing:2px;margin-top:2px">
    ENFORCEMENT ANALYTICS SYSTEM
  </div>
  <div style="margin-top:8px;font-size:9px;letter-spacing:2px;color:#9CA3AF">
    <span style="color:#00e5ff;animation:blink 1.2s step-start infinite">◉</span>
    &nbsp;SYSTEM ONLINE
  </div>
</div>""", unsafe_allow_html=True)
    st.divider()
    page = st.radio(
        "NAV",
        ["▣  TACTICAL MAP", "◈  PREDICTIONS", "◉  ANALYTICS", "◎  ACCURACY"],
        label_visibility="collapsed",
    )
    st.divider()

df_all = load_arrests()
preds  = load_predictions()

with st.sidebar:
    st.markdown(
        '<div style="font-size:9px;letter-spacing:2px;color:#9CA3AF;margin-bottom:6px">'
        'SYSTEM STATUS</div>',
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns(2)
    c1.metric("RECORDS",     len(df_all))
    c2.metric("PERSONS",     int(df_all["count"].sum()))
    c1.metric("DISTRICTS",   df_all["district"].nunique())
    c2.metric("PREDICTIONS", len(preds.get("predictions", [])))
    note = preds.get("data_stats", {}).get("model_confidence_note", "")
    if note:
        st.caption(note.upper())


def _status_bar():
    """Render a status strip below the page title."""
    sync_ts = datetime.now().strftime('%Y-%m-%d %H:%MZ')
    n_rec   = len(df_all)
    n_dst   = df_all['district'].nunique()
    n_pred  = len(preds.get('predictions', []))
    st.markdown(
        f'<div style="font-family:\'Courier New\',monospace;background:#0b1118;border:1px solid #003040;border-left:3px solid #00e5ff;padding:5px 16px;font-size:10px;color:#9CA3AF;letter-spacing:1.5px;display:flex;gap:28px;align-items:center;margin:6px 0 14px">'
        f'<span style="color:#00e5ff">&#9679; ONLINE</span>'
        f'<span>DB&nbsp;{n_rec}&nbsp;RECORDS</span>'
        f'<span>DISTRICTS&nbsp;{n_dst}</span>'
        f'<span>PREDICTIONS&nbsp;{n_pred}</span>'
        f'<span style="margin-left:auto;color:#6B7280">SYNC&nbsp;{sync_ts}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — TACTICAL MAP
# ══════════════════════════════════════════════════════════════════════════════
if page == "▣  TACTICAL MAP":
    st.title("TACTICAL ARREST DENSITY MAP")
    _status_bar()

    with st.sidebar:
        st.markdown(
            '<div style="font-size:9px;letter-spacing:2px;color:#9CA3AF;margin-bottom:6px">'
            'MAP FILTERS</div>',
            unsafe_allow_html=True,
        )
        districts_opts = ["ALL DISTRICTS"] + sorted(
            df_all["district"].dropna().unique().tolist()
        )
        sel_district = st.selectbox("DISTRICT", districts_opts)

        valid_dates = df_all["date"].dropna()
        d_min = valid_dates.min().date() if len(valid_dates) else date(2020, 1, 1)
        d_max = valid_dates.max().date() if len(valid_dates) else date.today()
        date_range = st.date_input(
            "DATE RANGE",
            value=(d_min, d_max),
            min_value=date(2018, 1, 1),
            max_value=date.today() + timedelta(days=60),
        )
        sel_time = st.selectbox("TIME WINDOW", list(TIME_WINDOWS.keys()))

        st.markdown(
            '<div style="font-size:9px;letter-spacing:2px;color:#9CA3AF;margin:10px 0 4px">'
            'LAYERS</div>',
            unsafe_allow_html=True,
        )
        show_heat    = st.checkbox("HEAT DENSITY",     value=True)
        show_pred    = st.checkbox("PREDICTION MARKERS", value=True)
        show_hist    = st.checkbox("HISTORICAL ARRESTS", value=True)
        cluster_hist = st.checkbox("CLUSTER MARKERS",    value=True)

    # ── Filter ────────────────────────────────────────────────────────────────
    df = df_all.copy()
    if sel_district != "ALL DISTRICTS":
        df = df[df["district"] == sel_district]
    if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
        d0, d1 = date_range
        df = df[(df["date"].dt.date >= d0) & (df["date"].dt.date <= d1)]
    tw_lo, tw_hi = TIME_WINDOWS[sel_time]
    if sel_time != "ALL DAY":
        no_t = df["hour_decimal"].isna()
        iw   = ((df["hour_decimal"] >= tw_lo) | (df["hour_decimal"] < (tw_hi - 24))
                if tw_hi > 24
                else (df["hour_decimal"] >= tw_lo) & (df["hour_decimal"] < tw_hi))
        df = df[no_t | iw]
    geo_df = df.dropna(subset=["latitude", "longitude"])

    # ── KPI strip ─────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("FILTERED",    len(df))
    c2.metric("PERSONS",     int(df["count"].sum()))
    c3.metric("DISTRICTS",   df["district"].nunique())
    c4.metric("GEO-LOCATED", len(geo_df))
    st.divider()

    # ── Folium map ────────────────────────────────────────────────────────────
    m = folium.Map(
        location=NC_CENTER,
        zoom_start=NC_ZOOM,
        tiles="CartoDB dark_matter",
        prefer_canvas=True,
    )

    if show_heat and not geo_df.empty:
        HeatMap(
            [[r.latitude, r.longitude, int(r["count"])] for _, r in geo_df.iterrows()],
            name="Heat Density",
            radius=24, blur=20, max_zoom=13,
            gradient={
                "0.15": "rgba(0,40,60,0)",
                "0.35": "rgba(0,80,100,0.6)",
                "0.60": "rgba(0,160,180,0.8)",
                "0.85": "rgba(0,210,230,0.9)",
                "1.00": "rgba(0,229,255,1.0)",
            },
        ).add_to(m)

    if show_hist and not geo_df.empty:
        hist_layer = (
            MarkerCluster(name="Historical Arrests", disableClusteringAtZoom=13,
                          options={"maxClusterRadius": 40})
            if cluster_hist
            else folium.FeatureGroup(name="Historical Arrests")
        )
        for _, row in geo_df.iterrows():
            sz = 5 + min(int(row["count"]) * 1.8, 14)
            folium.CircleMarker(
                location=[row.latitude, row.longitude],
                radius=sz,
                color="#005f6e",
                fill=True,
                fill_color="#00b8cc",
                fill_opacity=0.65,
                weight=1,
                popup=folium.Popup(arrest_popup(row), max_width=250),
                tooltip=(
                    f"{row['date'].strftime('%Y-%m-%d') if pd.notna(row['date']) else '?'}"
                    f"  {int(row['count'])}p  {row['district'].upper()}"
                ),
            ).add_to(hist_layer)
        hist_layer.add_to(m)

    if show_pred and preds.get("predictions"):
        pred_layer = folium.FeatureGroup(name="Predictions", show=True)
        for p in preds["predictions"][:20]:
            lat, lng = p.get("latitude"), p.get("longitude")
            if not lat or not lng:
                continue
            conf = p.get("confidence", 0)
            folium.Marker(
                location=[lat, lng],
                icon=pulse_icon(conf),
                popup=prediction_popup(p),
                tooltip=(
                    f"TGT {p.get('predicted_date','?')} "
                    f"{(p.get('specific_area') or p.get('district','')).upper()} "
                    f"CONF {conf:.0%}"
                ),
            ).add_to(pred_layer)
        pred_layer.add_to(m)

    folium.LayerControl(collapsed=False, position="topright").add_to(m)
    st_folium(m, height=570, use_container_width=True, returned_objects=[])

    # Legend strip
    st.markdown("""
<div style="font-family:'Courier New',monospace;font-size:10px;color:#9CA3AF;letter-spacing:1.5px;display:flex;gap:26px;flex-wrap:wrap;margin-top:4px;border-top:1px solid #003040;padding-top:6px">
  <span><span style="display:inline-block;width:10px;height:10px;background:#00b8cc;
    margin-right:5px;vertical-align:middle"></span>HISTORICAL ARREST</span>
  <span><span style="display:inline-block;width:10px;height:10px;background:#00e5ff;
    box-shadow:0 0 4px #00e5ff;margin-right:5px;vertical-align:middle"></span>PREDICTION HIGH ≥60%</span>
  <span><span style="display:inline-block;width:10px;height:10px;background:#ffaa00;
    box-shadow:0 0 4px #ffaa00;margin-right:5px;vertical-align:middle"></span>PREDICTION MOD 40–60%</span>
  <span><span style="display:inline-block;width:10px;height:10px;background:#ff3333;
    box-shadow:0 0 4px #ff3333;margin-right:5px;vertical-align:middle"></span>PREDICTION LOW &lt;40%</span>
  <span>SIZE ∝ COUNT/CONF</span>
</div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — PREDICTIONS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "◈  PREDICTIONS":
    st.title("ENFORCEMENT PREDICTION REPORT")
    _status_bar()

    gen_at    = preds.get("generated_at", "N/A")
    conf_note = preds.get("data_stats", {}).get("model_confidence_note", "")
    st.markdown(
        f'<div style="font-family:\'Courier New\',monospace;font-size:10px;color:#9CA3AF;letter-spacing:1px;margin-bottom:8px">'
        f'GENERATED: {gen_at.upper()} &nbsp;|&nbsp; {conf_note.upper()}</div>',
        unsafe_allow_html=True,
    )

    predictions    = preds.get("predictions",    [])
    daily_forecast = preds.get("daily_forecast", [])
    repeat_pats    = preds.get("repeat_patterns", [])
    op_pats        = preds.get("operation_patterns", {})

    if not predictions:
        st.warning("NO PREDICTIONS FOUND. RUN PREDICTOR FIRST.")
        st.stop()

    # ── TOP 5 INTELLIGENCE CARDS ──────────────────────────────────────────────
    st.subheader("PRIMARY TARGETS — TOP 5 HOTSPOTS")
    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    for i, p in enumerate(predictions[:5]):
        conf     = p.get("confidence", 0)
        color    = conf_color(conf)
        label    = conf_label(conf)
        r, g, b  = hex_to_rgb(color)
        district = (p.get("district") or "UNKNOWN").upper()
        area     = (p.get("specific_area") or "").upper()
        pdate    = p.get("predicted_date", "?")
        pday     = (p.get("predicted_day") or "")[:3].upper()
        ptw      = TW_LABELS.get(p.get("predicted_time_window",""), "?")
        ph       = p.get("predicted_hour")
        hstr     = f"~{int(ph):02d}:{int((ph%1)*60):02d}" if ph is not None else "?:??"
        cnt      = p.get("predicted_count", "?")
        hist_ev  = p.get("historical_events", 0)
        is_rep   = p.get("has_temporal_pattern", False)
        intv     = p.get("pattern_interval_days")
        bar_w    = int(conf * 100)
        loc_str  = district + (f" / {area}" if area else "")

        repeat_badge = ""
        if is_rep and intv:
            repeat_badge = (
                f'<span style="background:#200030;color:#cc44ff;border:1px solid #7700aa;'
                f'padding:1px 8px;font-size:10px;letter-spacing:1px;margin-left:10px">'
                f'⚠ REPEAT ~{intv:.1f}d</span>'
            )

        card_style = f"background:#1A1F2E;border:1px solid {color};border-left:4px solid {color};box-shadow:0 0 12px rgba({r},{g},{b},0.18),inset 0 0 30px rgba({r},{g},{b},0.04);padding:14px 18px;margin-bottom:10px;font-family:'Courier New',monospace"
        badge_style = f"color:{color};font-size:12px;letter-spacing:2px;border:1px solid {color};padding:3px 12px;text-shadow:0 0 6px {color};box-shadow:0 0 8px rgba({r},{g},{b},0.25)"
        st.markdown(
            f'<div style="{card_style}">'
            f'<div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:8px">'
            f'<div style="display:flex;align-items:center;flex-wrap:wrap;gap:0">'
            f'<span style="color:#6B7280;font-size:11px;margin-right:10px">#{i+1:02d}</span>'
            f'<span style="color:#F0F0F0;font-size:15px;letter-spacing:2px">{loc_str}</span>'
            f'{repeat_badge}'
            f'</div>'
            f'<div style="{badge_style}">{conf:.0%} &#9670; {label}</div>'
            f'</div>'
            f'<div style="border-top:1px solid #2D3748;margin:10px 0 8px"></div>'
            f'<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;font-size:13px">'
            f'<div><span style="color:#9CA3AF;letter-spacing:1px">DATE</span><br><span style="color:#F0F0F0">{pdate}</span> <span style="color:#94A3B8">{pday}</span></div>'
            f'<div><span style="color:#9CA3AF;letter-spacing:1px">WINDOW</span><br><span style="color:#F0F0F0">{ptw}</span></div>'
            f'<div><span style="color:#9CA3AF;letter-spacing:1px">TIME / COUNT</span><br><span style="color:#F0F0F0">{hstr}</span> <span style="color:#94A3B8">/ ~{cnt}p</span></div>'
            f'<div><span style="color:#9CA3AF;letter-spacing:1px">HIST EVENTS</span><br><span style="color:#F0F0F0">{hist_ev}</span></div>'
            f'</div>'
            f'<div style="margin-top:10px;display:flex;align-items:center;gap:10px">'
            f'<div style="flex:1;background:#2D3748;height:8px;border-radius:2px">'
            f'<div style="background:{color};height:8px;width:{bar_w}%;border-radius:2px;box-shadow:0 0 6px {color}"></div>'
            f'</div>'
            f'<span style="color:#F0F0F0;font-size:12px;font-weight:bold;white-space:nowrap">{bar_w}%</span>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # ── TWO-COLUMN: forecast + prediction table ───────────────────────────────
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.subheader("7-DAY ARREST FORECAST")
        if daily_forecast:
            fc_df = pd.DataFrame(daily_forecast[:7])
            fc_df["date"] = pd.to_datetime(fc_df["date"])
            fc_df["x_label"] = fc_df["date"].dt.strftime("%a<br>%b %d")
            bar_colors = [
                "#ff3333" if v >= 10 else "#ffaa00" if v >= 5 else "#00b8cc"
                for v in fc_df["predicted_count"]
            ]
            ey = None
            if "upper_bound" in fc_df.columns:
                ep = (fc_df["upper_bound"] - fc_df["predicted_count"]).clip(lower=0).tolist()
                em = (fc_df["predicted_count"] - fc_df.get("lower_bound", fc_df["predicted_count"]).clip(lower=0)).tolist()
                ey = dict(type="data", symmetric=False, array=ep, arrayminus=em,
                          thickness=1.5, width=6, color="#6B7280")

            fig_fc = dark_fig(height=360, showlegend=False)
            fig_fc.add_trace(go.Bar(
                x=fc_df["x_label"],
                y=fc_df["predicted_count"],
                marker=dict(
                    color=bar_colors,
                    line=dict(color="rgba(0,0,0,0)", width=0),
                ),
                error_y=ey,
                text=fc_df["predicted_count"].apply(lambda x: f"{x:.1f}"),
                textposition="outside",
                textfont=dict(family="'Courier New',monospace", color="#9CA3AF", size=11),
            ))
            fig_fc.update_layout(
                xaxis=dict(**CHART_BASE["xaxis"], tickangle=0),
                yaxis_title="PREDICTED ARRESTS",
            )
            st.plotly_chart(fig_fc, use_container_width=True)
            mdl = fc_df["model"].iloc[0].upper() if "model" in fc_df.columns else "UNKNOWN"
            st.caption(f"MODEL: {mdl} | CYAN <5 | AMBER 5–9 | RED ≥10 ARRESTS")
        else:
            st.info("NO FORECAST DATA")

    with col_right:
        st.subheader("ALL PREDICTIONS")
        if predictions:
            rows = []
            for p in predictions:
                conf = p.get("confidence", 0)
                rows.append({
                    "DATE":  p.get("predicted_date", "?"),
                    "DOW":   (p.get("predicted_day") or "")[:3].upper(),
                    "LOCATION": (
                        (p.get("district") or "") + " / " +
                        (p.get("specific_area") or "")
                    ).strip(" /").upper(),
                    "WINDOW":  (p.get("predicted_time_window") or "?").upper(),
                    "~CNT":    p.get("predicted_count", "?"),
                    "CONF":    f"{conf:.0%}",
                    "REP":     "✓" if p.get("has_temporal_pattern") else "",
                })
            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True, height=370, hide_index=True,
            )

    st.divider()

    # ── 7-DAY OPERATIONAL BREAKDOWN ───────────────────────────────────────────
    st.subheader("7-DAY OPERATIONAL BREAKDOWN")
    st.caption("Each day shows predicted arrest locations, time windows, coordinates and confidence. Click a day to expand.")

    _today = date.today()

    # forecast volume lookup: "YYYY-MM-DD" → predicted_count
    _fc_lookup = {
        d.get("date", ""): d.get("predicted_count", d.get("yhat", 0))
        for d in daily_forecast[:7]
    }

    # group predictions by target date (only upcoming 7 days)
    _by_date: dict = {}
    for _p in predictions:
        _raw = _p.get("predicted_date") or _p.get("target_date", "")
        try:
            _pd = datetime.strptime(_raw[:10], "%Y-%m-%d").date()
            if _today <= _pd <= _today + timedelta(days=7):
                _by_date.setdefault(_pd, []).append(_p)
        except Exception:
            pass

    # sorted forecast dates
    _fc_dates = []
    for _d in daily_forecast[:7]:
        try:
            _fc_dates.append(datetime.strptime(_d["date"], "%Y-%m-%d").date())
        except Exception:
            pass

    for _tgt in sorted(_fc_dates):
        _dname  = _tgt.strftime("%A").upper()
        _dfmt   = _tgt.strftime("%d %B %Y")
        _dkey   = _tgt.strftime("%Y-%m-%d")
        _fc_cnt = _fc_lookup.get(_dkey, 0)
        _preds  = sorted(_by_date.get(_tgt, []), key=lambda p: -p.get("confidence", 0))
        _alert  = "!! ELEVATED" if _fc_cnt >= 10 else "MODERATE" if _fc_cnt >= 5 else "NOMINAL"
        _label  = f"{_dname}  |  {_dfmt}  |  ~{_fc_cnt:.1f} arrests  |  {_alert}"

        with st.expander(_label, expanded=(_fc_cnt >= 10)):
            if not _preds:
                st.markdown(
                    f'<div style="font-family:\'Courier New\',monospace;color:#9CA3AF;'
                    f'font-size:13px;padding:8px 0">'
                    f'No specific location predictions identified for this day.<br>'
                    f'Expected volume: <span style="color:#00e5ff">~{_fc_cnt:.1f} arrests</span>'
                    f' across general patrol areas.</div>',
                    unsafe_allow_html=True,
                )
            else:
                for _i, _p in enumerate(_preds, 1):
                    _conf   = _p.get("confidence", 0)
                    _clr    = conf_color(_conf)
                    _r, _g, _b = hex_to_rgb(_clr)
                    _dist   = (_p.get("district") or "?").upper()
                    _area   = (_p.get("specific_area") or "").strip()
                    _loc    = _dist + (f" / {_area}" if _area else "")
                    _twk    = _p.get("predicted_time_window", "")
                    _tw     = TW_LABELS.get(_twk, _twk.upper() if _twk else "?")
                    _ph     = _p.get("predicted_hour")
                    _hstr   = f"{int(_ph):02d}:{int((_ph % 1) * 60):02d}" if _ph is not None else "?"
                    _cnt    = _p.get("predicted_count", "?")
                    _lat    = _p.get("latitude")
                    _lon    = _p.get("longitude")
                    _mdl    = (_p.get("model") or "?").upper()
                    _hist   = _p.get("historical_events", 0)
                    _rep    = _p.get("has_temporal_pattern", False)
                    _intv   = _p.get("pattern_interval_days")
                    _bw     = int(_conf * 100)
                    _clabel = conf_label(_conf)
                    _coords = f"{_lat:.4f}&#176;N &nbsp; {_lon:.4f}&#176;E" if _lat and _lon else "No coordinates"
                    _rep_html = (
                        f'<div style="color:#cc44ff;margin-top:6px;font-size:12px">'
                        f'&#9888; REPEAT CYCLE DETECTED &mdash; every ~{_intv:.1f} days</div>'
                        if _rep and _intv else ""
                    )
                    st.markdown(
                        f'<div style="background:#1A1F2E;border:1px solid {_clr};'
                        f'border-left:4px solid {_clr};padding:12px 16px;margin-bottom:8px;'
                        f'font-family:\'Courier New\',monospace">'
                        f'<div style="display:flex;justify-content:space-between;'
                        f'align-items:center;flex-wrap:wrap;gap:6px">'
                        f'<span style="color:#F0F0F0;font-size:14px;font-weight:bold">'
                        f'#{_i} &nbsp; {_loc}</span>'
                        f'<span style="color:{_clr};border:1px solid {_clr};'
                        f'padding:2px 10px;font-size:11px;text-shadow:0 0 6px {_clr}">'
                        f'{_conf:.0%} &#9670; {_clabel}</span>'
                        f'</div>'
                        f'<div style="border-top:1px solid #2D3748;margin:8px 0 6px"></div>'
                        f'<div style="display:grid;grid-template-columns:repeat(3,1fr);'
                        f'gap:12px;font-size:13px">'
                        f'<div><span style="color:#9CA3AF;font-size:11px">TIME WINDOW</span>'
                        f'<br><span style="color:#F0F0F0">{_tw}</span></div>'
                        f'<div><span style="color:#9CA3AF;font-size:11px">PEAK HOUR</span>'
                        f'<br><span style="color:{_clr};font-size:16px;font-weight:bold">'
                        f'~{_hstr}</span></div>'
                        f'<div><span style="color:#9CA3AF;font-size:11px">EST. COUNT</span>'
                        f'<br><span style="color:#F0F0F0">~{_cnt} person(s)</span></div>'
                        f'<div><span style="color:#9CA3AF;font-size:11px">COORDINATES</span>'
                        f'<br><span style="color:#F0F0F0">{_coords}</span></div>'
                        f'<div><span style="color:#9CA3AF;font-size:11px">PRIOR ARRESTS</span>'
                        f'<br><span style="color:#F0F0F0">{_hist} at this location</span></div>'
                        f'<div><span style="color:#9CA3AF;font-size:11px">MODEL</span>'
                        f'<br><span style="color:#F0F0F0">{_mdl}</span></div>'
                        f'</div>'
                        f'<div style="margin-top:8px;display:flex;align-items:center;gap:10px">'
                        f'<div style="flex:1;background:#2D3748;height:6px;border-radius:2px">'
                        f'<div style="background:{_clr};height:6px;width:{_bw}%;'
                        f'border-radius:2px;box-shadow:0 0 4px {_clr}"></div>'
                        f'</div>'
                        f'<span style="color:#F0F0F0;font-size:12px;white-space:nowrap">'
                        f'{_bw}%</span>'
                        f'</div>'
                        f'{_rep_html}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

    st.divider()

    # ── REPEAT PATTERNS ───────────────────────────────────────────────────────
    if repeat_pats:
        st.subheader("REPEAT RAID PATTERNS DETECTED")
        for pat in repeat_pats:
            cons  = pat.get("consistency_score", 0)
            last  = pat.get("last_raid_date", "?")
            nxt   = pat.get("predicted_next_date", "?")
            avg   = pat.get("avg_interval_days", "?")
            n     = pat.get("event_count", "?")
            area  = (pat.get("specific_area") or "").upper()
            dist  = pat["district"].upper()
            loc   = dist + (f" / {area}" if area else "")
            cbar  = int(cons * 100)
            st.markdown(
                f'<div style="background:#1A1F2E;border:1px solid #7700aa;border-left:3px solid #cc44ff;padding:10px 16px;margin-bottom:8px;font-family:\'Courier New\',monospace">'
                f'<div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px">'
                f'<span style="color:#F0F0F0;font-size:14px;letter-spacing:2px">&#9888; {loc}</span>'
                f'<span style="background:#200030;color:#cc44ff;border:1px solid #7700aa;padding:2px 10px;font-size:11px;letter-spacing:1px">{n} RAIDS</span>'
                f'</div>'
                f'<div style="border-top:1px solid #2D1B4E;margin:8px 0 6px"></div>'
                f'<div style="display:flex;gap:24px;flex-wrap:wrap;font-size:13px">'
                f'<span><span style="color:#9CA3AF">INTERVAL</span>&nbsp;<span style="color:#cc44ff;font-weight:bold">{avg}d</span></span>'
                f'<span><span style="color:#9CA3AF">LAST</span>&nbsp;<span style="color:#F0F0F0">{last}</span></span>'
                f'<span><span style="color:#9CA3AF">NEXT</span>&nbsp;<span style="color:#cc44ff;font-weight:bold">{nxt}</span></span>'
                f'<span><span style="color:#9CA3AF">CONSISTENCY</span>&nbsp;<span style="color:#cc44ff;font-weight:bold">{cons:.0%}</span></span>'
                f'</div>'
                f'<div style="margin-top:8px;display:flex;align-items:center;gap:10px">'
                f'<div style="flex:1;background:#2D3748;height:8px;border-radius:2px">'
                f'<div style="background:#cc44ff;height:8px;width:{cbar}%;border-radius:2px;box-shadow:0 0 5px #cc44ff"></div>'
                f'</div>'
                f'<span style="color:#F0F0F0;font-size:12px;font-weight:bold;white-space:nowrap">{cbar}%</span>'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ── OPERATION DAYS ────────────────────────────────────────────────────────
    op_days = op_pats.get("operation_days", [])
    if op_days:
        st.subheader("MAJOR OPERATION DAYS — HISTORICAL")
        col1, col2 = st.columns([2, 1])
        with col1:
            op_df = pd.DataFrame(op_days)
            op_df["date"] = pd.to_datetime(op_df["date"])
            op_df = op_df.sort_values("date", ascending=False).head(10)
            for _, row in op_df.iterrows():
                lvl   = row["total_people"]
                color = "#ff3333" if lvl >= 10 else "#ffaa00" if lvl >= 5 else "#00b8cc"
                dot   = "▰▰▰" if lvl >= 10 else "▰▰░" if lvl >= 5 else "▰░░"
                st.markdown(
                    f'<div style="font-family:\'Courier New\',monospace;font-size:13px;color:#9CA3AF;padding:3px 0;letter-spacing:1px">'
                    f'<span style="color:{color}">{dot}</span>&nbsp;'
                    f'<span style="color:#F0F0F0">{row["date"].strftime("%Y-%m-%d")}</span>'
                    f'&nbsp;({row["dow"][:3].upper()})&nbsp;&#8212;&nbsp;'
                    f'{int(row["events"])} EVT&nbsp;{int(row["districts"])} DST&nbsp;'
                    f'<span style="color:{color};font-weight:bold">{int(row["total_people"])}p</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        with col2:
            pref = op_pats.get("preferred_days", {})
            if pref:
                pref_df = pd.DataFrame(
                    sorted(pref.items(), key=lambda x: -x[1]),
                    columns=["Day", "Ops"],
                )
                fig_p = dark_fig(height=260, showlegend=False)
                fig_p.add_trace(go.Bar(
                    x=pref_df["Day"],
                    y=pref_df["Ops"],
                    marker_color="#6040a0",
                    text=pref_df["Ops"],
                    textposition="outside",
                    textfont=dict(color="#9CA3AF", size=10),
                ))
                fig_p.update_layout(
                    title=dict(text="OPS BY DAY", font=dict(color="#9CA3AF", size=10),
                               x=0, pad=dict(l=0)),
                    margin=dict(t=30, b=10, l=10, r=10),
                )
                st.plotly_chart(fig_p, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "◉  ANALYTICS":
    st.title("ENFORCEMENT ANALYTICS")
    _status_bar()

    df = df_all.copy()

    # ── KPI row ───────────────────────────────────────────────────────────────
    total_recs   = len(df)
    total_people = int(df["count"].sum())
    n_dist       = df["district"].nunique()
    vd           = df["date"].dropna()
    span         = (vd.max() - vd.min()).days if len(vd) else 0
    avg_ev       = total_people / total_recs if total_recs else 0

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("TOTAL RECORDS",   total_recs)
    k2.metric("PERSONS ARRESTED", total_people)
    k3.metric("DISTRICTS",       n_dist)
    k4.metric("DATE SPAN",       f"{span}d")
    k5.metric("AVG PER EVENT",   f"{avg_ev:.1f}")
    st.divider()

    # helper to apply CHART_BASE to px figures
    def _px_dark(fig, height=380):
        fig.update_layout(**{
            **CHART_BASE,
            "height": height,
            "coloraxis_showscale": False,
        })
        return fig

    # ── ROW 1: District  +  Day of week ──────────────────────────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("ARRESTS BY DISTRICT")
        by_dist = (
            df.groupby("district", dropna=True)["count"].sum()
            .reset_index().rename(columns={"count": "people"})
            .sort_values("people", ascending=False)
        )
        fig_d = _px_dark(px.bar(
            by_dist, x="district", y="people",
            color="people",
            color_continuous_scale=["#003040", "#00e5ff"],
            labels={"district": "", "people": "PERSONS"},
            text="people",
        ))
        fig_d.update_traces(
            textposition="outside",
            textfont=dict(color="#9CA3AF", size=10, family="'Courier New',monospace"),
            marker_line_width=0,
        )
        fig_d.update_layout(xaxis=dict(**CHART_BASE["xaxis"], tickangle=-30))
        st.plotly_chart(fig_d, use_container_width=True)

    with col_b:
        st.subheader("ARRESTS BY DAY OF WEEK")
        df_dow = df.copy()
        df_dow["day_of_week"] = pd.Categorical(
            df_dow["day_of_week"], categories=DAY_ORDER, ordered=True
        )
        by_day = (
            df_dow.groupby("day_of_week", observed=False)["count"].sum()
            .reset_index().rename(columns={"count": "people"})
        )
        fig_day = _px_dark(px.bar(
            by_day, x="day_of_week", y="people",
            color="people",
            color_continuous_scale=["#002820", "#00e5a0"],
            labels={"day_of_week": "", "people": "PERSONS"},
            text="people",
        ))
        fig_day.update_traces(
            textposition="outside",
            textfont=dict(color="#9CA3AF", size=10, family="'Courier New',monospace"),
            marker_line_width=0,
        )
        st.plotly_chart(fig_day, use_container_width=True)

    # ── ROW 2: Hour histogram  +  Monthly trend ───────────────────────────────
    col_c, col_d = st.columns(2)

    with col_c:
        st.subheader("ARRESTS BY HOUR OF DAY")
        df_h = df.dropna(subset=["hour_decimal"]).copy()
        df_h["hour_int"] = df_h["hour_decimal"].astype(int)
        by_hour = (
            df_h.groupby("hour_int")["count"].sum()
            .reindex(range(24), fill_value=0).reset_index()
            .rename(columns={"hour_int": "hour", "count": "people"})
        )
        fig_h = dark_fig(height=380)
        fig_h.add_trace(go.Bar(
            x=[f"{h:02d}:00" for h in by_hour["hour"]],
            y=by_hour["people"],
            marker=dict(
                color=[hour_bar_color(h) for h in by_hour["hour"]],
                line=dict(width=0),
            ),
            text=by_hour["people"].apply(lambda x: str(int(x)) if x > 0 else ""),
            textposition="outside",
            textfont=dict(color="#9CA3AF", size=10),
        ))
        fig_h.update_layout(
            xaxis=dict(**CHART_BASE["xaxis"], tickangle=-45),
            yaxis_title="PERSONS",
        )
        st.plotly_chart(fig_h, use_container_width=True)
        st.caption(
            "EARLY AM 05–08 | MORNING 08–12 | AFTERNOON 12–17 | "
            "EVENING 17–21 | NIGHT 21–05"
        )

    with col_d:
        st.subheader("MONTHLY ARREST TREND")
        df_m = df.dropna(subset=["date"]).copy()
        df_m["ym"] = df_m["date"].dt.to_period("M").dt.to_timestamp()
        by_m = (
            df_m.groupby("ym")
            .agg(events=("id","count"), people=("count","sum"))
            .reset_index()
        )
        fig_line = dark_fig(height=380)
        fig_line.add_trace(go.Scatter(
            x=by_m["ym"], y=by_m["people"],
            mode="lines+markers+text",
            name="PERSONS",
            line=dict(color="#00e5ff", width=2),
            marker=dict(size=7, color="#00e5ff",
                        line=dict(color="#060a0d", width=1)),
            fill="tozeroy",
            fillcolor="rgba(0,229,255,0.06)",
            text=by_m["people"].astype(str),
            textposition="top center",
            textfont=dict(color="#9CA3AF", size=10),
        ))
        fig_line.add_trace(go.Scatter(
            x=by_m["ym"], y=by_m["events"],
            mode="lines+markers",
            name="EVENTS",
            line=dict(color="#005f6e", width=1.5, dash="dot"),
            marker=dict(size=5, color="#005f6e"),
        ))
        fig_line.update_layout(
            xaxis_title="MONTH", yaxis_title="COUNT",
            legend=dict(**CHART_BASE["legend"]),
        )
        st.plotly_chart(fig_line, use_container_width=True)

    # ── ROW 3: Time window  +  Hotspot table ─────────────────────────────────
    col_e, col_f = st.columns([1, 1])

    with col_e:
        st.subheader("ARRESTS BY TIME WINDOW")
        df_tw = df.dropna(subset=["time_window"]).copy()
        by_tw = (
            df_tw.groupby("time_window")["count"].sum().reset_index()
            .rename(columns={"count": "people", "time_window": "window"})
        )
        tw_cat = ["early_morning","morning","afternoon","evening","night"]
        by_tw["window"] = pd.Categorical(by_tw["window"], categories=tw_cat, ordered=True)
        by_tw = by_tw.sort_values("window")
        by_tw["label"] = by_tw["window"].map(TW_LABELS).fillna(by_tw["window"].astype(str))
        by_tw["color"] = by_tw["window"].astype(str).map(TW_COLORS).fillna("#6B7280")

        fig_tw = dark_fig(height=320)
        fig_tw.add_trace(go.Bar(
            x=by_tw["label"],
            y=by_tw["people"],
            marker=dict(color=by_tw["color"].tolist(), line=dict(width=0)),
            text=by_tw["people"],
            textposition="outside",
            textfont=dict(color="#9CA3AF", size=11),
        ))
        fig_tw.update_layout(
            xaxis=dict(**CHART_BASE["xaxis"], tickangle=-15),
            yaxis_title="PERSONS",
        )
        st.plotly_chart(fig_tw, use_container_width=True)

    with col_f:
        st.subheader("TOP 10 HOTSPOT AREAS")
        top_areas = (
            df[df["specific_area"].notna() & (df["specific_area"].str.strip() != "")]
            .groupby(["district", "specific_area"])
            .agg(events=("id","count"), people=("count","sum"))
            .reset_index().sort_values("people", ascending=False)
            .head(10).reset_index(drop=True)
        )
        top_areas.index += 1
        top_areas.columns = ["DISTRICT", "AREA", "EVENTS", "PERSONS"]
        top_areas["PERSONS"] = top_areas["PERSONS"].apply(lambda x: f"{x:,}")
        top_areas["EVENTS"]  = top_areas["EVENTS"].apply(lambda x: f"{x:,}")
        st.dataframe(top_areas, use_container_width=True, height=340)

    # ── ROW 4: Season  +  Weekend/Weekday ────────────────────────────────────
    col_g, col_h = st.columns(2)

    with col_g:
        st.subheader("ARRESTS BY SEASON")
        by_s = (
            df.dropna(subset=["season"]).groupby("season")["count"].sum()
            .reset_index().rename(columns={"count": "people"})
        )
        by_s["color"] = by_s["season"].map(SEASON_COLORS).fillna("#6B7280")
        fig_s = dark_fig(height=300)
        fig_s.add_trace(go.Bar(
            x=by_s["season"].str.upper(),
            y=by_s["people"],
            marker=dict(color=by_s["color"].tolist(), line=dict(width=0)),
            text=by_s["people"],
            textposition="outside",
            textfont=dict(color="#9CA3AF", size=11),
        ))
        fig_s.update_layout(yaxis_title="PERSONS")
        st.plotly_chart(fig_s, use_container_width=True)

    with col_h:
        st.subheader("WEEKEND VS WEEKDAY")
        by_w = (
            df.dropna(subset=["is_weekend"]).groupby("is_weekend")["count"].sum()
            .reset_index().rename(columns={"count": "people"})
        )
        by_w["label"] = by_w["is_weekend"].map({0:"WEEKDAY MON–FRI", 1:"WEEKEND SAT–SUN"})
        fig_pie = dark_fig(height=300)
        fig_pie.add_trace(go.Pie(
            labels=by_w["label"],
            values=by_w["people"],
            hole=0.44,
            marker=dict(
                colors=["#005f6e", "#003040"],
                line=dict(color="#00e5ff", width=1),
            ),
            textinfo="label+percent",
            textfont=dict(family="'Courier New',monospace", color="#E0E0E0", size=11),
        ))
        fig_pie.update_layout(showlegend=False, margin=dict(t=20,b=10,l=10,r=10))
        st.plotly_chart(fig_pie, use_container_width=True)

    # ── ROW 5: Arresting units ────────────────────────────────────────────────
    units = df.dropna(subset=["operating_unit"])
    units = units[units["operating_unit"].str.strip() != ""]
    if not units.empty:
        st.subheader("TOP ARRESTING UNITS")
        by_u = (
            units.groupby("operating_unit")["count"].sum().reset_index()
            .rename(columns={"count":"people","operating_unit":"unit"})
            .sort_values("people", ascending=True).tail(12)
        )
        fig_u = dark_fig(height=max(220, len(by_u) * 32))
        fig_u.add_trace(go.Bar(
            x=by_u["people"], y=by_u["unit"],
            orientation="h",
            marker=dict(
                color=by_u["people"],
                colorscale=[[0,"#003040"],[1,"#00e5ff"]],
                line=dict(width=0),
            ),
            text=by_u["people"],
            textposition="outside",
            textfont=dict(color="#9CA3AF", size=10),
        ))
        fig_u.update_layout(
            xaxis_title="PERSONS",
            margin=dict(t=10, b=10, l=10, r=60),
        )
        st.plotly_chart(fig_u, use_container_width=True)

    # ── Raw records expander ──────────────────────────────────────────────────
    with st.expander("RAW ARREST RECORDS", expanded=False):
        show_cols = [c for c in
            ["date","time","district","specific_area","count","reason",
             "day_of_week","time_window","operating_unit","source_name"]
            if c in df.columns]
        raw = df[show_cols].copy()
        raw["date"] = raw["date"].dt.strftime("%Y-%m-%d")
        raw.columns = [c.upper() for c in raw.columns]
        st.dataframe(raw, use_container_width=True, height=400)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — ACCURACY
# ══════════════════════════════════════════════════════════════════════════════
elif page == "◎  ACCURACY":
    st.title("PREDICTION ACCURACY TRACKER")
    _status_bar()

    acc_df = load_accuracy_data()

    if acc_df.empty:
        st.warning("NO EVALUATED PREDICTIONS YET.")
        st.info(
            "Run the accuracy tracker to score last week's predictions:\n\n"
            "```\npython accuracy.py\n```\n\n"
            "Or for a specific date range:\n\n"
            "```\npython accuracy.py check --from 2026-02-01 --to 2026-02-28\n```"
        )
        st.stop()

    # ── Big KPI numbers ────────────────────────────────────────────────────
    total_eval   = len(acc_df)
    total_hits   = int(acc_df["was_correct"].sum())
    total_misses = total_eval - total_hits
    hit_rate     = total_hits / total_eval if total_eval > 0 else 0.0

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("EVALUATED",  total_eval)
    k2.metric("HITS ✓",     total_hits)
    k3.metric("MISSES ✗",   total_misses)
    k4.metric("HIT RATE",   f"{hit_rate:.0%}")
    st.divider()

    # ── Running hit rate over time ─────────────────────────────────────────
    st.subheader("RUNNING HIT RATE — WEEKLY")
    wk_df = acc_df.dropna(subset=["target_date"]).copy()
    wk_df["week"] = wk_df["target_date"].dt.to_period("W").dt.start_time
    wk_stats = (
        wk_df.groupby("week")
        .agg(total=("was_correct", "count"), hits=("was_correct", "sum"))
        .reset_index()
    )
    wk_stats["rate"] = wk_stats["hits"] / wk_stats["total"]

    if len(wk_stats) >= 2:
        fig_rate = dark_fig(height=300)
        fig_rate.add_trace(go.Scatter(
            x=wk_stats["week"],
            y=wk_stats["rate"],
            mode="lines+markers+text",
            name="HIT RATE",
            line=dict(color="#00e5ff", width=2),
            marker=dict(size=8, color="#00e5ff", line=dict(color="#060a0d", width=1)),
            fill="tozeroy",
            fillcolor="rgba(0,229,255,0.06)",
            text=[f"{r:.0%}" for r in wk_stats["rate"]],
            textposition="top center",
            textfont=dict(color="#9CA3AF", size=10),
        ))
        fig_rate.add_hline(
            y=0.5,
            line=dict(color="#ffaa00", width=1, dash="dot"),
            annotation_text="50% baseline",
            annotation_font_color="#ffaa00",
        )
        fig_rate.update_layout(
            yaxis=dict(
                **CHART_BASE["yaxis"],
                tickformat=".0%",
                range=[0, 1.1],
                title="HIT RATE",
            ),
        )
        st.plotly_chart(fig_rate, use_container_width=True)
    else:
        st.info(
            f"WEEKLY TREND NEEDS ≥ 2 WEEKS OF DATA — "
            f"currently {len(wk_stats)} week(s) evaluated."
        )

    st.divider()

    # ── Accuracy by district  +  by day of week ───────────────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("ACCURACY BY DISTRICT")
        dist_grp = (
            acc_df.groupby("district")
            .agg(total=("was_correct", "count"), hits=("was_correct", "sum"))
            .reset_index()
        )
        dist_grp["rate"]  = dist_grp["hits"] / dist_grp["total"]
        dist_grp          = dist_grp.sort_values("rate", ascending=False)
        dist_grp["color"] = dist_grp["rate"].apply(
            lambda r: "#00e5ff" if r >= 0.6 else "#ffaa00" if r >= 0.4 else "#ff3333"
        )

        fig_da = dark_fig(height=380)
        fig_da.add_trace(go.Bar(
            x=dist_grp["district"],
            y=dist_grp["rate"],
            marker=dict(color=dist_grp["color"].tolist(), line=dict(width=0)),
            text=[f"{r:.0%}" for r in dist_grp["rate"]],
            textposition="outside",
            textfont=dict(color="#9CA3AF", size=10),
            customdata=dist_grp[["hits", "total"]].values,
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Hit rate: %{y:.0%}<br>"
                "Hits: %{customdata[0]} / %{customdata[1]}<extra></extra>"
            ),
        ))
        fig_da.add_hline(y=0.5, line=dict(color="#ffaa00", width=1, dash="dot"))
        fig_da.update_layout(
            xaxis=dict(**CHART_BASE["xaxis"], tickangle=-30),
            yaxis=dict(**CHART_BASE["yaxis"], tickformat=".0%", range=[0, 1.15],
                       title="HIT RATE"),
        )
        st.plotly_chart(fig_da, use_container_width=True)

    with col_b:
        st.subheader("ACCURACY BY DAY OF WEEK")
        dow_df = acc_df.dropna(subset=["target_date"]).copy()
        dow_df["dow"] = dow_df["target_date"].dt.day_name()
        dow_df["dow"] = pd.Categorical(dow_df["dow"], categories=DAY_ORDER, ordered=True)
        dow_grp = (
            dow_df.groupby("dow", observed=False)
            .agg(total=("was_correct", "count"), hits=("was_correct", "sum"))
            .reset_index()
        )
        dow_grp["rate"] = (
            dow_grp["hits"] / dow_grp["total"].replace(0, float("nan"))
        )
        dow_grp["color"] = dow_grp["rate"].apply(
            lambda r: "#00e5ff"  if (pd.notna(r) and r >= 0.6) else
                      "#ffaa00"  if (pd.notna(r) and r >= 0.4) else
                      "#ff3333"  if pd.notna(r) else "#2D3748"
        )

        fig_dow = dark_fig(height=380)
        fig_dow.add_trace(go.Bar(
            x=dow_grp["dow"].astype(str),
            y=dow_grp["rate"].fillna(0),
            marker=dict(color=dow_grp["color"].tolist(), line=dict(width=0)),
            text=[f"{r:.0%}" if pd.notna(r) and r > 0 else "" for r in dow_grp["rate"]],
            textposition="outside",
            textfont=dict(color="#9CA3AF", size=10),
        ))
        fig_dow.add_hline(y=0.5, line=dict(color="#ffaa00", width=1, dash="dot"))
        fig_dow.update_layout(
            yaxis=dict(**CHART_BASE["yaxis"], tickformat=".0%", range=[0, 1.15],
                       title="HIT RATE"),
        )
        st.plotly_chart(fig_dow, use_container_width=True)

    st.divider()

    # ── Last 10 evaluated predictions ─────────────────────────────────────
    st.subheader("LAST 10 PREDICTIONS — VERDICT")
    recent = acc_df.head(10).copy()

    for _, row in recent.iterrows():
        is_hit       = row["was_correct"] == 1
        badge_color  = "#00e5ff" if is_hit else "#ff3333"
        badge_label  = "HIT  &#10003;" if is_hit else "MISS &#10007;"
        conf         = float(row.get("confidence") or 0)
        c_color      = conf_color(conf)
        district     = (row.get("district")      or "?").upper()
        area         = (row.get("specific_area") or "").upper()
        loc          = district + (f" / {area}" if area else "")
        tw_key       = row.get("predicted_time_window") or ""
        tw_label     = TW_LABELS.get(tw_key, tw_key.upper() or "?")
        pdate        = (
            row["target_date"].strftime("%Y-%m-%d")
            if pd.notna(row["target_date"]) else "?"
        )
        pred_id      = int(row["id"])

        st.markdown(
            f'<div style="background:#1A1F2E;border:1px solid {badge_color};'
            f'border-left:4px solid {badge_color};padding:9px 16px;'
            f'margin-bottom:6px;font-family:\'Courier New\',monospace;'
            f'display:flex;align-items:center;gap:16px;flex-wrap:wrap">'
            f'<span style="color:{badge_color};font-size:13px;font-weight:bold;'
            f'min-width:80px">{badge_label}</span>'
            f'<span style="color:#9CA3AF;font-size:11px;min-width:30px">#{pred_id:03d}</span>'
            f'<span style="color:#F0F0F0;font-size:14px">{loc}</span>'
            f'<span style="color:#9CA3AF;font-size:12px">{pdate}</span>'
            f'<span style="color:#9CA3AF;font-size:12px">{tw_label}</span>'
            f'<span style="color:{c_color};font-size:12px;margin-left:auto">'
            f'CONF {conf:.0%}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Last evaluated timestamp + help text ──────────────────────────────
    last_tgt = acc_df["target_date"].max()
    last_str = last_tgt.strftime("%Y-%m-%d") if pd.notna(last_tgt) else "?"
    st.markdown(
        f'<div style="font-family:\'Courier New\',monospace;font-size:10px;'
        f'color:#6B7280;letter-spacing:1px;margin-top:14px;'
        f'border-top:1px solid #0c1e28;padding-top:8px;'
        f'display:flex;justify-content:space-between;flex-wrap:wrap;gap:8px">'
        f'<span>LAST EVALUATED TARGET DATE: {last_str}</span>'
        f'<span>UPDATE: python accuracy.py &nbsp;|&nbsp; '
        f'OVERRIDE: python accuracy.py mark --prediction-id N --correct yes/no</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="
  font-family:'Courier New',Courier,monospace;
  margin-top:48px;
  padding: 14px 0 8px;
  border-top: 1px solid #003040;
  text-align: center;
  letter-spacing: 2px;
  font-size: 11px;
  color: #9CA3AF;
">
  made with
  <span style="
    color:#ff3355;
    text-shadow: 0 0 6px rgba(255,51,85,0.7), 0 0 14px rgba(255,51,85,0.3);
    font-size: 13px;
    vertical-align: middle;
  ">&#9829;</span>
  by
  <span style="
    color: #00e5ff;
    text-shadow: 0 0 8px #00e5ff, 0 0 20px rgba(0,229,255,0.3);
    letter-spacing: 3px;
  ">solomon Egheose</span>
</div>
""", unsafe_allow_html=True)
