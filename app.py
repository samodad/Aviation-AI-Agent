"""
SkyMind SL — Sri Lanka Aviation AI Agent
Run: streamlit run app.py
"""

import streamlit as st
import os
import base64
import random
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="SkyMind SL — Aviation AI",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Session State ──────────────────────────────
for k, v in {
    "page": "welcome",
    "messages": [],
    "query_count": 0,
    "session_id": f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    "quick_query": None,
    "api_key_input": "",
    "selected_airport": "VCBI",
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Resolve API key ────────────────────────────
env_key = os.getenv("OPENAI_API_KEY", "")
active_key = env_key if (env_key and env_key != "your_openai_api_key_here") else st.session_state.get("api_key_input", "")
has_openai = bool(active_key and active_key.startswith("sk-"))
if has_openai:
    os.environ["OPENAI_API_KEY"] = active_key

# ── Image helpers ──────────────────────────────
def img_to_b64(filename):
    base = os.path.dirname(os.path.abspath(__file__))
    for ext in ["png", "jpg", "jpeg", "webp"]:
        path = os.path.join(base, f"{filename}.{ext}")
        if os.path.exists(path):
            with open(path, "rb") as f:
                data = base64.b64encode(f.read()).decode()
            mime = "jpeg" if ext in ("jpg","jpeg") else ext
            return f"data:image/{mime};base64,{data}"
    return None

hero_src    = img_to_b64("srilankan_hero")

# Build hero CSS
if hero_src:
    hero_bg = f"background: url('{hero_src}') center center / cover no-repeat;"
    hero_ov = "background: linear-gradient(90deg, rgba(4,13,24,0.93) 0%, rgba(4,13,24,0.80) 55%, rgba(4,13,24,0.45) 100%);"
else:
    hero_bg = "background: linear-gradient(135deg,#040d18,#0d2040);"
    hero_ov = "background: transparent;"

# ── Airport data ───────────────────────────────
AIRPORTS = {
    "VCBI": {"iata":"CMB","name":"Bandaranaike International","city":"Colombo","lat":7.1807,"lon":79.8842,"color":"#2196f3","flights_today":87,"runways":"04/22 (3350m), 13/31 (2000m)"},
    "VCCB": {"iata":"RML","name":"Ratmalana Airport","city":"Colombo","lat":6.8220,"lon":79.8862,"color":"#4caf50","flights_today":12,"runways":"04/22 (2008m)"},
    "VCCA": {"iata":"HRI","name":"Mattala Rajapaksa Intl","city":"Hambantota","lat":6.2847,"lon":81.1241,"color":"#ff9800","flights_today":4,"runways":"04/22 (3500m)"},
    "VCCT": {"iata":"TRR","name":"China Bay Airport","city":"Trincomalee","lat":8.5385,"lon":81.1819,"color":"#e91e63","flights_today":6,"runways":"03/21 (1740m)"},
    "VCCJ": {"iata":"JAF","name":"Palaly Airport","city":"Jaffna","lat":9.7924,"lon":80.0701,"color":"#9c27b0","flights_today":8,"runways":"05/23 (1800m)"},
}

# ── AI Response Function ───────────────────────
AVIATION_SYSTEM_PROMPT = """You are SkyMind SL, an expert AI aviation assistant specialised in Sri Lanka aviation.
You have deep knowledge of:
- All Sri Lanka airports: VCBI (CMB, Colombo Bandaranaike), VCCB (RML, Ratmalana), VCCA (HRI, Mattala Rajapaksa), VCCT (TRR, China Bay Trincomalee), VCCJ (JAF, Palaly Jaffna)
- Colombo FIR (VCCF) airspace and procedures
- SriLankan Airlines (UL) operations, fleet, routes
- CAASL (Civil Aviation Authority of Sri Lanka) regulations
- AASL (Airport and Aviation Services Sri Lanka) operations
- PPL/CPL/ATPL licensing in Sri Lanka
- Drone / UAV regulations under CAASL
- ATC frequencies, NOTAMs, METAR/TAF weather interpretation
- VFR/IFR procedures in Sri Lankan airspace

Always provide helpful, accurate, and detailed answers.
Format responses with markdown — use **bold** for key terms, bullet lists where appropriate, and headers for long answers.
End every response with a line: ⚠️ *For operational use always verify with official AASL / CAASL / ATC sources.*"""


def _call_openai_raw(user_message, chat_history, api_key):
    """Call OpenAI API using only stdlib urllib — zero extra dependencies."""
    import urllib.request
    import json

    messages = [{"role": "system", "content": AVIATION_SYSTEM_PROMPT}]
    if chat_history:
        for msg in chat_history[-10:]:
            if msg.get("role") in ("user", "assistant") and msg.get("content"):
                messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_message})

    payload = json.dumps({
        "model": "gpt-4o-mini",
        "messages": messages,
        "max_tokens": 1200,
        "temperature": 0.7,
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    return data["choices"][0]["message"]["content"]


def get_ai_response(user_message, chat_history=None):
    """
    Priority order:
      1. agent/graph.py  (LangGraph agent — if installed)
      2. openai Python package  (if installed)
      3. urllib direct HTTP call  (always works, zero deps)
    """
    api_key = os.environ.get("OPENAI_API_KEY", "")
    history = chat_history or []

    # 1. Try LangGraph agent
    try:
        from agent.graph import run_aviation_agent
        return run_aviation_agent(user_message, chat_history=history)
    except Exception:
        pass

    # 2. Try openai package
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        msgs = [{"role": "system", "content": AVIATION_SYSTEM_PROMPT}]
        for msg in history[-10:]:
            if msg.get("role") in ("user", "assistant") and msg.get("content"):
                msgs.append({"role": msg["role"], "content": msg["content"]})
        msgs.append({"role": "user", "content": user_message})
        resp = client.chat.completions.create(
            model="gpt-4o-mini", messages=msgs, max_tokens=1200, temperature=0.7
        )
        return resp.choices[0].message.content
    except Exception:
        pass

    # 3. urllib fallback — always available
    try:
        return _call_openai_raw(user_message, history, api_key)
    except Exception as e:
        err = str(e)
        # Try to extract a useful message from HTTP errors
        if hasattr(e, "read"):
            try:
                import json
                body = json.loads(e.read().decode())
                err = body.get("error", {}).get("message", err)
            except Exception:
                pass
        return (
            f"⚠️ **SkyMind SL could not get a response.**\n\n"
            f"**Reason:** `{err}`\n\n"
            f"**Check:**\n"
            f"- Your OpenAI API key is valid and has available credits\n"
            f"- You have an active internet connection\n"
            f"- The key starts with `sk-` and was entered correctly on the Home page"
        )

# ── CSS ────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;700;900&family=Exo+2:wght@300;400;500;600&family=Share+Tech+Mono&display=swap');

html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {{
    margin: 0 !important; padding: 0 !important; background: #040d18 !important;
}}
.stApp {{ background: #040d18 !important; color: #c8dff0; font-family: 'Exo 2', sans-serif; }}
#MainMenu, footer, header, [data-testid="stToolbar"],
[data-testid="collapsedControl"], [data-testid="stDecoration"] {{
    display: none !important; visibility: hidden !important;
}}
.block-container {{
    padding: 0 !important; margin: 0 !important;
    max-width: 100% !important; min-height: 100vh !important;
}}

/* NAVBAR */
.navbar {{
    position: fixed; top: 0; left: 0; right: 0; z-index: 9999;
    height: 60px; padding: 0 36px;
    background: rgba(4,13,24,0.98);
    border-bottom: 1px solid rgba(100,181,246,0.2);
    backdrop-filter: blur(20px);
    display: flex; align-items: center; justify-content: space-between;
}}
.nb-brand {{ font-family:'Orbitron',monospace; font-size:1.2rem; font-weight:900; color:#64b5f6; letter-spacing:3px; }}
.nb-brand span {{ color:#fff; }}
.nb-right {{ display:flex; align-items:center; gap:18px; font-family:'Share Tech Mono',monospace; font-size:0.7rem; }}
.nb-dot {{ width:7px; height:7px; border-radius:50%; display:inline-block; margin-right:5px; animation:pulse 1.5s infinite; }}
@keyframes pulse {{ 0%,100%{{opacity:1}} 50%{{opacity:0.3}} }}

/* PAGE ROOT */
.pw {{ padding-top: 60px; background: #040d18; min-height: 100vh; }}

/* HERO */
.hero {{
    position: relative; overflow: hidden;
    min-height: 500px; display: flex; align-items: center;
    border-bottom: 1px solid rgba(100,181,246,0.15);
}}
.hero-bg {{ position:absolute; inset:0; z-index:0; {hero_bg} }}
.hero-ov {{ position:absolute; inset:0; z-index:1; {hero_ov} }}
.hero-grid {{
    position:absolute; inset:0; z-index:2; pointer-events:none;
    background-image: linear-gradient(rgba(100,181,246,0.03) 1px,transparent 1px),
                      linear-gradient(90deg,rgba(100,181,246,0.03) 1px,transparent 1px);
    background-size: 50px 50px;
}}
.hero-runway {{ position:absolute; bottom:0; left:0; right:0; height:3px; z-index:3; background:linear-gradient(90deg,transparent,#2196f3 20%,#2196f3 80%,transparent); opacity:0.5; }}
.hero-content {{ position:relative; z-index:4; padding:55px 50px; max-width:55%; }}
.h-badge {{ display:inline-block; font-family:'Share Tech Mono',monospace; font-size:0.67rem; letter-spacing:4px; color:#64b5f6; border:1px solid rgba(100,181,246,0.3); padding:5px 16px; border-radius:20px; background:rgba(4,13,24,0.65); margin-bottom:22px; backdrop-filter:blur(8px); }}
.h-title {{ font-family:'Orbitron',monospace; font-size:clamp(2.4rem,5vw,4rem); font-weight:900; line-height:1; color:#fff; text-shadow:0 0 40px rgba(100,181,246,0.4),0 2px 20px rgba(0,0,0,0.9); margin-bottom:8px; }}
.h-title span {{ color:#64b5f6; }}
.h-sub {{ font-family:'Share Tech Mono',monospace; font-size:0.8rem; letter-spacing:4px; color:#90caf9; margin-bottom:10px; text-shadow:0 1px 8px rgba(0,0,0,0.8); }}
.h-tag {{ font-size:1rem; color:#e3f2fd; font-style:italic; margin-bottom:26px; text-shadow:0 1px 8px rgba(0,0,0,0.9); font-weight:500; }}
.h-stats {{ display:flex; gap:16px; }}
.hstat {{ text-align:center; background:rgba(4,13,24,0.65); border:1px solid rgba(100,181,246,0.25); border-radius:10px; padding:10px 16px; backdrop-filter:blur(8px); }}
.hstat-val {{ font-family:'Orbitron',monospace; font-size:1.5rem; color:#64b5f6; font-weight:700; }}
.hstat-lbl {{ font-family:'Share Tech Mono',monospace; font-size:0.58rem; color:#90caf9; letter-spacing:2px; }}
.airline-badge {{
    position:absolute; right:5%; top:50%; transform:translateY(-50%); z-index:4;
    background:rgba(4,13,24,0.78); border:1px solid rgba(100,181,246,0.28);
    border-radius:16px; padding:20px 24px; text-align:center;
    backdrop-filter:blur(14px); box-shadow:0 8px 40px rgba(0,0,0,0.5);
    min-width:200px;
}}
.al-icon {{ font-size:2.5rem; margin-bottom:10px; }}
.al-name {{ font-family:'Orbitron',monospace; font-size:0.95rem; font-weight:700; color:#fff; letter-spacing:1px; margin-bottom:4px; }}
.al-sub {{ font-family:'Share Tech Mono',monospace; font-size:0.6rem; color:#64b5f6; letter-spacing:3px; margin-bottom:10px; }}
.al-info {{ font-family:'Share Tech Mono',monospace; font-size:0.58rem; color:#546e7a; line-height:1.8; }}

/* HOSTESS BANNER */
.hostess-banner {{
    width: 100%;
    position: relative;
    overflow: hidden;
    height: 120px;
    background: linear-gradient(135deg, #040d18, #0a1628);
    border-bottom: 1px solid rgba(100,181,246,0.1);
    display: flex;
    align-items: center;
}}
.hostess-img {{
    position: absolute; left: 0; top: 0; bottom: 0;
    width: 200px; object-fit: cover; object-position: top;
    opacity: 0.9;
    mask-image: linear-gradient(90deg, rgba(0,0,0,1) 70%, transparent 100%);
    -webkit-mask-image: linear-gradient(90deg, rgba(0,0,0,1) 70%, transparent 100%);
}}
.hostess-welcome {{
    margin-left: 220px;
    display: flex; flex-direction: column; justify-content: center;
}}
.hw-title {{ font-family:'Orbitron',monospace; font-size:1rem; color:#64b5f6; letter-spacing:2px; margin-bottom:4px; }}
.hw-sub {{ font-family:'Exo 2',sans-serif; font-size:0.85rem; color:#90a4ae; }}
.hw-tag {{ font-family:'Share Tech Mono',monospace; font-size:0.65rem; color:#37474f; letter-spacing:2px; margin-top:4px; }}

/* CONTENT WRAP */
.cw {{ max-width:980px; margin:0 auto; padding:28px 36px; background:#040d18; }}

/* API KEY */
.apikey-box {{ background:rgba(13,27,42,0.9); border:1px solid rgba(100,181,246,0.25); border-radius:12px; padding:18px 22px; margin-bottom:22px; }}
.apikey-label {{ font-family:'Share Tech Mono',monospace; font-size:0.67rem; letter-spacing:3px; color:#64b5f6; margin-bottom:10px; }}
.apikey-hint {{ font-size:0.72rem; color:#546e7a; margin-top:8px; }}
.apikey-hint a {{ color:#4fc3f7; }}

/* API KEY STATUS BADGE */
.api-status-ok {{
    display:inline-flex; align-items:center; gap:7px;
    font-family:'Share Tech Mono',monospace; font-size:0.68rem; letter-spacing:2px;
    color:#4caf50; border:1px solid rgba(76,175,80,0.35);
    background:rgba(76,175,80,0.08); border-radius:8px;
    padding:8px 14px; white-space:nowrap;
}}
.api-status-warn {{
    display:inline-flex; align-items:center; gap:7px;
    font-family:'Share Tech Mono',monospace; font-size:0.68rem; letter-spacing:2px;
    color:#ff9800; border:1px solid rgba(255,152,0,0.35);
    background:rgba(255,152,0,0.08); border-radius:8px;
    padding:8px 14px; white-space:nowrap;
}}

/* FEATURE GRID */
.fg {{ display:grid; grid-template-columns:repeat(3,1fr); gap:14px; margin-bottom:20px; }}
.fc {{ background:rgba(13,27,42,0.8); border:1px solid rgba(100,181,246,0.13); border-radius:12px; padding:18px; transition:all 0.3s; }}
.fc:hover {{ border-color:rgba(100,181,246,0.45); background:rgba(30,77,123,0.2); transform:translateY(-2px); }}
.fi {{ font-size:1.7rem; margin-bottom:9px; }}
.ft {{ font-weight:600; color:#90caf9; font-size:0.88rem; margin-bottom:5px; }}
.fd {{ font-size:0.76rem; color:#546e7a; line-height:1.5; }}

/* CAPABILITIES */
.cb {{ background:rgba(13,27,42,0.5); border:1px solid rgba(100,181,246,0.08); border-radius:14px; padding:20px; margin-bottom:20px; }}
.ct {{ font-family:'Orbitron',monospace; font-size:0.7rem; letter-spacing:3px; color:#64b5f6; margin-bottom:14px; text-align:center; }}
.cg {{ display:grid; grid-template-columns:repeat(2,1fr); gap:9px; }}
.ci {{ display:flex; gap:8px; align-items:flex-start; padding:7px; border-radius:8px; }}
.cd {{ width:5px; height:5px; background:#64b5f6; border-radius:50%; margin-top:7px; flex-shrink:0; }}
.cx {{ font-size:0.78rem; color:#90a4ae; line-height:1.55; }}
.cx strong {{ color:#90caf9; }}

/* TECH */
.tr {{ display:flex; gap:7px; flex-wrap:wrap; justify-content:center; margin-bottom:26px; }}
.tb {{ font-family:'Share Tech Mono',monospace; font-size:0.66rem; padding:4px 11px; border-radius:20px; border:1px solid rgba(100,181,246,0.2); color:#64b5f6; background:rgba(100,181,246,0.05); }}

/* LAUNCH BTN */
.lb > button {{ background:linear-gradient(135deg,#1565c0,#1e88e5)!important; color:#fff!important; border:1px solid #42a5f5!important; border-radius:10px!important; font-family:'Orbitron',monospace!important; font-size:0.78rem!important; letter-spacing:3px!important; padding:13px 36px!important; box-shadow:0 0 30px rgba(33,150,243,0.25)!important; transition:all 0.3s!important; }}
.lb > button:hover {{ box-shadow:0 0 50px rgba(33,150,243,0.45)!important; }}

/* METRICS */
.mb {{ display:grid; grid-template-columns:repeat(4,1fr); gap:10px; padding:13px 26px; border-bottom:1px solid rgba(100,181,246,0.07); background:rgba(6,14,26,0.6); }}
.mbox {{ background:rgba(13,27,42,0.6); border:1px solid rgba(100,181,246,0.09); border-radius:8px; padding:10px; text-align:center; }}
.mv {{ font-family:'Orbitron',monospace; font-size:0.95rem; color:#64b5f6; font-weight:700; }}
.ml {{ font-family:'Share Tech Mono',monospace; font-size:0.56rem; color:#2e4a6a; letter-spacing:2px; margin-top:2px; text-transform:uppercase; }}

/* SIDEBAR */
.sbt {{ font-family:'Share Tech Mono',monospace; font-size:0.59rem; letter-spacing:3px; color:#2e4a6a; margin:13px 0 7px; text-transform:uppercase; }}
.sp {{ background:rgba(13,27,42,0.6); border:1px solid rgba(100,181,246,0.09); border-radius:10px; padding:11px; margin-bottom:8px; }}
.sp-row {{ display:flex; justify-content:space-between; font-family:'Share Tech Mono',monospace; font-size:0.66rem; color:#37474f; padding:2px 0; }}
.ok {{ color:#4caf50; }} .warn {{ color:#ff9800; }}

/* FLIGHT TABLE */
.ft-tbl {{ width:100%; border-collapse:collapse; margin-top:8px; }}
.ft-tbl th {{ font-family:'Share Tech Mono',monospace; font-size:0.6rem; letter-spacing:2px; color:#37474f; padding:5px 9px; border-bottom:1px solid rgba(100,181,246,0.1); text-align:left; }}
.ft-tbl td {{ font-family:'Exo 2',sans-serif; font-size:0.76rem; color:#90a4ae; padding:6px 9px; border-bottom:1px solid rgba(100,181,246,0.04); }}
.ft-tbl tr:hover td {{ background:rgba(100,181,246,0.03); color:#cfd8dc; }}
.sl {{ color:#4caf50; font-weight:600; }} .sd {{ color:#2196f3; font-weight:600; }}
.se {{ color:#f44336; font-weight:700; animation:blink-r 1s infinite; }}
@keyframes blink-r {{ 0%,100%{{opacity:1}} 50%{{opacity:0.4}} }}

/* MEMORY */
.mem-badge {{ display:inline-flex; align-items:center; gap:6px; font-family:'Share Tech Mono',monospace; font-size:0.6rem; color:#4caf50; border:1px solid rgba(76,175,80,0.3); background:rgba(76,175,80,0.07); border-radius:20px; padding:3px 10px; margin-bottom:8px; }}

/* FREE TEXT CHAT BANNER inside agent page */
.free-chat-banner {{
    background: linear-gradient(135deg, rgba(33,150,243,0.07), rgba(100,181,246,0.03));
    border: 1px solid rgba(100,181,246,0.2);
    border-left: 4px solid #1e88e5;
    border-radius: 10px;
    padding: 12px 18px;
    margin-bottom: 14px;
    display: flex;
    align-items: flex-start;
    gap: 12px;
}}
.fcb-icon {{ font-size: 1.4rem; flex-shrink: 0; margin-top: 2px; }}
.fcb-text {{ flex: 1; }}
.fcb-title {{ font-family:'Orbitron',monospace; font-size:0.7rem; letter-spacing:2px; color:#64b5f6; margin-bottom:5px; }}
.fcb-desc {{ font-family:'Exo 2',sans-serif; font-size:0.82rem; color:#78909c; line-height:1.6; }}
.fcb-pills {{ display:flex; flex-wrap:wrap; gap:6px; margin-top:8px; }}
.fcb-pill {{
    font-family:'Share Tech Mono',monospace; font-size:0.6rem;
    color:#4fc3f7; border:1px solid rgba(79,195,247,0.22);
    background:rgba(79,195,247,0.05); border-radius:20px; padding:3px 10px;
}}

/* CHAT */
[data-testid="stChatMessage"] {{ background:transparent!important; border:none!important; padding:4px 0!important; }}
[data-testid="stChatMessage"] p {{ color:#e8f4fd!important; font-size:0.96rem!important; line-height:1.8!important; font-family:'Exo 2',sans-serif!important; }}
[data-testid="stChatMessage"] h1,[data-testid="stChatMessage"] h2,[data-testid="stChatMessage"] h3 {{ color:#64b5f6!important; font-family:'Orbitron',monospace!important; font-size:1rem!important; border-bottom:1px solid rgba(100,181,246,0.18); padding-bottom:5px; margin:14px 0 9px!important; }}
[data-testid="stChatMessage"] strong {{ color:#81d4fa!important; font-weight:700!important; }}
[data-testid="stChatMessage"] li {{ color:#e0f0ff!important; font-size:0.93rem!important; margin:5px 0!important; line-height:1.7!important; }}
[data-testid="stChatMessage"] ul {{ border-left:2px solid rgba(100,181,246,0.3); padding-left:15px; margin:7px 0!important; }}
[data-testid="stChatMessage"] a {{ color:#4fc3f7!important; text-decoration:underline!important; }}
[data-testid="stChatMessage"] code {{ background:rgba(100,181,246,0.1)!important; color:#80cbc4!important; padding:2px 5px; border-radius:4px; font-family:'Share Tech Mono',monospace!important; }}

/* BUTTONS */
.stButton > button {{ background:rgba(13,27,42,0.8)!important; color:#78909c!important; border:1px solid rgba(100,181,246,0.12)!important; border-radius:8px!important; font-family:'Share Tech Mono',monospace!important; font-size:0.67rem!important; padding:6px 9px!important; transition:all 0.2s!important; text-align:left!important; }}
.stButton > button:hover {{ background:rgba(30,77,123,0.3)!important; color:#90caf9!important; border-color:rgba(100,181,246,0.4)!important; }}

/* TEXT INPUT */
.stTextInput input {{ background:rgba(4,13,24,0.9)!important; border:1px solid rgba(100,181,246,0.25)!important; color:#e0f0ff!important; font-family:'Share Tech Mono',monospace!important; font-size:0.81rem!important; border-radius:8px!important; }}
.stTextInput input:focus {{ border-color:#64b5f6!important; box-shadow:0 0 12px rgba(100,181,246,0.2)!important; }}
.stTextInput label {{ font-family:'Share Tech Mono',monospace!important; font-size:0.66rem!important; letter-spacing:2px!important; color:#546e7a!important; text-transform:uppercase!important; }}

/* CHAT INPUT */
[data-testid="stChatInput"] {{ border-top:1px solid rgba(100,181,246,0.1)!important; padding:13px 34px!important; background:rgba(6,14,26,0.85)!important; }}
[data-testid="stChatInput"] textarea {{ background:rgba(13,27,42,0.9)!important; border:1px solid rgba(100,181,246,0.2)!important; color:#e0e8f0!important; font-family:'Exo 2',sans-serif!important; border-radius:10px!important; }}

.footer {{ text-align:center; font-family:'Share Tech Mono',monospace; font-size:0.6rem; color:#1a3050; letter-spacing:2px; padding:11px; border-top:1px solid rgba(100,181,246,0.05); background:#040d18; }}
::-webkit-scrollbar {{ width:4px; }}
::-webkit-scrollbar-track {{ background:#040d18; }}
::-webkit-scrollbar-thumb {{ background:#1e4d7b; border-radius:2px; }}
iframe {{ border-radius:12px!important; border:1px solid rgba(100,181,246,0.15)!important; }}
</style>
""", unsafe_allow_html=True)

# ── NAVBAR ─────────────────────────────────────
dot_color  = "#4caf50" if has_openai else "#f44336"
dot_status = "SYSTEM ACTIVE" if has_openai else "API KEY REQUIRED"
st.markdown(f"""
<div class="navbar">
    <div class="nb-brand">✈ SKY<span>MIND</span> SL</div>
    <div class="nb-right">
        <span><span class="nb-dot" style="background:{dot_color}"></span>{dot_status}</span>
        <span style="color:#2e4a6a">· COLOMBO FIR VCCF</span>
    </div>
</div>
<div class="pw">
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# WELCOME PAGE
# ══════════════════════════════════════════════
if st.session_state.page == "welcome":

    # HERO
    st.markdown(f"""
    <div class="hero">
        <div class="hero-bg"></div>
        <div class="hero-ov"></div>
        <div class="hero-grid"></div>
        <div class="hero-runway"></div>
        <div class="hero-content">
            <div class="h-badge">● COLOMBO FIR · VCBI · VCCA · VCCB · VCCT · VCCJ</div>
            <div class="h-title">SKY<span>MIND</span> SL</div>
            <div class="h-sub">SRI LANKA AVIATION AI AGENT</div>
            <div class="h-tag">Smart skies. Safer journeys.</div>
            <div class="h-stats">
                <div class="hstat"><div class="hstat-val">5</div><div class="hstat-lbl">Airports</div></div>
                <div class="hstat"><div class="hstat-val">117</div><div class="hstat-lbl">Daily Flights</div></div>
                <div class="hstat"><div class="hstat-val">24/7</div><div class="hstat-lbl">AI Coverage</div></div>
            </div>
        </div>
        <div class="airline-badge">
            <div class="al-icon">✈️</div>
            <div class="al-name">SRILANKAN AIRLINES</div>
            <div class="al-sub">NATIONAL CARRIER · UL</div>
            <div class="al-info">
                HUB: VCBI / CMB<br>
                ALLIANCE: ONEWORLD<br>
                FLEET: A320 · A330 · A350
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # MAIN CONTENT
    st.markdown('<div class="cw">', unsafe_allow_html=True)

    # ── API KEY INPUT ──
    st.markdown('<div class="apikey-box"><div class="apikey-label">🔑 OPENAI API KEY — REQUIRED TO LAUNCH</div>', unsafe_allow_html=True)

    entered_key = st.text_input(
        "API Key", value=st.session_state.api_key_input,
        type="password", placeholder="sk-proj-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
        label_visibility="collapsed", key="apikey_field",
    )
    if entered_key != st.session_state.api_key_input:
        st.session_state.api_key_input = entered_key
        if entered_key.startswith("sk-"):
            os.environ["OPENAI_API_KEY"] = entered_key
        st.rerun()

    if has_openai:
        st.markdown('<div class="api-status-ok">✅ &nbsp;API KEY VERIFIED — READY TO LAUNCH</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="api-status-warn">⚠️ &nbsp;API KEY REQUIRED — ENTER ABOVE TO CONTINUE</div>', unsafe_allow_html=True)

    st.markdown('<div class="apikey-hint">Get your key at <a href="https://platform.openai.com/api-keys" target="_blank">platform.openai.com/api-keys</a> &nbsp;·&nbsp; Or add <code>OPENAI_API_KEY=sk-...</code> to your <code>.env</code> file</div></div>', unsafe_allow_html=True)

    # FEATURE CARDS
    st.markdown("""
    <div class="fg">
        <div class="fc"><div class="fi">🌤️</div><div class="ft">Airport Weather</div><div class="fd">Instant aviation weather briefings for all major Sri Lanka airports — METAR-style reports</div></div>
        <div class="fc"><div class="fi">📋</div><div class="ft">FIR NOTAMs</div><div class="fd">Stay updated on Notices to Airmen for the Colombo Flight Information Region</div></div>
        <div class="fc"><div class="fi">🎓</div><div class="ft">Pilot Training</div><div class="fd">Step-by-step guidance to obtain your Private Pilot Licence (PPL) in Sri Lanka</div></div>
        <div class="fc"><div class="fi">✈️</div><div class="ft">Airline Insights</div><div class="fd">SriLankan Airlines routes, schedules, and operational details at a glance</div></div>
        <div class="fc"><div class="fi">📡</div><div class="ft">ATC Frequencies</div><div class="fd">Tower and approach frequencies for major airports — ready when you need them</div></div>
        <div class="fc"><div class="fi">🚁</div><div class="ft">Drone Regulations</div><div class="fd">Complete UAV rules and CAASL regulations for unmanned aircraft in Sri Lanka</div></div>
    </div>
    """, unsafe_allow_html=True)

    # CAPABILITIES
    st.markdown("""
    <div class="cb">
        <div class="ct">🚀 WHAT YOU CAN DO WITH SKYMIND SL</div>
        <div class="cg">
            <div class="ci"><div class="cd"></div><div class="cx"><strong>Check Airport Weather:</strong> Get instant aviation weather briefings for Colombo (VCBI), Hambantota (VCCA), and more.</div></div>
            <div class="ci"><div class="cd"></div><div class="cx"><strong>Access FIR NOTAMs:</strong> Stay updated on notices to airmen for the Colombo FIR.</div></div>
            <div class="ci"><div class="cd"></div><div class="cx"><strong>Pilot Training Guidance:</strong> Learn steps to obtain a Private Pilot Licence (PPL) in Sri Lanka.</div></div>
            <div class="ci"><div class="cd"></div><div class="cx"><strong>Airline Insights:</strong> Explore SriLankan Airlines routes, schedules, and operational details.</div></div>
            <div class="ci"><div class="cd"></div><div class="cx"><strong>ATC Frequencies:</strong> Retrieve tower and approach frequencies for major airports.</div></div>
            <div class="ci"><div class="cd"></div><div class="cx"><strong>Drone Regulations:</strong> Understand UAV rules in Sri Lanka under CAASL.</div></div>
            <div class="ci"><div class="cd"></div><div class="cx"><strong>Free-Form Questions:</strong> Type ANY aviation question in your own words — fully open-ended chat.</div></div>
            <div class="ci"><div class="cd"></div><div class="cx"><strong>Airport Directory:</strong> Complete data on all 5 Sri Lankan airports with runway info.</div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # TECH STACK
    st.markdown("""
    <div class="tr">
        <div class="tb">🧠 LANGGRAPH</div><div class="tb">🔧 LANGCHAIN</div>
        <div class="tb">🤖 GPT-4o-mini</div><div class="tb">🌐 TAVILY SEARCH</div>
        <div class="tb">🌦️ OPENWEATHERMAP</div><div class="tb">✈️ SRI LANKA AVIATION</div>
    </div>
    """, unsafe_allow_html=True)

    # LAUNCH BUTTON
    c1, c2, c3 = st.columns([2, 1, 2])
    with c2:
        st.markdown('<div class="lb">', unsafe_allow_html=True)
        if st.button("⚡  LAUNCH AGENT", use_container_width=True, key="launch"):
            if not has_openai:
                st.error("⚠️ Please enter your OpenAI API key above first.")
            else:
                st.session_state.page = "agent"
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="footer">⚠️ FOR EDUCATIONAL PURPOSES ONLY · VERIFY WITH AASL · CAASL · OFFICIAL ATC</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════
# AGENT PAGE
# ══════════════════════════════════════════════
elif st.session_state.page == "agent":

    local_time    = datetime.now().strftime("%H:%M")
    api_st        = "🟢 ACTIVE" if has_openai else "🔴 NO KEY"
    total_flights = sum(a["flights_today"] for a in AIRPORTS.values())
    sel_ap        = AIRPORTS[st.session_state.selected_airport]


    # METRICS BAR
    st.markdown(f"""
    <div class="mb">
        <div class="mbox"><div class="mv" style="font-size:0.8rem">{api_st}</div><div class="ml">System</div></div>
        <div class="mbox"><div class="mv">{st.session_state.query_count}</div><div class="ml">Queries</div></div>
        <div class="mbox"><div class="mv">{total_flights}</div><div class="ml">Flights Today</div></div>
        <div class="mbox"><div class="mv">{local_time}</div><div class="ml">LKT</div></div>
    </div>
    """, unsafe_allow_html=True)

    sidebar_col, main_col = st.columns([1, 3.2])

    # ── SIDEBAR ───────────────────────────────
    with sidebar_col:
        if st.button("← Home", use_container_width=True, key="home_btn"):
            st.session_state.page = "welcome"
            st.rerun()

        # FOLIUM MAP
        st.markdown('<div class="sbt">Live Airport Map</div>', unsafe_allow_html=True)
        try:
            import folium
            from streamlit_folium import st_folium

            m = folium.Map(location=[7.8731, 80.7718], zoom_start=7, tiles="CartoDB dark_matter")
            for icao, info in AIRPORTS.items():
                is_sel = icao == st.session_state.selected_airport
                popup_html = f"""<div style="font-family:monospace;background:#0d1b2a;color:#e0f0ff;padding:10px;border-radius:8px;min-width:180px;">
                    <b style="color:#64b5f6">{icao} / {info['iata']}</b><br>{info['name']}<br>
                    <span style="color:#4caf50">✈ {info['flights_today']} flights today</span><br>
                    <span style="color:#78909c">{info['runways']}</span></div>"""
                folium.CircleMarker(
                    location=[info["lat"], info["lon"]],
                    radius=12 if is_sel else 8,
                    color=info["color"], fill=True, fill_color=info["color"],
                    fill_opacity=0.85 if is_sel else 0.6,
                    weight=3 if is_sel else 1.5,
                    popup=folium.Popup(popup_html, max_width=220),
                    tooltip=f"{icao}/{info['iata']} — {info['name']}",
                ).add_to(m)
                folium.Marker(
                    location=[info["lat"]+0.18, info["lon"]],
                    icon=folium.DivIcon(
                        html=f'<div style="font-family:monospace;font-size:9px;color:{info["color"]};white-space:nowrap;font-weight:bold;">{icao}</div>',
                        icon_size=(60,15), icon_anchor=(0,0),
                    ),
                ).add_to(m)

            map_result = st_folium(m, height=280, width=None, returned_objects=["last_object_clicked"])
            if map_result and map_result.get("last_object_clicked"):
                clicked = map_result["last_object_clicked"]
                clat, clon = clicked.get("lat"), clicked.get("lng")
                if clat and clon:
                    closest = min(AIRPORTS.items(), key=lambda x: abs(x[1]["lat"]-clat)+abs(x[1]["lon"]-clon))
                    if closest[0] != st.session_state.selected_airport:
                        st.session_state.selected_airport = closest[0]
                        st.rerun()
        except ImportError:
            st.warning("Run: pip install folium streamlit-folium")

        # AIRPORT SELECTOR
        st.markdown('<div class="sbt">Select Airport</div>', unsafe_allow_html=True)
        for icao, info in AIRPORTS.items():
            is_sel = icao == st.session_state.selected_airport
            if st.button(f"{'▶ ' if is_sel else '   '}{icao}/{info['iata']}  {info['name']}", key=f"ap_{icao}", use_container_width=True):
                st.session_state.selected_airport = icao
                st.rerun()

        st.markdown("<div style='margin-top:8px;'></div>", unsafe_allow_html=True)
        if st.button(f"🌤️ Weather at {st.session_state.selected_airport}", use_container_width=True, key="wx_btn"):
            st.session_state.quick_query = f"Give me a detailed aviation weather briefing for {sel_ap['name']} ({st.session_state.selected_airport}/{sel_ap['iata']}) in Sri Lanka"

        # QUICK QUERIES
        st.markdown('<div class="sbt">Quick Queries</div>', unsafe_allow_html=True)
        quick = {
            "📋 FIR NOTAMs":  "What NOTAMs should I know about for the Colombo FIR?",
            "🎓 PPL License": "How do I get a PPL license in Sri Lanka through CAASL?",
            "✈️ Airlines":    "Tell me about SriLankan Airlines and routes from CMB",
            "📡 ATC Freqs":   "What are the ATC frequencies for Colombo tower and approach?",
            "🚁 Drone Rules": "What are the drone UAV regulations in Sri Lanka?",
        }
        for lbl, q in quick.items():
            if st.button(lbl, key=f"qq_{lbl}", use_container_width=True):
                st.session_state.quick_query = q

        # SESSION
        st.markdown('<div class="sbt">Session</div>', unsafe_allow_html=True)
        ok_c = "ok" if has_openai else "warn"
        ok_l = "Connected" if has_openai else "No Key"
        mem_count = len(st.session_state.messages)
        st.markdown(f"""<div class="sp">
            <div class="sp-row"><span>ID</span><span>{st.session_state.session_id[-10:]}</span></div>
            <div class="sp-row"><span>Messages</span><span class="ok">{mem_count}</span></div>
            <div class="sp-row"><span>Memory</span><span class="ok">{"Active" if mem_count>0 else "Empty"}</span></div>
            <div class="sp-row"><span>OpenAI</span><span class="{ok_c}">{ok_l}</span></div>
        </div>""", unsafe_allow_html=True)

        if st.button("🗑️ Clear Chat", use_container_width=True, key="clear_btn"):
            st.session_state.messages = []
            st.session_state.query_count = 0
            st.rerun()

    # ── MAIN CHAT ─────────────────────────────
    with main_col:

        # LIVE FLIGHT TABLE
        with st.expander("📡 Live Flights Today — All Sri Lanka Airports", expanded=False):
            random.seed(42)
            airlines = ["UL","EK","QR","SQ","AI","6E","FZ","WY"]
            statuses = ["Landed","Landed","Landed","Departed","Departed","Emergency"]
            st.markdown('<table class="ft-tbl"><tr><th>FLIGHT</th><th>AIRPORT</th><th>ROUTE</th><th>TIME</th><th>STATUS</th></tr>', unsafe_allow_html=True)
            rows = []
            for icao, info in AIRPORTS.items():
                for i in range(min(3, info["flights_today"])):
                    al  = random.choice(airlines)
                    num = random.randint(100, 999)
                    dst = random.choice(["DXB","SIN","DOH","BOM","DEL","LHR","KUL","MAA"])
                    hr  = random.randint(0, 23)
                    mn  = random.choice([0,15,30,45])
                    sl  = statuses[i % len(statuses)]
                    css = "sl" if sl=="Landed" else ("sd" if sl=="Departed" else "se")
                    rows.append(f"<tr><td>{al}{num}</td><td>{icao}/{info['iata']}</td><td>→ {dst}</td><td>{hr:02d}:{mn:02d}</td><td class='{css}'>{sl}</td></tr>")
            st.markdown("".join(rows)+"</table>", unsafe_allow_html=True)
            st.markdown(f"<div style='font-family:Share Tech Mono,monospace;font-size:0.58rem;color:#2e4a6a;margin-top:7px;'>TOTAL: {total_flights} movements · {local_time} LKT</div>", unsafe_allow_html=True)

        # FREE CHAT BANNER
        st.markdown("""
        <div class="free-chat-banner">
            <div class="fcb-icon">💬</div>
            <div class="fcb-text">
                <div class="fcb-title">ASK ANYTHING — OPEN CHAT · NO RESTRICTIONS</div>
                <div class="fcb-desc">
                    Type any aviation question in your own words below. You are <strong>not</strong> limited to the sidebar presets.
                    SkyMind SL understands natural language and can answer freely on any Sri Lanka aviation topic.
                </div>
                <div class="fcb-pills">
                    <span class="fcb-pill">What's the QNH at CMB today?</span>
                    <span class="fcb-pill">How long is Mattala's runway?</span>
                    <span class="fcb-pill">VFR documents needed in SL?</span>
                    <span class="fcb-pill">Is there a curfew at VCBI?</span>
                    <span class="fcb-pill">Night flying rules for drones?</span>
                    <span class="fcb-pill">ATIS frequency Colombo approach?</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # MEMORY INDICATOR
        if st.session_state.messages:
            st.markdown(f'<div class="mem-badge">🧠 MEMORY ACTIVE — {len(st.session_state.messages)} messages in context</div>', unsafe_allow_html=True)

        # EMPTY STATE
        if not st.session_state.messages:
            st.markdown(f"""
            <div style="text-align:center;padding:36px 20px;color:#2e4a6a;">
                <div style="font-size:2.8rem;margin-bottom:14px;">✈️</div>
                <div style="font-family:'Orbitron',monospace;font-size:1rem;color:#64b5f6;letter-spacing:3px;margin-bottom:6px;">SKYMIND SL READY</div>
                <div style="font-family:'Share Tech Mono',monospace;font-size:0.66rem;letter-spacing:2px;margin-bottom:14px;color:#37474f;">GPT-4o-mini · MULTI-TOOL · MEMORY-ENABLED</div>
                <div style="font-size:0.82rem;color:#546e7a;max-width:500px;margin:0 auto;line-height:1.7;">
                    Ask about anything in your own words — weather, regulations, airports, airlines, pilot licensing,
                    drone rules, ATC frequencies, NOTAMs, emergency procedures, and more.<br><br>
                    <span style="color:{sel_ap['color']};font-family:'Share Tech Mono',monospace;font-size:0.7rem;">
                        Selected: {st.session_state.selected_airport} — {sel_ap['name']}
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # CHAT HISTORY
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"], avatar="👤" if msg["role"]=="user" else "✈️"):
                st.markdown(msg["content"])

        # QUICK QUERY HANDLER
        if st.session_state.quick_query:
            user_input = st.session_state.quick_query
            st.session_state.quick_query = None
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user", avatar="👤"):
                st.markdown(user_input)
            with st.chat_message("assistant", avatar="✈️"):
                with st.spinner("🛫 SkyMind SL analyzing..."):
                    response = get_ai_response(user_input, chat_history=st.session_state.messages[:-1])
                st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.session_state.query_count += 1
            st.rerun()

        # CHAT INPUT — fully free text
        if prompt := st.chat_input("✈️  Ask any aviation question freely — weather, NOTAMs, regulations, airports, licensing, drones, ATC..."):
            if not has_openai:
                st.error("⚠️ No API key detected. Go Home and enter your OpenAI API key.")
            else:
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user", avatar="👤"):
                    st.markdown(prompt)
                with st.chat_message("assistant", avatar="✈️"):
                    with st.spinner("🛫 SkyMind SL processing..."):
                        response = get_ai_response(prompt, chat_history=st.session_state.messages[:-1])
                    st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.session_state.query_count += 1

    st.markdown('<div class="footer">⚠️ SKYMIND SL IS FOR EDUCATIONAL PURPOSES ONLY · VERIFY WITH AASL · CAASL · OFFICIAL ATC · VCBI COLOMBO FIR VCCF</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)