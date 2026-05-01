import streamlit as st
import time
from llmsearchindex import LLMIndex

st.set_page_config(page_title="LLMSearchIndex!", page_icon="🔍", layout="centered")

st.markdown("""
<style>
#MainMenu, footer {visibility: hidden;}

/* ===== CENTER CONTAINER ===== */
.main {
    max-width: 720px;
    margin: 0 auto;
}

/* ===== SEARCH BOX ===== */
div[data-baseweb="input"] {
    border-radius: 999px !important;
    height: 52px !important;
    padding: 0 8px !important;
}

.stTextInput input {
    font-size: 16px !important;
    padding: 0 16px !important;
    border-radius: 999px !important;
}

/* ===== BRAND ===== */
.brand {
    text-align: center;
    font-size: 56px;
    font-weight: 600;
    margin-bottom: 6px;
    letter-spacing: -1px;
}

.brand-llm { color: #4285F4; }
.brand-search { color: #EA4335; }
.brand-index { color: #34A853; }
.brand-excl { color: #FBBC05; }

/* ===== BADGE ===== */
.local-badge {
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 8px;
    font-size: 13px;
    opacity: 0.6;
    margin-bottom: 30px;
}

/* ===== RESULT CARD ===== */
.result-card {
    padding: 14px 14px;
    border-radius: 12px;
    margin-bottom: 14px;
    transition: background 0.2s ease;
}

.result-card:hover {
    background: rgba(255,255,255,0.04);
}

.result-url {
    font-size: 12px;
    opacity: 0.6;
    margin-bottom: 4px;
    word-break: break-all;
}

.result-title {
    font-size: 18px;
    font-weight: 500;
    color: #8ab4f8;
    text-decoration: none;
    display: inline-block;
    margin-bottom: 6px;
}

.result-title:hover {
    text-decoration: underline;
}

.result-snippet {
    font-size: 14px;
    line-height: 1.4;
    opacity: 0.85;
}
</style>
""", unsafe_allow_html=True)

# ===== HEADER =====
st.markdown("""
<div class="brand">
  <span class="brand-llm">LLM</span>
  <span class="brand-search">Search</span>
  <span class="brand-index">Index</span>
  <span class="brand-excl">!</span>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="local-badge">
    🔒 Private, Local Search Index • 203M+ Webpages •
    <a href="https://github.com/zakerytclarke/llmsearchindex" target="_blank" style="color:inherit;">GitHub</a>
</div>
""", unsafe_allow_html=True)


@st.cache_resource
def load_index():
    return LLMIndex()

index = load_index()

# Updated to collapse the label element completely
query = st.text_input("Search Query", placeholder="Who invented sliced bread?", label_visibility="collapsed")

# ===== RESULTS =====
if query:
    with st.spinner("Scanning 203,169,792 pages..."):
        results = index.search(query, top_k=3, rerank=True)

    st.markdown("---")

    if not results:
        st.info("No results found.")
    else:
        for res in results:
            url = res.get("url", "local://db")
            text = res.get("text", "")
            # Split once to avoid redundant processing
            words = text.split() if text else []
            
            # Capture up to the first 20 words; defaults to "Untitled Result" if empty
            title = " ".join(words[:20]) if words else "Untitled Result"
            
            # Capture words from index 20 to 60 (or whatever is available)
            # The snippet will naturally be an empty string if len(words) <= 20
            snippet = " ".join(words[20:60]) if len(words) > 20 else ""
            st.markdown(f"""
            <div class="result-card">
                <div class="result-url">{url}</div>
                <a class="result-title" href="{url}" target="_blank">
                    {title}
                </a>
                <div class="result-snippet">{snippet}</div>
            </div>
            """, unsafe_allow_html=True)