"""Global CSS for the Streamlit fire-analysis dashboard."""

DASHBOARD_CSS = """
<style>
  div[data-testid="stVerticalBlock"] > div:has(> div > div > div[data-testid="stMarkdown"] h1.fire-main-title) {
    padding-top: 0.25rem;
  }
  .fire-card {
    border-radius: 12px;
    padding: 1rem 1.25rem;
    border: 1px solid rgba(255,255,255,0.08);
    background: linear-gradient(135deg, rgba(30,35,45,0.95), rgba(20,24,32,0.98));
    box-shadow: 0 8px 32px rgba(0,0,0,0.35);
    margin-bottom: 0.75rem;
  }
  .fire-prob-massive {
    font-size: clamp(2rem, 4vw, 3.2rem);
    font-weight: 800;
    letter-spacing: -0.03em;
    line-height: 1.1;
    margin: 0.15rem 0 0.35rem 0;
  }
  .fire-subtle {
    opacity: 0.72;
    font-size: 0.92rem;
  }
  .fire-badge {
    display: inline-block;
    padding: 0.35rem 0.75rem;
    border-radius: 999px;
    font-weight: 600;
    font-size: 0.88rem;
    border: 1px solid rgba(255,255,255,0.15);
  }
  .fire-badge-ok { background: rgba(46,204,113,0.2); color: #a8f0c4; }
  .fire-badge-warn { background: rgba(241,196,15,0.22); color: #ffeaa7; }
  .fire-badge-danger { background: rgba(231,76,60,0.25); color: #ffb4a8; }
  .fire-frame-thumb {
    border-radius: 8px;
    overflow: hidden;
    border: 2px solid rgba(255,255,255,0.12);
  }
  .fire-frame-thumb-active {
    border-color: rgba(52,152,219,0.85);
    box-shadow: 0 0 0 2px rgba(52,152,219,0.35);
  }
  h1.fire-main-title {
    font-weight: 800;
    letter-spacing: -0.02em;
    margin-bottom: 0.25rem;
  }
  .fire-hero {
    padding: 1rem 0 0.5rem 0;
  }
</style>
"""


def inject_global_styles() -> None:
    import streamlit as st

    st.markdown(DASHBOARD_CSS, unsafe_allow_html=True)
