"""Streamlit UI for the LangGraph email agent."""

import os

import streamlit as st
from dotenv import load_dotenv

from agent import EmailAgent

load_dotenv()

st.set_page_config(page_title="Email Agent", page_icon="📬", layout="wide")
st.title("📬 Email Agent")
st.caption("LangGraph agent: fetch inbox → extract insights → build your brief.")

with st.sidebar:
    st.header("Connection")
    address = st.text_input("Email", value=os.getenv("EMAIL_ADDRESS", ""))
    password = st.text_input(
        "Password / App password",
        type="password",
        value=os.getenv("EMAIL_PASSWORD", ""),
    )
    imap_host = st.text_input("IMAP host", value=os.getenv("IMAP_HOST", "imap.gmail.com"))
    imap_port = st.number_input("IMAP port", value=int(os.getenv("IMAP_PORT", "993")))
    gemini_key = st.text_input(
        "Gemini API key",
        type="password",
        value=os.getenv("GEMINI_API_KEY", ""),
    )
    model = st.text_input("Model", value=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"))

    st.header("Fetch")
    limit = st.slider("Max emails", 1, 30, 10)
    unread_only = st.checkbox("Unread only", value=True)
    mailbox = st.text_input("Mailbox", value="INBOX")
    run = st.button("Analyze inbox", type="primary", use_container_width=True)

st.markdown(
    """
**Gmail:** enable IMAP and use an [App Password](https://myaccount.google.com/apppasswords).  
**Graph steps:** `fetch_inbox` → `extract_insights` → `build_brief`
"""
)

if run:
    if not all([address, password, gemini_key]):
        st.error("Email, password, and Gemini API key are required.")
        st.stop()
    try:
        with st.spinner("Running LangGraph agent…"):
            agent = EmailAgent(
                address=address,
                password=password,
                imap_host=imap_host,
                imap_port=int(imap_port),
                gemini_api_key=gemini_key,
                model=model,
            )
            emails, summary, insights = agent.run(
                limit=limit, unread_only=unread_only, mailbox=mailbox
            )
    except Exception as e:
        st.error(f"Failed: {e}")
        st.stop()

    st.subheader("Inbox brief")
    st.markdown(summary)

    if insights:
        with st.expander("Per-email insights (graph step 2)"):
            st.markdown(insights)

    with st.expander(f"Raw emails ({len(emails)})"):
        for em in emails:
            st.markdown(f"**{em.subject}** — {em.sender} ({em.date})")
            st.text(em.body[:2000] + ("…" if len(em.body) > 2000 else ""))
            st.divider()
