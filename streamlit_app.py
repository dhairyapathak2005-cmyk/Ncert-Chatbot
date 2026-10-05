"""Streamlit interface for the NCERT Class 10 Science helper."""
from __future__ import annotations

import os
from typing import Any

import streamlit as st

st.set_page_config(page_title="NCERT Science Helper", page_icon=None, layout="centered")

# Streamlit secrets must reach app.config before the graph is imported and built.
try:
    nvidia_api_key = st.secrets.get("NVIDIA_API_KEY")
except FileNotFoundError:
    nvidia_api_key = None
if nvidia_api_key:
    os.environ["NVIDIA_API_KEY"] = str(nvidia_api_key)

from app.ui_client import UIClientError, ask, cache_stats


EXAMPLES = (
    "What is photosynthesis?",
    "Why does the sky appear blue?",
    "State Ohm's law",
    "What are acids and bases?",
)
MAX_MESSAGES = 20


def _init_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("is_asking", False)
    st.session_state.setdefault("next_error_id", 0)


def _trim_history() -> None:
    st.session_state.messages = st.session_state.messages[-MAX_MESSAGES:]


def _latency_text(latency_ms: Any) -> str:
    try:
        latency = float(latency_ms)
    except (TypeError, ValueError):
        return ""
    if latency >= 1000:
        return f"{latency / 1000:.1f} s"
    return f"{latency:.0f} ms"


def _cache_badge(cache: dict[str, Any], latency_ms: Any) -> str:
    latency = _latency_text(latency_ms)
    if cache.get("hit"):
        cache_type = cache.get("type", "semantic")
        similarity = cache.get("similarity")
        detail = (
            f"{cache_type}, {float(similarity):.2f}"
            if cache_type == "semantic" and similarity is not None
            else cache_type
        )
        return f"⚡ From cache ({detail}) · {latency}"
    return f"✨ Fresh answer · {latency}"


def _render_message(message: dict[str, Any], index: int) -> str | None:
    with st.chat_message(message["role"]):
        if message.get("error"):
            st.markdown("Something went wrong. Please try again in a moment.")
            retry_id = message.get("id", index)
            if st.button("Retry", key=f"retry-{retry_id}"):
                return str(message["question"])
            return None

        st.markdown(message["content"])
        if message["role"] == "assistant":
            st.caption(_cache_badge(message["cache"], message["latency_ms"]))
            sources = message.get("sources", [])
            if sources:
                with st.expander("Sources"):
                    for source in sources:
                        chapter = source.get("chapter", "NCERT Science")
                        page = source.get("page", "—")
                        snippet = source.get("snippet", "")
                        st.markdown(f"**{chapter}** · Page {page}")
                        if snippet:
                            st.caption(snippet)
    return None


def _answer(question: str, show_question: bool) -> None:
    st.session_state.is_asking = True
    try:
        if show_question:
            with st.chat_message("user"):
                st.markdown(question)
            st.session_state.messages.append({"role": "user", "content": question})

        with st.chat_message("assistant"):
            try:
                with st.spinner("Looking through your textbook..."):
                    response = ask(question)
                st.markdown(response["answer"])
                st.caption(_cache_badge(response["cache"], response["latency_ms"]))
                sources = response.get("sources", [])
                if sources:
                    with st.expander("Sources"):
                        for source in sources:
                            st.markdown(f"**{source['chapter']}** · Page {source['page']}")
                            st.caption(source.get("snippet", ""))
            except (UIClientError, KeyError, TypeError):
                error_id = st.session_state.next_error_id
                st.session_state.next_error_id += 1
                st.markdown("Something went wrong. Please try again in a moment.")
                st.button("Retry", key=f"retry-{error_id}")
                st.session_state.messages.append(
                    {"role": "assistant", "error": True, "question": question, "id": error_id}
                )
            else:
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": response["answer"],
                        "cache": response["cache"],
                        "latency_ms": response["latency_ms"],
                        "sources": response.get("sources", []),
                    }
                )
    finally:
        st.session_state.is_asking = False
        _trim_history()


def _footer() -> None:
    left, right = st.columns((1, 3))
    with left:
        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    with right:
        with st.expander("Cache stats"):
            try:
                stats = cache_stats()
                st.caption(f"{stats['hits']} hits · {stats['misses']} misses · {stats['size']} saved answers")
            except UIClientError:
                st.caption("Cache statistics are unavailable right now.")


def main() -> None:
    _init_state()
    st.markdown(
        """<style>
        #MainMenu, footer, header {visibility: hidden;}
        .block-container {max-width: 760px; padding-top: 3.5rem; padding-bottom: 2rem;}
        h1 {font-size: 1.55rem !important; margin-bottom: .15rem !important;}
        div[data-testid="stChatMessage"] {padding: .35rem 0 .85rem;}
        div[data-testid="stCaptionContainer"] {color: #64707d;}
        div.stButton > button {border-radius: .55rem; white-space: normal;}
        div[data-testid="stExpander"] {border-color: #e3e8eb;}
        @media (max-width: 600px) {
          .block-container {padding: 2rem 1rem 1.25rem;}
        }
        </style>""",
        unsafe_allow_html=True,
    )
    st.title("NCERT Science Helper")
    st.caption("Class 10 Science. Ask any doubt from your textbook.")

    retry_question = None
    retry_index = None
    for index, message in enumerate(st.session_state.messages):
        retry = _render_message(message, index)
        if retry:
            retry_question = retry
            retry_index = index
    if retry_index is not None:
        del st.session_state.messages[retry_index]

    selected_example = None
    if not st.session_state.messages:
        st.write("")
        cols = st.columns(2)
        for index, example in enumerate(EXAMPLES):
            if cols[index % 2].button(example, key=f"example-{index}", use_container_width=True):
                selected_example = example

    question = st.chat_input(
        "Ask a science doubt...", max_chars=500, disabled=st.session_state.is_asking
    )
    submitted = (question or selected_example or retry_question or "").strip()
    if submitted:
        _answer(submitted, show_question=retry_question is None)

    _footer()


if __name__ == "__main__":
    main()
