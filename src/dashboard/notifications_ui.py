"""Notification draft UI for CostSense pages.

Prepares email-style drafts and supports optional **manual** SMTP send on
button click. No automatic delivery.
"""
from __future__ import annotations

from dataclasses import dataclass

import streamlit as st

from src.dashboard.costsense_theme import callout, pill, section
from src.dashboard.notification_delivery import (
    draft_fingerprint,
    notify_recipient,
    send_notification_email,
    smtp_configured,
)


@dataclass(frozen=True)
class NotificationDraft:
    title: str
    severity: str  # Low | Medium | High | Critical
    reason: str
    recipient: str
    subject: str
    body: str
    source_page: str
    source_type: str


def _open_key(state_key: str) -> str:
    return f"cs_notif_open::{state_key}"


def _prepared_key(state_key: str) -> str:
    return f"cs_notif_prepared::{state_key}"


def _sent_fp_key(state_key: str) -> str:
    return f"cs_notif_sent_fp::{state_key}"


def _status_key(state_key: str) -> str:
    return f"cs_notif_status::{state_key}"


def _plaintext(recipient: str, draft: NotificationDraft) -> str:
    return (
        f"To: {recipient}\n"
        f"Subject: {draft.subject}\n\n"
        f"{draft.body}\n"
    )


def render_notification_button(
    *,
    button_label: str,
    draft: NotificationDraft,
    state_key: str,
    visible: bool = True,
) -> None:
    """Render a notification trigger and optional draft panel."""
    if not visible:
        return

    btn_key = f"cs_notif_btn::{state_key}"
    if st.button(button_label, key=btn_key, type="secondary"):
        st.session_state[_open_key(state_key)] = True
        st.session_state[_prepared_key(state_key)] = True
        st.session_state.pop(_status_key(state_key), None)

    if not st.session_state.get(_open_key(state_key)):
        return

    recipient = notify_recipient(draft.recipient)
    fingerprint = draft_fingerprint(
        recipient=recipient,
        subject=draft.subject,
        body=draft.body,
        source_type=draft.source_type,
    )
    already_sent = st.session_state.get(_sent_fp_key(state_key)) == fingerprint

    smtp_ready, smtp_msg = smtp_configured()
    status = st.session_state.get(_status_key(state_key))

    with st.container(border=True):
        section(
            draft.title,
            "Review the draft below, then copy manually or click **Send email**. "
            "CostSense never sends automatically.",
            kicker="Notification draft",
        )
        st.markdown(pill(draft.severity), unsafe_allow_html=True)
        st.markdown(f"**Why:** {draft.reason}")
        st.caption(
            f"Source: **{draft.source_page}** · {draft.source_type}"
        )

        if status:
            tone, message = status
            callout(message, tone=tone)

        if already_sent and not (
            status and status[0] == "success"
        ):
            callout(
                "This exact draft was already sent in this session. "
                "Change the underlying data or dismiss and reopen to send again.",
                tone="info",
            )

        if not smtp_ready:
            callout(smtp_msg, tone="warning")

        c1, c2 = st.columns(2, gap="medium")
        with c1:
            st.text_input("Recipient", value=recipient, disabled=True)
        with c2:
            st.text_input("Subject", value=draft.subject, disabled=True)
        st.text_area(
            "Message",
            value=draft.body,
            height=160,
            disabled=True,
            label_visibility="visible",
        )
        st.markdown("**Copy-friendly draft**")
        st.code(_plaintext(recipient, draft), language=None)

        if st.session_state.get(_prepared_key(state_key)):
            st.caption("Draft prepared — ready to copy or send.")

        btn_send, btn_dismiss, _spacer = st.columns([1, 1, 2])
        with btn_send:
            send_disabled = already_sent or not smtp_ready
            if st.button(
                "Send email",
                key=f"cs_notif_send::{state_key}",
                type="primary",
                disabled=send_disabled,
            ):
                if already_sent:
                    st.session_state[_status_key(state_key)] = (
                        "info",
                        "This draft was already sent.",
                    )
                else:
                    ok, message = send_notification_email(
                        recipient=recipient,
                        subject=draft.subject,
                        body=draft.body,
                    )
                    if ok:
                        st.session_state[_sent_fp_key(state_key)] = fingerprint
                        st.session_state[_status_key(state_key)] = (
                            "success",
                            message,
                        )
                    else:
                        st.session_state[_status_key(state_key)] = (
                            "error",
                            message,
                        )
                st.rerun()
        with btn_dismiss:
            if st.button("Dismiss", key=f"cs_notif_dismiss::{state_key}"):
                st.session_state[_open_key(state_key)] = False
                st.session_state.pop(_status_key(state_key), None)
                st.rerun()
