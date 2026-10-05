"""
Alert system for FII sell-pressure and other regime triggers.

Sends alerts by email (SMTP). Configure via environment variables or
config/settings.yaml.
"""
from __future__ import annotations

import os
import smtplib
import time
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Dict, List, Optional, Set

import logging
logger = logging.getLogger(__name__)  # will be patched if missing


# ---------- Config helpers ----------
def _env(key: str, default: str = "") -> str:
    return os.getenv(key, default).strip()


# ---------- Email ----------
def send_email(
    subject: str,
    body: str,
    to_addrs: Optional[List[str]] = None,
) -> bool:
    """
    Send email via SMTP. Works with Gmail (smtp.gmail.com, port 587) using a
    Google App Password - regular Gmail passwords are rejected by Google for
    SMTP login.
    Required env:
      SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS
      ALERT_EMAIL_TO  (comma-separated)
    Optional: SMTP_FROM
    """
    host = _env("SMTP_HOST")
    port = int(_env("SMTP_PORT", "587") or "587")
    user = _env("SMTP_USER")
    password = _env("SMTP_PASS")
    from_addr = _env("SMTP_FROM") or user
    to_raw = _env("ALERT_EMAIL_TO")
    recipients = to_addrs or [x.strip() for x in to_raw.split(",") if x.strip()]

    if not all([host, user, password, recipients]):
        logger.warning("Email not configured (SMTP_* / ALERT_EMAIL_TO)")
        return False

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = from_addr
    msg["To"] = ", ".join(recipients)
    msg.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP(host, port, timeout=20) as server:
            server.starttls()
            server.login(user, password)
            server.sendmail(from_addr, recipients, msg.as_string())
        return True
    except Exception as e:
        logger.error(f"Email send failed: {e}")
        return False


# ---------- Alert Manager ----------
# Regimes that should fire alerts
DEFAULT_TRIGGER_REGIMES = {
    "FII_SELL_PRESSURE",
    "DII_SUPPORTED",  # optional – still useful context
}


class AlertManager:
    """
    Deduplicated, cooldown-aware alert dispatcher.
    Call `check_and_alert(analysis)` after each full_fii_nifty_analysis().
    """

    def __init__(
        self,
        trigger_regimes: Optional[Set[str]] = None,
        cooldown_minutes: int = 240,  # 4 hours default – avoid spam on same regime
        channels: Optional[List[str]] = None,  # ["email"]
    ):
        self.trigger_regimes = trigger_regimes or DEFAULT_TRIGGER_REGIMES
        self.cooldown_sec = cooldown_minutes * 60
        self.channels = channels or ["email"]
        self._last_sent: Dict[str, float] = {}  # regime -> timestamp

    def _cooldown_ok(self, key: str) -> bool:
        last = self._last_sent.get(key, 0)
        return (time.time() - last) >= self.cooldown_sec

    def _mark_sent(self, key: str) -> None:
        self._last_sent[key] = time.time()

    def format_message(self, analysis: Dict[str, Any]) -> str:
        regime = analysis.get("regime", {})
        latest = analysis.get("latest", {})
        streak = analysis.get("streak", {})
        fo = analysis.get("fo", {})
        abs_ = analysis.get("dii_absorption", {})

        lines = [
            "FII x Nifty Alert",
            f"Regime: {regime.get('regime')} ({regime.get('confidence', 0):.0%})",
            f"Reason: {regime.get('reason', '-')}",
            "",
            f"Date: {latest.get('date')}",
            f"Nifty: {latest.get('nifty_close')} ({latest.get('nifty_ret_pct') or 0:+.2f}%)",
            f"FII Net: Rs {latest.get('fii_net_cr')} Cr",
            f"DII Net: Rs {latest.get('dii_net_cr')} Cr",
            f"FII Streak: {streak.get('direction')} x {streak.get('streak')}d (cum Rs {streak.get('cum_flow_cr')} Cr)",
            "",
            f"F&O LS ratio: {fo.get('fii_ls_ratio')} | bias: {fo.get('fo_bias')}",
            f"FII Idx Fut Net: {fo.get('fii_idx_fut_net')}",
            f"DII absorption (sell days): {abs_.get('avg_absorption_pct')}%",
            "",
            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M IST')}",
        ]
        return "\n".join(lines)

    def check_and_alert(self, analysis: Dict[str, Any], force: bool = False) -> Dict[str, bool]:
        """
        Returns dict of channel -> success.
        Only fires when regime is in trigger set and cooldown allows.
        """
        if not analysis.get("ok"):
            return {}
        regime_name = analysis.get("regime", {}).get("regime", "NEUTRAL")
        if regime_name not in self.trigger_regimes and not force:
            return {}

        key = regime_name
        if not force and not self._cooldown_ok(key):
            logger.info(f"Alert suppressed (cooldown): {key}")
            return {}

        msg = self.format_message(analysis)

        results = {}
        if "email" in self.channels:
            subject = f"[FII Alert] {regime_name} - Nifty {analysis.get('latest', {}).get('nifty_close')}"
            results["email"] = send_email(subject, msg)

        if any(results.values()):
            self._mark_sent(key)
            logger.info(f"Alert sent for {key}: {results}")
        return results
