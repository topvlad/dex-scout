# ============================================================
# LEGACY FILE — not imported by any active codepath as of v0.5.6
# Original role: passes_safe_filters() + score_row() primitives.
# Scoring logic now lives entirely inside app.py (entry engine v1).
# Retained for reference. Safe to delete after verification.
# ============================================================

# scoring.py
from config import (
    MIN_LIQ_USD, MIN_TXNS_M5, MAX_ABS_PRICECHANGE_M5, MAX_PRICECHANGE_H1, MIN_VOLUME_M5,
    MIN_BUY_SELL_RATIO,
    W_TXNS_IMBALANCE, W_VOLUME_M5, W_LIQ, W_PCHG_M5_PENALTY,
    SCORE_VOL_SCALE, SCORE_LIQ_SCALE, SCORE_LIQ_BELOW_MIN_PENALTY, SCORE_MIN_VALUE,
)


def _safe_float(value, default=None):
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value, default=None):
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def passes_safe_filters(row: dict) -> bool:
    liq_raw = row.get("liqUsd")
    txns_raw = row.get("txns_m5")
    vol_raw = row.get("vol_m5")
    liq = _safe_float(liq_raw, 0.0 if liq_raw is None else None)
    txns_m5 = _safe_int(txns_raw, 0 if txns_raw is None else None)
    pc_m5 = _safe_float(row.get("pc_m5"), None)
    pc_h1 = _safe_float(row.get("pc_h1"), None)
    vol_m5 = _safe_float(vol_raw, 0.0 if vol_raw is None else None)
    try:
        buys = int(row.get("buys_m5") or 0)
        sells = int(row.get("sells_m5") or 0)
    except (TypeError, ValueError):
        return False

    if None in (liq, txns_m5, pc_m5, vol_m5):
        return False

    ratio = buys / max(1, sells)

    if txns_m5 < MIN_TXNS_M5:
        return False
    if vol_m5 < MIN_VOLUME_M5:
        return False
    if pc_m5 is None or abs(pc_m5) > MAX_ABS_PRICECHANGE_M5:
        return False
    if pc_h1 is not None and pc_h1 > MAX_PRICECHANGE_H1:
        return False
    if ratio < MIN_BUY_SELL_RATIO:
        return False

    return True


def score_row(row: dict) -> float:
    buys = _safe_int(row.get("buys_m5"), 0) or 0
    sells = _safe_int(row.get("sells_m5"), 0) or 0
    imbalance = buys - sells

    vol_m5 = _safe_float(row.get("vol_m5"), 0.0) or 0.0
    liq = _safe_float(row.get("liqUsd"), 0.0) or 0.0
    pc_m5 = _safe_float(row.get("pc_m5"), 0.0) or 0.0

    score = 0.0
    score += W_TXNS_IMBALANCE * imbalance
    score += W_VOLUME_M5 * (vol_m5 / SCORE_VOL_SCALE)
    score += W_LIQ * (liq / SCORE_LIQ_SCALE)
    score -= W_PCHG_M5_PENALTY * abs(pc_m5)

    if liq < MIN_LIQ_USD:
        score -= SCORE_LIQ_BELOW_MIN_PENALTY

    return float(max(SCORE_MIN_VALUE, score))
