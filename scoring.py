# scoring.py
from config import (
    MIN_LIQ_USD, MIN_TXNS_M5, MAX_ABS_PRICECHANGE_M5, MAX_PRICECHANGE_H1, MIN_VOLUME_M5,
    MIN_BUY_SELL_RATIO,
    W_TXNS_IMBALANCE, W_VOLUME_M5, W_LIQ, W_PCHG_M5_PENALTY
)

def passes_safe_filters(row: dict) -> bool:
    liq = row.get("liqUsd") or 0
    txns_m5 = row.get("txns_m5") or 0
    pc_m5 = row.get("pc_m5")
    pc_h1 = row.get("pc_h1")
    vol_m5 = row.get("vol_m5") or 0

    buys = row.get("buys_m5") or 0
    sells = row.get("sells_m5") or 0
    ratio = (buys / max(1, sells))

    if liq < MIN_LIQ_USD:
        return False
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
    buys = row.get("buys_m5") or 0
    sells = row.get("sells_m5") or 0
    imbalance = (buys - sells)

    vol_m5 = row.get("vol_m5") or 0
    liq = row.get("liqUsd") or 0
    pc_m5 = row.get("pc_m5") or 0

    score = 0.0
    score += W_TXNS_IMBALANCE * imbalance
    score += W_VOLUME_M5 * (vol_m5 / 10_000)
    score += W_LIQ * (liq / 50_000)
    score -= W_PCHG_M5_PENALTY * abs(pc_m5)

    return float(score)

