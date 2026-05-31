# config.py

CHAINS_DEFAULT = ["solana", "bsc"]
TG_CALLBACK_VALID_CHAINS = CHAINS_DEFAULT


# -------------------------
# HARD FILTERS (SAFE)
# -------------------------
MIN_LIQ_USD = 8_000
MIN_TXNS_M5 = 4
MAX_ABS_PRICECHANGE_M5 = 25.0
MAX_PRICECHANGE_H1 = 80.0
MIN_VOLUME_M5 = 1_500

# Signal ratios
MIN_BUY_SELL_RATIO = 1.15

# -------------------------
# SCORING WEIGHTS
# -------------------------
W_TXNS_IMBALANCE = 1.0
W_VOLUME_M5 = 1.0
W_LIQ = 0.7
W_PCHG_M5_PENALTY = 1.2
SCORE_VOL_SCALE = 10_000.0
SCORE_LIQ_SCALE = 50_000.0
SCORE_LIQ_BELOW_MIN_PENALTY = 5.0
SCORE_MIN_VALUE = 0.0

TOP_N = 50

# -------------------------
# WATCHLIST / ALERTS
# -------------------------
WATCH_TTL_MINUTES = 60          # тримати сигнал у watchlist стільки хвилин
WATCH_MAX_ITEMS = 200           # ліміт, щоб не роздувалося
ALERT_MIN_SCORE = 2.0           # мінімальний score, щоб попасти в alerts (підкрутиш під себе)
ALERT_COOLDOWN_SECONDS = 900    # щоб один і той самий pair не “спамив” щохвилини

# -------------------------
# POSITION HEALTH THRESHOLDS
# -------------------------
HEALTH_THRESHOLDS = {
    "liq_low_usd": 1_000.0,
    "vol24_low_usd": 100.0,
    "stale_minutes": 240.0,
    "min_history_points": 4,
    "flat_eps": {
        "price": 0.003,
        "liq": 0.01,
        "vol24": 0.05,
        "vol5": 0.05,
    },
}

# -------------------------
# LEGACY MONITORING APP SCORING
# -------------------------
MONITORING_SCORE_LIQ_SCALE = 2_000.0
MONITORING_SCORE_LIQ_CAP = 200.0
MONITORING_SCORE_VOL24_SCALE = 8_000.0
MONITORING_SCORE_VOL24_CAP = 200.0
MONITORING_SCORE_VOL5_SCALE = 1_500.0
MONITORING_SCORE_VOL5_CAP = 200.0
MONITORING_SCORE_PC1H_ABS_CAP = 30.0
MONITORING_SCORE_PC1H_WEIGHT = 2.0
MONITORING_SCORE_PC5_ABS_CAP = 20.0
MONITORING_SCORE_PC5_WEIGHT = 1.5
MONITORING_DROP_MIN_LIQ_USD = 3_000.0
MONITORING_DROP_MIN_VOL24_USD = 4_000.0
MONITORING_DROP_MIN_VOL5_USD = 150.0
MONITORING_DROP_MAX_PC1H_NEG = -12.0
