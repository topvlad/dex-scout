# config.py

CHAINS_DEFAULT = ["solana", "bsc"]

# -------------------------
# HARD FILTERS (SAFE)
# -------------------------
MIN_LIQ_USD = 20_000
MIN_TXNS_M5 = 20
MAX_ABS_PRICECHANGE_M5 = 8.0
MAX_PRICECHANGE_H1 = 40.0
MIN_VOLUME_M5 = 15_000

# Signal ratios
MIN_BUY_SELL_RATIO = 1.7

# -------------------------
# SCORING WEIGHTS
# -------------------------
W_TXNS_IMBALANCE = 1.0
W_VOLUME_M5 = 1.0
W_LIQ = 0.7
W_PCHG_M5_PENALTY = 1.2

TOP_N = 15

# -------------------------
# WATCHLIST / ALERTS
# -------------------------
WATCH_TTL_MINUTES = 15          # тримати сигнал у watchlist стільки хвилин
WATCH_MAX_ITEMS = 50            # ліміт, щоб не роздувалося
ALERT_MIN_SCORE = 3.0           # мінімальний score, щоб попасти в alerts (підкрутиш під себе)
ALERT_COOLDOWN_SECONDS = 120    # щоб один і той самий pair не “спамив” щохвилини

