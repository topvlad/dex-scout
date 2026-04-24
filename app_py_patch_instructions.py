# app.py — PATCH INSTRUCTIONS
# ============================================================
# app.py is 13 000+ lines so this file contains EXACT diff-style
# instructions rather than a full copy. Apply each change in order.
# All line numbers are approximate — search by the snippet shown.
# ============================================================


# ────────────────────────────────────────────────────────────
# FIX #2  load_csv() — remove silent `del fields`
# ────────────────────────────────────────────────────────────
# FIND this inside load_csv():
#
#     def load_csv(path: str, fields: Optional[List[str]] = None) -> List[Dict[str, Any]]:
#         ensure_storage()
#         key = storage_key_for_path(path)
#         del fields  # ← DELETE THIS LINE
#
# CHANGE TO:
#
#     def load_csv(path: str, fields: Optional[List[str]] = None) -> List[Dict[str, Any]]:
#         ensure_storage()
#         key = storage_key_for_path(path)
#         # fields is intentionally unused here; kept for API compatibility.
#         _ = fields
#
# WHY: `del fields` silently drops schema hints callers pass in.
# Renaming to `_ = fields` makes the intent explicit and avoids
# accidental "variable used before assignment" issues in future edits.


# ────────────────────────────────────────────────────────────
# FIX #3  save_csv() — real round-trip verification
# ────────────────────────────────────────────────────────────
# FIND this block inside save_csv():
#
#     if ok:
#         check = sb_get_storage(key)
#         if check:
#             debug_log(f"supabase_store_verified key={key}")
#             st.session_state["_save_badge"] = "💾 saved (supabase verified)"
#         else:
#             debug_log(f"supabase_store_unverified key={key}")
#             st.session_state["_save_badge"] = "⚠️ saved local, supabase unverified"
#
# CHANGE TO:
#
#     if ok:
#         check = sb_get_storage(key)
#         # FIX #3: compare content, not just truthiness, to catch stale reads.
#         if check is not None and check == content:
#             debug_log(f"supabase_store_verified key={key}")
#             st.session_state["_save_badge"] = "💾 saved (supabase verified)"
#         else:
#             debug_log(f"supabase_store_unverified key={key}")
#             st.session_state["_save_badge"] = "⚠️ saved local, supabase unverified"
#
# WHY: PostgREST can return a cached/stale value. Checking `check == content`
# confirms the actual bytes were stored, not just that *something* is there.


# ────────────────────────────────────────────────────────────
# FIX #5  Move WORKER_FAST_MODE definition to the TOP of app.py
#         so st.set_page_config() can be guarded before it fires.
# ────────────────────────────────────────────────────────────
# FIND the imports block near the top of app.py (after all `import` lines).
# Look for the line:
#
#     WORKER_FAST_MODE = os.getenv("DEX_SCOUT_WORKER_MODE", "0") == "1"
#
# This line currently appears ~50 lines into the file, AFTER st.set_page_config().
# 
# STEP 1 — DELETE that line from its current location.
#
# STEP 2 — INSERT the following block immediately after the last `import` line
#           and BEFORE any `st.*` call:
#
#     # ── Worker-mode flag — must be defined before any Streamlit calls ──
#     WORKER_FAST_MODE = os.getenv("DEX_SCOUT_WORKER_MODE", "0") == "1"
#
# STEP 3 — FIND the st.set_page_config() call:
#
#     st.set_page_config(page_title="DEX Scout", layout="wide")
#
# CHANGE TO:
#
#     if not WORKER_FAST_MODE:
#         st.set_page_config(page_title="DEX Scout", layout="wide")
#
# WHY: worker.py imports app.py via import_module("app"). Module-level
# st.set_page_config() raises StreamlitAPIException when there is no active
# Streamlit session, crashing the worker on every invocation.


# ────────────────────────────────────────────────────────────
# FIX #7  DIGEST_UI_HEARTBEAT_HOURS / DIGEST_DISCOVERY_HEARTBEAT_HOURS
#         — use _env_int() to avoid ValueError on bad env values
# ────────────────────────────────────────────────────────────
# FIND:
#
#     DIGEST_UI_HEARTBEAT_HOURS = max(1, int(os.getenv("DIGEST_UI_HEARTBEAT_HOURS", "12") or 12))
#     DIGEST_DISCOVERY_HEARTBEAT_HOURS = max(1, int(os.getenv("DIGEST_DISCOVERY_HEARTBEAT_HOURS", "12") or 12))
#
# CHANGE TO:
#
#     DIGEST_UI_HEARTBEAT_HOURS = max(1, _env_int("DIGEST_UI_HEARTBEAT_HOURS", 12))
#     DIGEST_DISCOVERY_HEARTBEAT_HOURS = max(1, _env_int("DIGEST_DISCOVERY_HEARTBEAT_HOURS", 12))
#
# NOTE: _env_int() must already be defined before this point.
#       It is currently defined further down in app.py — move it above
#       these constants, or move the constants below _env_int().
#       The simplest approach: move _env_int() to just after the imports block.
#
# WHY: int("12h") raises ValueError and crashes the app at import time.
#      _env_int() has a try/except and returns the default safely.


# ────────────────────────────────────────────────────────────
# FIX #8  sb_get_storage() — handle non-404 HTTP errors gracefully
# ────────────────────────────────────────────────────────────
# FIND inside sb_get_storage():
#
#     if r.status_code == 404:
#         debug_log(f"supabase_read_404 key={key}")
#         return None
#     r.raise_for_status()
#
# CHANGE TO:
#
#     if r.status_code == 404:
#         debug_log(f"supabase_read_404 key={key}")
#         return None
#     # FIX #8: treat all other HTTP errors (500, 503, 429, etc.) the same as
#     # 404 — return None rather than raising an unhandled exception in callers.
#     if r.status_code >= 400:
#         debug_log(f"supabase_read_http_error key={key} status={r.status_code}")
#         return None
#
# WHY: raise_for_status() on a 429 or 503 propagates an uncaught exception
# up through load_csv() and save_csv(), potentially breaking persistence
# for the entire request cycle rather than degrading gracefully.
