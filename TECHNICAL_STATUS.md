# DEX Scout — Technical Status Brief

## What DEX Scout currently does

### 1) Runtime architecture and process separation
- Splits runtime into three roles plus CI/runtime orchestration:
  - Streamlit UI (`app.py`) for visualization, manual controls, and manual add-to-monitoring operations.
  - FastAPI webhook (`tg_webhook.py`) for Telegram callback actions and control-path routing.
  - Worker loop (`worker.py` / `scanner_worker.py`) for autonomous `scan_cycle`, `monitor_cycle`, `notify_cycle`, `digest_cycle`, and `outcome_cycle` execution.
  - GitHub Actions workflows for runtime jobs orchestration and a separate CI workflow for compile/test/runtime-safety checks.
- Live Pulse render path in UI is non-mutating: UI reads compact pulse history but does not write or backfill it.
- Pulse history write path is worker/scanner-driven after scan/monitor cycles.

### 2) Import/runtime safety and contracts
- `WORKER_FAST_MODE` is defined before `st.set_page_config`, enabling import-safe worker usage of `app.py`.
- Worker imports UI module in a runtime-safe manner (without forcing UI bootstrap behavior).
- Runtime contract checks validate required Supabase entities and operational state prerequisites.
- Worker uses mode-scoped lock/heartbeat/runtime diagnostics to detect stale loops and degraded execution.

### 3) Data sources and token discovery
- Pulls market/pair data from DexScreener API.
- Supports rotating scanner seeds and optional Birdeye trending feed integration (env-gated).
- Live Pulse candidates are sourced from backend discovery candidate pool.
- Includes 429-pressure controls for pair fetching.
- Chain focus remains Solana + BSC.

### 4) Signal processing and scoring
- Applies hard filters + weighted scoring (liquidity, tx imbalance, volume, price-change penalties).
- Maintains monitoring prioritization logic (`run_priority_scanner_cycle`) with adaptive sleep hints.
- Includes queue-order telemetry assertions for tier invariants in background loop.

### 5) State, persistence, and storage fallback
- Uses Supabase-backed persistence when credentials are available.
- Retains local CSV fallback for operational continuity.
- `monitoring_history.csv` no longer relies on default hot-path full Supabase blob read/write.
- `MON_HISTORY_SUPABASE_MODE` controls monitoring-history persistence mode; low-egress default is `local_only`.
- `pulse_history_compact.json` provides compact bounded shared history for Live Pulse mini-sparklines.
- Pulse history is written by worker/scanner path, not by UI rendering path.

### 6) Telegram and notification workflows
- Worker emits notifications from `notify_cycle`.
- Discovery digest is actionable-only; empty/non-actionable digest is intentionally suppressed.
- Raw backend candidate list payloads (e.g., `solana|...`) are not intended operator-facing digest output.
- Portfolio meaningful alerts are emitted for:
  - `REDUCE`, `CLOSE`, `EXIT`, `TRIM`, `TAKE_PROFIT`;
  - `HIGH` / `CRITICAL` risk;
  - dead/cold, liquidity drop, drawdown, stale data;
  - linked monitoring `NO_ENTRY` / `EXIT` / `AVOID`.
- `HOLD`, `UNKNOWN`, and `score n/a` are intentionally not treated as meaningful portfolio alerts.
- Compact Telegram suppression heartbeat exists.

### 7) Manual Monitoring control plane
- User can manually add tokens to Monitoring from UI using:
  - raw contract address (CA);
  - DexScreener URL;
  - pair address / pair-style input with safe minimal fallback.
- Canonical path is `manual_add_token_to_monitoring(...)`.
- Telegram `+ Monitor` route uses the same canonical helper path.
- Active-row dedupe prevents duplicate Monitoring entries.
- Solana CA casing is preserved.
- If enrichment is incomplete/fails, system can persist minimal deferred rows (e.g., `MANUAL WATCH` / `PAIR_FETCH_DEFERRED`) so tracking is not lost.
- Manual add does not auto-add to Portfolio and does not trigger auto-buy/auto-execution.

### 8) Operational safety and security controls (current)
- Webhook bootstrap includes fail-fast validation for critical endpoints/config.
- `/health` remains lightweight and independently available.

## What DEX Scout should do (target behavior)

### A) Deterministic runtime and idempotency
- Guarantee exactly-once semantics for per-token notification events (cross-worker/process safe).
- Ensure every cycle mode remains idempotent and restart-safe under concurrency.

### B) Strong typed domain model
- Replace dict-heavy row passing with typed schemas (Pydantic/dataclass) across ingest→score→persist→notify.
- Enforce canonical entity key consistency at typed boundaries.

### C) Storage reliability
- Migrate journals/history to normalized append-only durable tables with strict constraints and migration management.
- Add write-ahead buffering / retry queues for transient DB/API outages.

### D) Strategy and evaluation
- Version scoring/filter policies and persist policy version with emitted events.
- Provide offline replay/backtest over historical snapshots for precision/recall and PnL-proxy evaluation.

### E) Observability
- Export structured metrics (latency, error rate, queue depth, emit counts, duplicate suppressions).
- Add machine-readable health and alert thresholds tied to runtime SLOs.

### F) Security and controls
- Strengthen webhook auth and replay protection.
- Introduce scoped operator RBAC.
- Implement secret rotation workflow.
- Separate least-privilege runtime roles/accounts.

## What it does not do yet / gaps

### Runtime/eventing and control plane
- No guaranteed exactly-once cross-worker notification/event bus.
- No robust full operator command surface.
- No dedicated Telegram DLQ.

### Data model and persistence
- No full typed domain model at all ingestion/processing boundaries.
- Journals/history are not yet fully migrated to normalized append-only DB schema.
- No write-ahead retry queue / DLQ for failed persistence writes.

### Testing and CI maturity
- Baseline automated tests exist, but no comprehensive integration/e2e/replay/backtest suite.
- No full integration suite against frozen market snapshots.
- No CI environment smoke against real Supabase/Telegram secrets.
- Basic CI quality gates exist; advanced runtime-quality gates remain limited.

### Live pulse and history depth
- Compact pulse history is bounded operational state, not a full historical OHLCV/feature store.
- Sparse/new candidates can show warm-up behavior (insufficient points).
- No extra fetches are performed solely to draw UI sparklines.

### Manual monitoring enrichment limits
- Full pair-address-to-base-token enrichment is not guaranteed in all cases.
- Pair-style manual inputs can remain in minimal/deferred tracking path until later enrichment.

### Product scope limits
- No portfolio execution automation / auto-trading.
- No backtesting-grade full historical research pipeline.
- No ML-based scoring pipeline in production.
- No multi-user/SaaS tenancy model.
- No on-chain transaction analysis engine.
- No real-time websocket market ingest layer.

## Testing and CI (current state)
- Minimal pytest baseline exists and is active.
- Covered baseline areas include:
  - `alerts.py` behavior;
  - `dex.py` behavior;
  - `scoring.py` behavior;
  - `worker.py` behavior;
  - `tg_webhook.py` bootstrap behavior;
  - manual monitoring add parsing/dedupe paths (including Solana casing preservation).
- CI currently runs:
  - `py_compile`;
  - `pytest -q`;
  - import-order guard asserting `WORKER_FAST_MODE` placement before `st.set_page_config`.

## Good to have across major functional areas

### 1) Ingestion
- Multi-source adapters with source-health scoring and fallback ordering.
- Raw payload archival for replay and incident forensics.

### 2) Signal engine
- Policy versioning + feature flags + canary rollout.
- Regime-aware scoring templates (trend/mean-revert/high-volatility).

### 3) State and storage
- Strict DB constraints for event-key uniqueness.
- Migration tooling + backward-compatible readers.

### 4) Notifications
- Per-channel QoS (rate limits, retries, DLQ, exponential backoff).
- User-level routing preferences and suppression windows.

### 5) Runtime orchestration
- Distributed lock observability dashboard.
- Automatic stuck-lock scavenging with causal metadata.

### 6) Monitoring/observability
- Unified metrics + traces + logs with correlation IDs per cycle/run.
- Error-budget policy and runtime SLO enforcement.

### 7) Security
- Signed webhook payload validation and nonce/timestamp replay defense.
- Secret rotation + key provenance audit.

### 8) QA and delivery
- Deterministic replay tests against frozen market snapshots.
- Pre-deploy conformance suite for runtime contracts and schema invariants.
