# DEX Scout — Technical Status Brief

## What DEX Scout currently does

### 1) Runtime architecture and process separation
- Splits runtime into three roles:
  - Streamlit UI (`app.py`) for visualization, controls, manual operations.
  - FastAPI webhook (`tg_webhook.py`) for Telegram callback actions.
  - Worker loop (`worker.py` / `scanner_worker.py`) for autonomous scan/monitor/notify cycles.
- Uses env-gated behavior to keep notification emission primarily in worker mode.

### 2) Data sources and token discovery
- Pulls market/pair data from DexScreener API.
- Supports rotating scanner seeds and optional Birdeye trending feed integration (toggle by env flags).
- Focused on Solana + BSC chain handling.

### 3) Signal processing and scoring
- Applies hard filters + weighted scoring (liquidity, tx imbalance, volume, price-change penalties).
- Maintains monitoring queues and prioritization logic (`run_priority_scanner_cycle`) with adaptive sleep hints.
- Includes queue-order telemetry assertions for tier invariants in background loop.

### 4) State, persistence, and storage fallback
- Supports Supabase-backed persistence when credentials are available.
- Falls back to local CSV storage for portfolio/monitoring/history and multiple journal logs.
- Includes runtime-state/lock/heartbeat contract checks and lock-mediated runtime updates.

### 5) Notification and operator workflows
- Runs auto-notifications from worker context.
- Supports digest notifications and outcome journal evaluation cycles.
- Telegram webhook can add/remove/suppress tokens and trigger digest behavior.

### 6) Operational safety and recovery
- Startup fail-fast checks for required env vars and runtime entities.
- Job heartbeats and stale-loop detection to surface degraded/stuck execution.
- Emergency reset script for runtime locks/state + GitHub Actions redispatch + log validation.

## What DEX Scout should do (target behavior)

### A) Deterministic runtime and idempotency
- Guarantee exactly-once semantics for per-token notification events (cross-worker/process safe).
- Ensure every cycle mode (`scan_cycle`, `monitor_cycle`, `notify_cycle`, `digest_cycle`, `outcome_cycle`) is idempotent and restart-safe.

### B) Strong typed domain model
- Replace dict-heavy untyped row passing with typed schemas (Pydantic/dataclass) across ingest→score→persist→notify.
- Enforce canonical entity key consistency (chain + address normalization rules) at type boundary.

### C) Storage reliability
- Move journals/history to append-only durable tables with strict constraints and migration management.
- Add write-ahead buffering / retry queues for transient DB/API outages.

### D) Strategy and evaluation
- Version scoring/filter policies and persist policy version with emitted events.
- Provide offline replay/backtest over historical snapshots to evaluate precision/recall and PnL proxies before production rollout.

### E) Observability
- Export structured metrics (latency, error-rate, queue depth, emit counts, duplicate suppressions).
- Add machine-readable health endpoints and alerting thresholds for runtime SLOs.

### F) Security and controls
- Tighten webhook authn/authz and replay protection.
- Separate secrets per runtime role (least privilege service accounts).

## What it does not do yet / gaps

### Core pipeline
- No explicit exactly-once event bus; dedup appears stateful but not formally transactional.
- No strict schema validation at ingestion boundaries (heavy dict usage).
- No clear contract test suite for chain-specific address normalization edge cases.

### Data quality and modeling
- No explicit confidence/uncertainty modeling for low-liquidity noisy pairs.
- No robust anomaly/outlier detection layer separate from heuristic thresholds.
- No versioned feature store or reproducible model/scoring artifact pipeline.

### Reliability and scale
- No documented horizontal scaling protocol for multiple concurrent workers sharing same lock/state tables under load.
- No explicit dead-letter queue for failed notifications / failed persistence writes.
- No benchmarked throughput/latency profile for high token universe sizes.

### Product/control-plane
- No dedicated admin/API control plane for policy rollout/rollback, kill-switches, or scoped dry-run modes.
- No explicit RBAC/audit trail abstraction for operator actions beyond row-level note fields.

### Testing and release engineering
- No visible comprehensive automated tests (unit/integration/e2e) in repo.
- No explicit migration/versioning framework committed for DB schema evolution.
- Limited CI-visible quality gates for runtime safety regressions.

## Good to have across major functional areas

### 1) Ingestion
- Multi-source adapters with source-health scoring and fallback ordering.
- Raw payload archival for replay and incident forensics.

### 2) Signal engine
- Policy versioning + feature flags + canary policy rollout.
- Regime-aware scoring templates (trend, mean-revert, high-volatility buckets).

### 3) State and storage
- Strict DB schema with unique constraints for event keys.
- Migration tooling + backward-compatible readers during transitions.

### 4) Notifications
- Per-channel QoS (rate limits, retries, DLQ, exponential backoff).
- User-level preference routing and suppression windows.

### 5) Runtime orchestration
- Distributed lock observability dashboard.
- Automatic stuck-lock scavenging with causal metadata.

### 6) Monitoring/observability
- Unified metrics + traces + logs with correlation IDs per cycle/run.
- Error budget policy and SLOs (scan freshness, notify latency, digest timeliness).

### 7) Security
- Signed webhook payload validation and nonce/timestamp replay defense.
- Secret rotation workflow and key provenance audit.

### 8) QA and delivery
- Deterministic replay tests against frozen market snapshots.
- Pre-deploy conformance suite for runtime contract and data schema invariants.
