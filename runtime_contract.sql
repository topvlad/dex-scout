-- runtime_contract.sql — DEX Scout
-- Minimal runtime persistence contract for DEX Scout background automation.
-- Apply in Supabase SQL editor.

create table if not exists public.runtime_state (
    state_key      text primary key,
    state_json     jsonb not null default '{}'::jsonb,
    updated_ts     text not null default '',
    updated_epoch  double precision not null default 0
);

create table if not exists public.locks (
    lock_key      text primary key,
    owner         text not null default '',
    expires_epoch double precision not null default 0,
    updated_epoch double precision not null default 0
);

create table if not exists public.tg_state (
    state_key     text primary key,
    state_json    jsonb not null default '{}'::jsonb,
    updated_epoch double precision not null default 0
);

create table if not exists public.job_heartbeats (
    job_name        text primary key,
    job_mode        text not null default '',
    heartbeat_ts    text not null default '',
    heartbeat_epoch double precision not null default 0,
    status          text not null default '',
    meta_json       jsonb not null default '{}'::jsonb
);

-- Optional audit table (not required for D0.3 acceptance).
-- FIX #9a: added created_at for time-range queries.
create table if not exists public.job_runs (
    run_id     text primary key,
    job_name   text not null,
    job_mode   text not null default '',
    started_ts text not null default '',
    ended_ts   text not null default '',
    status     text not null default '',
    note       text not null default '',
    -- FIX #9a: timestamp column so you can query/prune by time.
    created_at timestamptz not null default now()
);

-- FIX #9b: index on job_name to avoid full-table scans as job_runs grows.
create index if not exists job_runs_job_name_idx on public.job_runs (job_name);
-- Optional composite index if you frequently query by job_name + status:
-- create index if not exists job_runs_job_name_status_idx on public.job_runs (job_name, status);

-- NOTE on RLS (Row Level Security):
-- If job_runs is ever exposed via the anon key (e.g. from Streamlit using
-- SUPABASE_ANON_KEY), enable RLS and add a policy. Example:
--
--   alter table public.job_runs enable row level security;
--   create policy "service_role_only" on public.job_runs
--     using (auth.role() = 'service_role');
--
-- Currently the worker uses SUPABASE_SERVICE_ROLE_KEY which bypasses RLS,
-- so this is low priority — but worth doing before any public-facing reads.

create or replace view public.app_storage_sizes as
select
    key,
    octet_length(content) as bytes,
    updated_at
from public.app_storage
order by octet_length(content) desc;
