-- Minimal runtime persistence contract for DEX Scout background automation.
-- Apply in Supabase SQL editor.

create table if not exists public.runtime_state (
  state_key text primary key,
  state_json jsonb not null default '{}'::jsonb,
  updated_ts text not null default '',
  updated_epoch double precision not null default 0
);

create table if not exists public.locks (
  lock_key text primary key,
  owner text not null default '',
  expires_epoch double precision not null default 0,
  updated_epoch double precision not null default 0
);

create table if not exists public.tg_state (
  state_key text primary key,
  state_json jsonb not null default '{}'::jsonb,
  updated_epoch double precision not null default 0
);

create table if not exists public.job_heartbeats (
  job_name text primary key,
  job_mode text not null default '',
  heartbeat_ts text not null default '',
  heartbeat_epoch double precision not null default 0,
  status text not null default '',
  meta_json jsonb not null default '{}'::jsonb
);

-- Optional audit table (not required for D0.3 acceptance).
create table if not exists public.job_runs (
  run_id text primary key,
  job_name text not null,
  job_mode text not null default '',
  started_ts text not null default '',
  ended_ts text not null default '',
  status text not null default '',
  note text not null default ''
);
