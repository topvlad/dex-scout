create table if not exists app_storage (
  key text primary key,
  content text not null,
  updated_at text not null
);

create table if not exists runtime_state (
  state_key text primary key,
  state_json text not null default '{}',
  updated_ts text not null default '',
  updated_epoch real not null default 0
);

create table if not exists locks (
  lock_key text primary key,
  owner text not null default '',
  expires_epoch real not null default 0,
  updated_epoch real not null default 0
);

create table if not exists tg_state (
  state_key text primary key,
  state_json text not null default '{}',
  updated_epoch real not null default 0
);

create table if not exists job_heartbeats (
  job_name text primary key,
  job_mode text not null default '',
  heartbeat_ts text not null default '',
  heartbeat_epoch real not null default 0,
  status text not null default '',
  meta_json text not null default '{}'
);

create table if not exists job_runs (
  run_id text primary key,
  job_name text not null,
  job_mode text not null default '',
  started_ts text not null default '',
  ended_ts text not null default '',
  status text not null default '',
  note text not null default '',
  created_at text not null default ''
);

create index if not exists job_runs_job_name_idx on job_runs (job_name);
create index if not exists app_storage_updated_idx on app_storage (updated_at);
create index if not exists locks_expires_idx on locks (expires_epoch);
create index if not exists job_heartbeats_epoch_idx on job_heartbeats (heartbeat_epoch);
