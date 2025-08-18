-- =========================
-- Extensions & schema
-- =========================
create extension if not exists pgcrypto;   -- for gen_random_uuid()
create extension if not exists vector;     -- pgvector

create schema if not exists clinical;

-- =========================
-- Core tables
-- =========================
create table if not exists clinical.doctors (
  id           uuid primary key default gen_random_uuid(),
  full_name    text        not null,
  specialization text      not null,
  email        text unique,
  phone        text,
  created_at   timestamptz not null default now()
);

create table if not exists clinical.patients (
  id           uuid primary key default gen_random_uuid(),
  full_name    text        not null,
  gender       text,
  dob          date,
  email        text,
  phone        text,
  created_at   timestamptz not null default now()
);

create table if not exists clinical.appointments (
  id           uuid primary key default gen_random_uuid(),
  patient_id   uuid        not null references clinical.patients(id) on delete cascade,
  doctor_id    uuid        not null references clinical.doctors(id)  on delete cascade,
  starts_at    timestamptz not null,
  ends_at      timestamptz not null,
  status       text        not null check (status in ('scheduled','completed','cancelled','no_show')),
  notes        text,
  created_at   timestamptz not null default now()
);
create index if not exists appt_patient_idx on clinical.appointments (patient_id);
create index if not exists appt_doctor_idx  on clinical.appointments (doctor_id);
create index if not exists appt_starts_idx  on clinical.appointments (starts_at);

create table if not exists clinical.tests (
  id              uuid primary key default gen_random_uuid(),
  patient_id      uuid        not null references clinical.patients(id) on delete cascade,
  doctor_id       uuid        references clinical.doctors(id) on delete set null,
  test_name       text        not null,
  ordered_at      timestamptz not null,
  collected_at    timestamptz,
  reported_at     timestamptz,
  status          text        not null default 'completed' check (status in ('ordered','completed','cancelled')),
  result_value    numeric,
  result_unit     text,
  reference_range text,
  created_at      timestamptz not null default now()
);
create index if not exists tests_patient_idx  on clinical.tests (patient_id);
create index if not exists tests_doctor_idx   on clinical.tests (doctor_id);
create index if not exists tests_reported_idx on clinical.tests (reported_at);

-- Helpful view: next appointment per patient (future, scheduled)
create or replace view clinical.v_patient_next_appointment as
select
  p.id as patient_id,
  p.full_name,
  a.id as appointment_id,
  a.starts_at as next_starts_at
from clinical.patients p
left join lateral (
  select a.*
  from clinical.appointments a
  where a.patient_id = p.id
    and a.status = 'scheduled'
    and a.starts_at >= now()
  order by a.starts_at
  limit 1
) a on true;

-- =========================
-- Schema embeddings (RAG for SQL)
-- =========================
create table if not exists clinical.schema_chunks (
  id        bigserial primary key,
  content   text           not null,
  embedding vector(1024)   not null
);

-- Choose ONE index type (IVFFlat baseline here). ANALYZE after bulk insert.
create index if not exists schema_chunks_ivfflat_idx
  on clinical.schema_chunks using ivfflat (embedding vector_cosine_ops)
  with (lists = 100);

-- (Optional alternative)
-- create index if not exists schema_chunks_hnsw_idx
--   on clinical.schema_chunks using hnsw (embedding vector_cosine_ops)
--   with (m = 16, ef_construction = 200);

-- =========================
-- Document RAG storage
-- =========================
create table if not exists clinical.documents (
  id         uuid primary key default gen_random_uuid(),
  title      text,
  filename   text,
  mime       text,
  size       bigint,
  created_at timestamptz not null default now()
);

create table if not exists clinical.doc_chunks (
  id        bigserial primary key,
  doc_id    uuid         not null references clinical.documents(id) on delete cascade,
  content   text         not null,
  embedding vector(1024) not null
);

create index if not exists doc_chunks_ivfflat_idx
  on clinical.doc_chunks using ivfflat (embedding vector_cosine_ops)
  with (lists = 100);

-- (Optional alternative)
-- create index if not exists doc_chunks_hnsw_idx
--   on clinical.doc_chunks using hnsw (embedding vector_cosine_ops)
--   with (m = 16, ef_construction = 200);

-- =========================
-- RPCs (PostgREST functions)
-- =========================

-- Read-only SQL runner with guardrails:
--  - single statement (no semicolons)
--  - blocks DDL/DML and dangerous commands by regex
--  - returns rows as JSONB array items
create or replace function clinical.run_sql_ro(q text)
returns setof jsonb
language plpgsql
security definer
set search_path = clinical, public
as $$
declare
  bad boolean;
begin
  -- Block multi-statement attempts
  if q ~ ';' then
    raise exception 'multiple statements not allowed';
  end if;

  -- Block dangerous keywords (case-insensitive, word boundary)
  bad := q ~* '(^|[^a-z])(insert|update|delete|alter|create|drop|grant|revoke|truncate|comment|vacuum|refresh|copy|call|do|security|owner|cluster|analyze|listen|notify|begin|commit|rollback|savepoint|lock|set|reset)\b';
  if bad then
    raise exception 'blocked keyword detected';
  end if;

  -- Keep queries read-only; optional statement timeout (~5s)
  perform set_config('statement_timeout','5000', true);

  -- Execute and materialize as JSONB rows
  return query execute format($f$
    select to_jsonb(t) from (%s) as t
  $f$, q);

end
$$;

-- KNN over schema chunks (Top-K schema)
create or replace function clinical.match_schema(
  query_embedding vector(1024),
  match_count int
) returns table (content text, similarity real)
language sql
stable
security definer
set search_path = clinical, public
as $$
  select sc.content,
         1 - (sc.embedding <=> query_embedding) as similarity
  from clinical.schema_chunks sc
  order by sc.embedding <=> query_embedding
  limit greatest(match_count, 1)
$$;

-- Document ingestion helpers
create or replace function clinical.add_document(
  p_title text,
  p_filename text,
  p_mime text,
  p_size bigint
) returns uuid
language sql
security definer
set search_path = clinical, public
as $$
  insert into clinical.documents(title, filename, mime, size)
  values (p_title, p_filename, p_mime, p_size)
  returning id
$$;

create or replace function clinical.add_doc_chunk(
  p_doc_id uuid,
  p_content text,
  p_embedding vector(1024)
) returns void
language sql
security definer
set search_path = clinical, public
as $$
  insert into clinical.doc_chunks(doc_id, content, embedding)
  values (p_doc_id, p_content, p_embedding)
$$;

create or replace function clinical.match_doc_chunks(
  query_embedding vector(1024),
  match_count int,
  p_doc_id uuid default null
) returns table (doc_id uuid, content text, similarity real)
language sql
stable
security definer
set search_path = clinical, public
as $$
  select dc.doc_id,
         dc.content,
         1 - (dc.embedding <=> query_embedding) as similarity
  from clinical.doc_chunks dc
  where p_doc_id is null or dc.doc_id = p_doc_id
  order by dc.embedding <=> query_embedding
  limit greatest(match_count, 1)
$$;

-- =========================
-- Grants for client usage
-- (client uses 'anon' role via Supabase)
-- =========================
grant usage on schema clinical to anon;

grant execute on function clinical.run_sql_ro(text)                 to anon;
grant execute on function clinical.match_schema(vector(1024), int)  to anon;
grant execute on function clinical.add_document(text, text, text, bigint) to anon;
grant execute on function clinical.add_doc_chunk(uuid, text, vector(1024)) to anon;
grant execute on function clinical.match_doc_chunks(vector(1024), int, uuid) to anon;

-- No broad table SELECT/INSERT to anon needed, thanks to SECURITY DEFINER RPCs.

-- =========================
-- Maintenance
-- =========================
-- Refresh PostgREST schema cache
select pg_notify('pgrst','reload schema');

-- After bulk inserts of vectors, analyze for best query plans
analyze clinical.schema_chunks;
analyze clinical.doc_chunks;
