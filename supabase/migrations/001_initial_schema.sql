-- Enable required extensions
create extension if not exists "uuid-ossp";

-- Organizations table (multi-tenancy root)
create table public.organizations (
  id          uuid primary key default uuid_generate_v4(),
  name        text not null,
  slug        text unique not null,
  plan        text not null default 'free' check (plan in ('free', 'starter', 'pro', 'enterprise')),
  twilio_account_sid   text,
  twilio_auth_token    text,
  twilio_phone_number  text,
  plivo_auth_id        text,
  plivo_auth_token     text,
  plivo_phone_number   text,
  bolna_engine_url     text not null default 'http://localhost:8000',
  created_at  timestamptz not null default now(),
  updated_at  timestamptz not null default now()
);

-- Profiles table (extends Supabase auth.users)
create table public.profiles (
  id              uuid primary key references auth.users on delete cascade,
  organization_id uuid references public.organizations on delete set null,
  full_name       text,
  avatar_url      text,
  role            text not null default 'member' check (role in ('owner', 'admin', 'member')),
  created_at      timestamptz not null default now(),
  updated_at      timestamptz not null default now()
);

-- Agents table
create table public.agents (
  id              uuid primary key default uuid_generate_v4(),
  organization_id uuid not null references public.organizations on delete cascade,
  created_by      uuid not null references public.profiles on delete set null,
  name            text not null,
  description     text,
  bolna_config    jsonb not null default '{}',
  prompts         jsonb not null default '{}',
  bolna_agent_id  text,
  status          text not null default 'draft' check (status in ('draft', 'active', 'inactive')),
  created_at      timestamptz not null default now(),
  updated_at      timestamptz not null default now()
);

-- Calls table (telephony interaction logs)
create table public.calls (
  id               uuid primary key default uuid_generate_v4(),
  organization_id  uuid not null references public.organizations on delete cascade,
  agent_id         uuid not null references public.agents on delete cascade,
  call_sid         text,
  direction        text not null default 'outbound' check (direction in ('inbound', 'outbound')),
  from_number      text,
  to_number        text,
  status           text not null default 'initiated' check (status in ('initiated', 'ringing', 'in-progress', 'completed', 'failed', 'busy', 'no-answer')),
  duration_seconds integer,
  recording_url    text,
  transcript_summary text,
  metadata         jsonb default '{}',
  started_at       timestamptz,
  ended_at         timestamptz,
  created_at       timestamptz not null default now()
);

-- Indexes
create index agents_organization_id_idx on public.agents (organization_id);
create index agents_status_idx on public.agents (status);
create index calls_organization_id_idx on public.calls (organization_id);
create index calls_agent_id_idx on public.calls (agent_id);
create index calls_status_idx on public.calls (status);
create index calls_created_at_idx on public.calls (created_at desc);

-- Updated_at trigger function
create or replace function public.handle_updated_at()
returns trigger language plpgsql as $$
begin
  new.updated_at = now();
  return new;
end;
$$;

create trigger organizations_updated_at before update on public.organizations
  for each row execute procedure public.handle_updated_at();

create trigger profiles_updated_at before update on public.profiles
  for each row execute procedure public.handle_updated_at();

create trigger agents_updated_at before update on public.agents
  for each row execute procedure public.handle_updated_at();

-- Auto-create profile on user signup
create or replace function public.handle_new_user()
returns trigger language plpgsql security definer set search_path = public as $$
begin
  insert into public.profiles (id, full_name, avatar_url)
  values (
    new.id,
    new.raw_user_meta_data->>'full_name',
    new.raw_user_meta_data->>'avatar_url'
  );
  return new;
end;
$$;

create trigger on_auth_user_created
  after insert on auth.users
  for each row execute procedure public.handle_new_user();

-- Row Level Security
alter table public.organizations enable row level security;
alter table public.profiles enable row level security;
alter table public.agents enable row level security;
alter table public.calls enable row level security;

-- Organizations: members can read their own org
create policy "org_members_select" on public.organizations
  for select using (
    id in (
      select organization_id from public.profiles where id = auth.uid()
    )
  );

-- Organizations: owners can update their own org
create policy "org_owners_update" on public.organizations
  for update using (
    id in (
      select organization_id from public.profiles
      where id = auth.uid() and role in ('owner', 'admin')
    )
  );

-- Profiles: users can see profiles within their org
create policy "profiles_org_select" on public.profiles
  for select using (
    organization_id in (
      select organization_id from public.profiles where id = auth.uid()
    )
  );

-- Profiles: users can update their own profile
create policy "profiles_self_update" on public.profiles
  for update using (id = auth.uid());

-- Agents: org members can do full CRUD on their org's agents
create policy "agents_org_select" on public.agents
  for select using (
    organization_id in (
      select organization_id from public.profiles where id = auth.uid()
    )
  );

create policy "agents_org_insert" on public.agents
  for insert with check (
    organization_id in (
      select organization_id from public.profiles where id = auth.uid()
    )
  );

create policy "agents_org_update" on public.agents
  for update using (
    organization_id in (
      select organization_id from public.profiles where id = auth.uid()
    )
  );

create policy "agents_org_delete" on public.agents
  for delete using (
    organization_id in (
      select organization_id from public.profiles where id = auth.uid()
    )
  );

-- Calls: org members can read/insert calls for their org
create policy "calls_org_select" on public.calls
  for select using (
    organization_id in (
      select organization_id from public.profiles where id = auth.uid()
    )
  );

create policy "calls_org_insert" on public.calls
  for insert with check (
    organization_id in (
      select organization_id from public.profiles where id = auth.uid()
    )
  );

create policy "calls_org_update" on public.calls
  for update using (
    organization_id in (
      select organization_id from public.profiles where id = auth.uid()
    )
  );
