-- Identity and login history for the DPDP Compliance Assistant.
-- Run once in the Supabase dashboard: SQL Editor → New query → Run.
--
-- Supabase is scoped to identity and authentication events ONLY — who
-- signed in, and when. No question, answer, or citation content is ever
-- written here. That content (every Q&A, full answer, retrieval trace) lives
-- in MongoDB instead — see backend/mongo.py for why and how, and for the
-- one thing that ties the two stores together: every `user_id` in either
-- database is the exact same value, auth.users.id, the UUID from the
-- verified JWT's `sub` claim. There is only one place a user id is ever
-- minted, so a Postgres row and a Mongo document for the same person always
-- share it, and can be correlated on it later without anything extra built
-- to keep them in sync.
--
-- Drop the old usage_events table if you ran an earlier version of this
-- file — it stored per-question content that has since moved to MongoDB,
-- and its shape has nothing in common with login_events below:
--   drop table if exists public.usage_events cascade;


-- ============================================================================
-- profiles — current snapshot. Who they are, when they were last seen.
--
-- Supabase already records every sign-in in its own internal `auth.users`,
-- visible in the dashboard under Authentication -> Users. That table is not
-- meant to be queried or joined against directly from application code, and
-- its schema is Supabase's to change. `profiles` is the standard pattern:
-- one row per person, kept in sync by a trigger, safe to query and extend.
--
-- Populated automatically. Nothing in the backend writes to this table —
-- it exists purely at the database level, so a user shows up here the
-- moment they complete Google sign-in, even before they ask anything.
-- ============================================================================

create table if not exists public.profiles (
    id               uuid primary key references auth.users(id) on delete cascade,
    email            text,
    full_name        text,
    avatar_url       text,
    created_at       timestamptz not null default now(),   -- first sign-in
    last_sign_in_at  timestamptz                            -- most recent sign-in
);

alter table public.profiles enable row level security;
-- Row-level security ON, and deliberately WITHOUT any policy on this or any
-- table below. With RLS enabled and no policy granted, the `anon` and
-- `authenticated` roles can do nothing at all here — no select, no insert,
-- not even the signed-in user's own row. Only the service-role key
-- (server-side only, never sent to a browser) can read or write. So even if
-- the anon key embedded in the page were extracted (it is public by design),
-- it grants zero access to anyone's identity or login history.


-- ============================================================================
-- login_events — full history. One row per sign-in, forever.
--
-- profiles.last_sign_in_at is overwritten on every login, so it can only
-- ever answer "when was this person last seen," never "how many times has
-- this person signed in, and when." login_events is the append-only record
-- that can. It costs nothing extra to maintain: the same trigger that
-- already upserts profiles just gains a second insert.
-- ============================================================================

create table if not exists public.login_events (
    id         bigint generated always as identity primary key,
    user_id    uuid        not null references auth.users(id) on delete cascade,
    email      text,
    full_name  text,
    login_at   timestamptz not null default now()
);

create index if not exists login_events_user_id_login_at_idx
    on public.login_events (user_id, login_at desc);

alter table public.login_events enable row level security;
-- Nothing in the backend currently reads this table — it exists for direct
-- SQL in the Supabase dashboard, same as profiles. A future endpoint reading
-- it is a deliberate addition, not implied by the table existing.

-- `security definer` is required here: this function reads/writes auth.users
-- and public tables, which the invoking role (the trigger firing on
-- auth.users) does not itself have privileges over. `set search_path = public`
-- pins name resolution so the function cannot be tricked by a schema placed
-- earlier on some other search path.
create or replace function public.handle_auth_user_change()
returns trigger
language plpgsql
security definer set search_path = public
as $$
begin
  insert into public.profiles (id, email, full_name, avatar_url, last_sign_in_at)
  values (
    new.id,
    new.email,
    new.raw_user_meta_data ->> 'full_name',
    new.raw_user_meta_data ->> 'avatar_url',
    new.last_sign_in_at
  )
  on conflict (id) do update
    set email           = excluded.email,
        full_name       = excluded.full_name,
        avatar_url      = excluded.avatar_url,
        last_sign_in_at = excluded.last_sign_in_at;

  -- Every fire of this function IS a login — the first-ever sign-up counts
  -- as the first login too — so log it unconditionally. This is the one
  -- addition beyond the profiles upsert above; no new trigger, no backend
  -- code, no involvement from this app's server at all. Supabase Auth
  -- updates auth.users on every Google sign-in on its own, whether or not
  -- the backend happens to be running at that moment.
  insert into public.login_events (user_id, email, full_name, login_at)
  values (new.id, new.email, new.raw_user_meta_data ->> 'full_name', new.last_sign_in_at);

  return new;
end;
$$;

-- Fires once per new sign-up (first-ever login).
drop trigger if exists on_auth_user_created on auth.users;
create trigger on_auth_user_created
    after insert on auth.users
    for each row execute function public.handle_auth_user_change();

-- Fires on every SUBSEQUENT login too, since Supabase updates
-- auth.users.last_sign_in_at each time.
drop trigger if exists on_auth_user_login on auth.users;
create trigger on_auth_user_login
    after update of last_sign_in_at on auth.users
    for each row execute function public.handle_auth_user_change();


-- ============================================================================
-- Sanity checks and useful queries
-- ============================================================================

-- Should show your own account after you sign in again:
--   select email, full_name, created_at, last_sign_in_at
--   from public.profiles
--   order by last_sign_in_at desc;

-- Full login history for one person — profiles only ever has one row per
-- user; this can have many:
--   select login_at from public.login_events
--   where user_id = '<uuid from profiles.id>'
--   order by login_at desc;

-- Who has logged in the most, and when they were first/last seen:
--   select p.email, p.full_name, count(l.*) as logins,
--          min(l.login_at) as first_login, max(l.login_at) as last_login
--   from public.profiles p
--   left join public.login_events l on l.user_id = p.id
--   group by p.id, p.email, p.full_name
--   order by logins desc;

-- Correlating with Q&A activity in MongoDB: take a user_id from either query
-- above, then in Mongo (mongosh, Compass, or Atlas's own query bar):
--   db.interactions.countDocuments({ user_id: "<same uuid>" })
-- Same UUID, two different database engines — there is no live SQL JOIN
-- across them, but every write on both sides sources user_id from the same
-- place (the verified JWT's `sub`), so this always lines up.
