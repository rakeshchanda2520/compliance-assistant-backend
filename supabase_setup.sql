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

-- ============================================================================
-- The trigger.
--
-- READ THIS BEFORE EDITING. This function runs INSIDE the transaction that
-- creates a user. If it raises, Postgres rolls that transaction back and
-- Supabase Auth returns:
--
--     ?error=server_error&error_code=unexpected_failure
--     &error_description=Database+error+saving+new+user
--
-- and the person cannot sign up AT ALL. Bookkeeping tables must never be
-- able to lock users out of the product. Two defences below, and both are
-- load-bearing:
--
--   1. NEVER pass an explicit NULL into a NOT NULL column. `login_at` has
--      `default now()`, but a default only applies when the column is
--      OMITTED — passing NULL explicitly is a constraint violation, not a
--      fallback. This bit exactly once, and expensively: on INSERT into
--      auth.users, `new.last_sign_in_at` is NULL (GoTrue sets it in a later
--      step), so every attempt at a NEW account failed here while existing
--      users, whose UPDATE carries a real timestamp, kept signing in
--      normally. That asymmetry is what made it look like an account
--      problem rather than a schema one.
--
--   2. The whole body is wrapped in an exception handler. Even a failure
--      nobody predicted degrades to "this sign-in was not recorded" —
--      logged as a warning, visible in the Postgres logs — rather than
--      "this person cannot use the product". Analytics is not worth an
--      authentication outage.
--
-- `security definer` is required: this function writes public tables that
-- the invoking role (the trigger firing on auth.users) has no privileges
-- over. `set search_path = public` pins name resolution so the function
-- cannot be tricked by a schema placed earlier on some other search path.
-- ============================================================================

create or replace function public.handle_auth_user_change()
returns trigger
language plpgsql
security definer set search_path = public
as $$
declare
  -- Resolved ONCE, here, so neither insert below can pass a NULL into a
  -- NOT NULL column. now() is the honest value: this trigger fires as part
  -- of the sign-in, so "now" IS when the sign-in happened.
  signed_in_at timestamptz := coalesce(new.last_sign_in_at, now());
begin
  insert into public.profiles (id, email, full_name, avatar_url, last_sign_in_at)
  values (
    new.id,
    new.email,
    new.raw_user_meta_data ->> 'full_name',
    new.raw_user_meta_data ->> 'avatar_url',
    signed_in_at
  )
  on conflict (id) do update
    set email           = excluded.email,
        full_name       = excluded.full_name,
        avatar_url      = excluded.avatar_url,
        -- greatest(), not a blind overwrite: the INSERT and the UPDATE of
        -- last_sign_in_at are two separate statements on auth.users, so this
        -- function fires twice per sign-up. Assigning excluded directly let
        -- the second fire move the timestamp backwards if the two disagreed.
        last_sign_in_at = greatest(public.profiles.last_sign_in_at,
                                   excluded.last_sign_in_at);

  -- Exactly one row per actual sign-in, which is NOT the same as one row per
  -- fire of this function. A sign-up fires it twice — once for the INSERT
  -- into auth.users, again for the UPDATE where GoTrue stamps
  -- last_sign_in_at — so logging unconditionally recorded every new user's
  -- first login twice, in the one table whose entire purpose is answering
  -- "how many times has this person signed in".
  --
  -- `last_sign_in_at is not null` is the discriminator, and it is exact
  -- rather than a heuristic: that column is null precisely on the INSERT
  -- half of a sign-up and populated on every fire that represents a real
  -- sign-in. No de-duplication window, no unique index to tune.
  if new.last_sign_in_at is not null then
    insert into public.login_events (user_id, email, full_name, login_at)
    values (new.id, new.email,
            new.raw_user_meta_data ->> 'full_name', signed_in_at);
  end if;

  return new;

exception when others then
  -- See defence 2 above. Never re-raise: that would abort the sign-in.
  raise warning 'handle_auth_user_change failed for user %: % (%)',
    new.id, sqlerrm, sqlstate;
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
-- Backfill. Safe to run repeatedly.
--
-- Re-running this file replaces the function, but it does not go back and
-- create rows for people the broken version skipped — anyone who signed in
-- while it was live, plus anyone whose account predates the trigger being
-- installed at all. This fills those in from auth.users, which Supabase
-- maintains itself and which was never affected.
--
-- Only profiles is backfilled. login_events is a record of observed
-- sign-ins; inventing history for logins nobody watched happen would make
-- the one table meant to answer "when did this person sign in" lie.
-- ============================================================================

insert into public.profiles (id, email, full_name, avatar_url, last_sign_in_at)
select u.id,
       u.email,
       u.raw_user_meta_data ->> 'full_name',
       u.raw_user_meta_data ->> 'avatar_url',
       u.last_sign_in_at
from auth.users u
on conflict (id) do update
  set email      = excluded.email,
      full_name  = coalesce(excluded.full_name, public.profiles.full_name),
      avatar_url = coalesce(excluded.avatar_url, public.profiles.avatar_url),
      last_sign_in_at = greatest(public.profiles.last_sign_in_at,
                                 excluded.last_sign_in_at);


-- ============================================================================
-- Sanity checks and useful queries
-- ============================================================================

-- 1. Is the fix actually live? Should print one row containing
--    "coalesce(new.last_sign_in_at, now())". If it does not, the old
--    function is still installed and new sign-ups will keep failing:
--   select prosrc from pg_proc where proname = 'handle_auth_user_change';

-- 2. Is anyone missing a profile? Should be 0 after the backfill above:
--   select count(*) from auth.users u
--    left join public.profiles p on p.id = u.id
--    where p.id is null;

-- 3. If a sign-up still fails, the real Postgres error is in
--    Dashboard -> Logs -> Postgres Logs, as a WARNING beginning
--    "handle_auth_user_change failed for user". The exception handler means
--    it can no longer block the sign-up, so this is now a diagnostic to
--    read at leisure rather than an outage to fix under pressure.

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
