"""
Diagnose both databases before trying the real app — Supabase (identity,
login history) and MongoDB (every question and answer). Safe to run before
either has any data in it; nothing here reads or writes real content.

    1. .env has the required Supabase and MongoDB variables set
    2. the Supabase project is reachable and the anon key is valid
    3. the service-role key is valid (tested against `login_events`, which
       does NOT need to exist yet — a 401/403 means the key itself is wrong;
       a 404/relation-does-not-exist means the key is fine and only the
       table hasn't been created yet, via supabase_setup.sql)
    4. Google is actually enabled as a sign-in provider
    5. MongoDB is reachable and the credentials in MONGODB_URI authenticate

Check 4 exists because "Unsupported provider: provider is not enabled" is a
browser-side error from Supabase Auth, not a Python exception — nothing in
the backend would ever surface it. This script asks Supabase directly
instead of waiting for a user to hit the button and be confused by it.

Supabase checks use stdlib `urllib` only, matching `llm.py`/`mongo.py`'s
preference for plain HTTP over an SDK for simple REST calls. The MongoDB
check uses `pymongo` since that's already a real dependency of the app
(`backend/mongo.py`) — no point avoiding it here.

    python test_db.py
"""
from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

ENV_PATH = Path(__file__).resolve().parent / ".env"


def load_env(path: Path) -> dict[str, str]:
    """Same minimal format as backend/config.py's load_dotenv, kept
    standalone here so this script has no import-time dependency on
    backend.config — that module hard-requires NEO4J_* to be set, which has
    nothing to do with whether Supabase is reachable."""
    env: dict[str, str] = {}
    if not path.is_file():
        return env
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        env[key.strip()] = value.strip().strip("'\"")
    return env


def get(url: str, headers: dict) -> tuple[int, bytes]:
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()
    except urllib.error.URLError as exc:
        raise SystemExit(f"cannot reach {url}: {exc.reason}")


def main() -> int:
    print(f"reading {ENV_PATH.name}\n")
    env = load_env(ENV_PATH)

    url = env.get("SUPABASE_URL", "").rstrip("/")
    anon_key = env.get("SUPABASE_ANON_KEY", "")
    service_key = env.get("SUPABASE_SERVICE_ROLE_KEY", "")
    mongo_uri = env.get("MONGODB_URI", "")
    mongo_db = env.get("MONGODB_DB", "compliance_assistant")

    ok = True

    # --- 1. Required variables present -------------------------------- #
    print("1. .env variables")
    for name, value in (("SUPABASE_URL", url),
                        ("SUPABASE_ANON_KEY", anon_key),
                        ("SUPABASE_SERVICE_ROLE_KEY", service_key),
                        ("MONGODB_URI", mongo_uri)):
        present = bool(value) and not value.startswith("<")
        print(f"   {'OK  ' if present else 'MISS'}  {name}")
        ok &= present
    if not ok:
        print("\nFill in the missing values in .env, then run this again.")
        return 1
    print()

    # --- 2. Project reachable, anon key valid -------------------------- #
    print("2. project reachable (anon key)")
    status, body = get(f"{url}/rest/v1/", {"apikey": anon_key})
    # PostgREST's root always answers even with zero tables — this proves
    # the project exists and the anon key authenticates, nothing more.
    if status < 500:
        print(f"   OK    reached {url}  (HTTP {status})")
    else:
        print(f"   FAIL  HTTP {status}: {body[:200]!r}")
        ok = False
    print()

    # --- 3. Service-role key valid --------------------------------------- #
    print("3. service-role key (queried against login_events)")
    status, body = get(
        f"{url}/rest/v1/login_events?select=id&limit=1",
        {"apikey": service_key, "Authorization": f"Bearer {service_key}"})
    if status == 404 or (status == 400 and b"does not exist" in body.lower()):
        print("   OK    key authenticates; login_events does not exist yet "
              "(expected — run supabase_setup.sql to create it)")
    elif status in (401, 403):
        print(f"   FAIL  HTTP {status}: the key itself is rejected — "
              f"re-check SUPABASE_SERVICE_ROLE_KEY. {body[:200]!r}")
        ok = False
    elif status == 200:
        print("   OK    key authenticates; login_events already exists")
    else:
        print(f"   ?     HTTP {status}: {body[:200]!r}")
        ok = False
    print()

    # --- 4. Google provider actually enabled ----------------------------- #
    print("4. Google sign-in provider")
    status, body = get(f"{url}/auth/v1/settings", {"apikey": anon_key})
    if status != 200:
        print(f"   FAIL  HTTP {status} reading auth settings: {body[:200]!r}")
        ok = False
    else:
        settings = json.loads(body)
        enabled = (settings.get("external") or {}).get("google")
        if enabled:
            print("   OK    Google is enabled")
        else:
            print("   MISS  Google is NOT enabled — this is exactly the "
                  "\"Unsupported provider\" error from the browser.")
            print("         Dashboard -> Authentication -> Providers -> Google -> "
                  "toggle it on, add the Google OAuth client id/secret.")
            ok = False
    print()

    # --- 5. MongoDB reachable and authenticated --------------------------- #
    print("5. MongoDB (question and answer storage)")
    try:
        from pymongo import MongoClient
        from pymongo.errors import PyMongoError
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=8000, connectTimeoutMS=8000)
        # A real round trip, not just constructing the client (which never
        # connects on its own) — `ping` is the standard way to force one.
        client.admin.command("ping")
        count = client[mongo_db]["interactions"].estimated_document_count()
        print(f"   OK    reachable; {mongo_db}.interactions has {count} document(s)")
        client.close()
    except ImportError:
        print("   FAIL  pymongo is not installed — pip install -r requirements.txt")
        ok = False
    except PyMongoError as exc:
        print(f"   FAIL  {type(exc).__name__}: re-check MONGODB_URI, and that "
              f"Network Access in Atlas allows this machine's IP.")
        ok = False
    print()

    print("ALL CHECKS PASSED" if ok else "SEE FAILURES ABOVE")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
