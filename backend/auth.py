"""
Who is asking. Google sign-in, verified server-side.

The browser signs in against Supabase (Google as the social provider) and gets
a JWT. It sends that JWT on every request; this module proves the token is
genuine before a single provision is retrieved or a single token is generated.
Nothing downstream ever takes the caller's word for who they are.

Verification is done locally against the signing key — not by calling
Supabase's `/auth/v1/user` on every request. A network round trip per question
would add latency to the one path users actually wait on, and would make
Supabase's availability a hard dependency of answering at all. Signature
verification gives the same guarantee without either cost.

Two signing algorithms, because Supabase supports both:

    ES256 / RS256   asymmetric "JWT signing keys". Public keys are fetched
                    from the project's JWKS endpoint and cached; no shared
                    secret is needed or stored.
    HS256           the legacy shared `JWT_SECRET`.

Supporting only one is not an option. Supabase is mid-migration between them,
their docs state plainly that the JWKS endpoint "does not return any keys if
you are not using asymmetric JWT signing keys", and they do not document which
way a newly created project defaults. A JWKS-only implementation silently
fails every login on a legacy project; an HS256-only one fails every login on a
migrated project. The token's own `alg` header says which is in use, so that is
what selects the path.

Failures are deliberately uninformative to the caller. "expired" versus
"wrong issuer" versus "malformed" tells an attacker which part of a forged
token to fix next; the real reason is logged, and the caller gets 401.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import jwt
from fastapi import Header, HTTPException

from . import config

log = logging.getLogger(__name__)

# Algorithms this service will accept. Listing them explicitly is the defence
# against the `alg: none` and HS256-signed-with-the-public-key confusion
# attacks: an attacker cannot talk us into a weaker algorithm than one of
# these, because anything else is rejected before a key is even chosen.
ASYMMETRIC = ("ES256", "RS256")
SYMMETRIC = ("HS256",)

_UNAUTHORIZED = HTTPException(
    status_code=401,
    detail="sign in to ask a question",
    headers={"WWW-Authenticate": "Bearer"},
)

# PyJWKClient caches fetched keys and refreshes on an unknown `kid`, so key
# rotation is handled without a restart and without a fetch per request.
_jwks_client: jwt.PyJWKClient | None = None


def _jwks() -> jwt.PyJWKClient:
    global _jwks_client
    if _jwks_client is None:
        _jwks_client = jwt.PyJWKClient(config.SUPABASE_JWKS_URL)
    return _jwks_client


@dataclass(frozen=True)
class Identity:
    """A verified caller. `sub` is Supabase's stable user id — the only field
    safe to key usage records on, since a user can change their email.

    `session_id` is Supabase's own `auth.sessions.id` (the `session_id`
    claim GoTrue puts in every access token) — not a value this service
    mints. Reusing it is what keeps Supabase, MongoDB and Langfuse in sync
    for free: one sign-in produces one UUID, and every store that wants to
    group "this browser session's questions" keys on that same UUID with no
    new state or handshake required anywhere."""

    sub: str
    email: str = ""
    name: str = ""
    session_id: str = ""


def verify(token: str) -> Identity:
    """Decode and validate a Supabase access token, or raise.

    Raises `jwt.PyJWTError` (or `ValueError` for an unusable configuration) —
    the caller is responsible for turning that into a 401 without echoing the
    reason back to the client.
    """
    # Reading the header is not trusting it: it selects which key to verify
    # *against*, and `algorithms=` below still constrains what is acceptable,
    # so a forged header cannot downgrade the check.
    algorithm = jwt.get_unverified_header(token).get("alg")

    if algorithm in ASYMMETRIC:
        key = _jwks().get_signing_key_from_jwt(token).key
        algorithms = list(ASYMMETRIC)
    elif algorithm in SYMMETRIC:
        if not config.SUPABASE_JWT_SECRET:
            raise ValueError(
                "token is signed with HS256 but SUPABASE_JWT_SECRET is not set "
                "— copy it from the Supabase dashboard (Project Settings → API → "
                "JWT Settings), or migrate the project to asymmetric signing keys")
        key = config.SUPABASE_JWT_SECRET
        algorithms = list(SYMMETRIC)
    else:
        raise jwt.InvalidAlgorithmError(f"unsupported signing algorithm {algorithm!r}")

    claims = jwt.decode(
        token,
        key,
        algorithms=algorithms,
        audience=config.SUPABASE_AUDIENCE,
        issuer=config.SUPABASE_ISSUER,
        # Signature alone is not enough. Without `aud`/`iss` pinned, a valid
        # token minted by any *other* Supabase project would verify here.
        # PyJWT does not check either by default.
        options={"require": ["exp", "sub", "aud", "iss"]},
    )

    metadata = claims.get("user_metadata") or {}
    return Identity(
        sub=claims["sub"],
        email=claims.get("email") or metadata.get("email") or "",
        name=metadata.get("full_name") or metadata.get("name") or "",
        session_id=claims.get("session_id") or "",
    )


async def require_user(authorization: str = Header(default="")) -> Identity:
    """FastAPI dependency. Attach to every route that answers questions.

    `/api/health` deliberately does not use this: an infrastructure health
    check must not need a credential to tell you the service is alive.
    """
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise _UNAUTHORIZED

    try:
        return verify(token.strip())
    except Exception as exc:                     # noqa: BLE001
        # Logged at warning, not exception: an expired or malformed token is
        # normal traffic (a stale browser tab), not a server fault. The type
        # name is enough to distinguish expiry from tampering in the log
        # without recording the token itself.
        log.warning("rejected token: %s", type(exc).__name__)
        raise _UNAUTHORIZED from exc
