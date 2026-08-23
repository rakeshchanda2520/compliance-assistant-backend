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

One exception to that, and it matters: a token this service could not
*check* is not the same as a token that failed the check. If Supabase's JWKS
endpoint is briefly unreachable, the token in the browser is perfectly valid
and the caller can do nothing about it — answering 401 there tells them to
sign in again, which cannot help, and they will loop. That case returns 503
instead, which says "us, not you", and which a client can retry.
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

# Not 401. See the module docstring: this is "we could not verify", not "your
# token is bad", and the two need different advice and different client
# behaviour.
_KEYS_UNAVAILABLE = HTTPException(
    status_code=503,
    detail="could not reach the sign-in service to verify your session; "
           "please retry in a moment",
    headers={"Retry-After": "5"},
)

# Small clock skew between Supabase's signing host and this one is normal, and
# without leeway it surfaces as a token that is "expired" a second before it
# is, or "not yet valid" a second after it was issued — both of which read to
# the user as a login that silently did not take.
LEEWAY_SECONDS = 30

# PyJWKClient caches fetched keys and refreshes on an unknown `kid`, so key
# rotation is handled without a restart and without a fetch per request.
# `timeout` is explicit: the default is 30s, and a stalled JWKS fetch holding
# a request thread for half a minute is its own outage.
_jwks_client: jwt.PyJWKClient | None = None


def _jwks() -> jwt.PyJWKClient:
    global _jwks_client
    if _jwks_client is None:
        _jwks_client = jwt.PyJWKClient(
            config.SUPABASE_JWKS_URL, cache_keys=True, lifespan=600, timeout=5)
    return _jwks_client


class KeysUnavailable(Exception):
    """The signing key could not be fetched. Says nothing about the token."""


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
        try:
            key = _jwks().get_signing_key_from_jwt(token).key
        except jwt.exceptions.PyJWKClientConnectionError as exc:
            # Network/DNS/TLS failure reaching Supabase. Nothing to do with
            # this token — see KeysUnavailable and the module docstring.
            raise KeysUnavailable(str(exc)) from exc
        except jwt.exceptions.PyJWKSetError as exc:
            # A JWKS document that parsed but held no usable key. On Supabase
            # this is the signature of a project still on legacy HS256 that is
            # somehow issuing an asymmetric token — a misconfiguration here,
            # not a forged token there.
            raise KeysUnavailable(str(exc)) from exc
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
        leeway=LEEWAY_SECONDS,
        # Signature alone is not enough. Without `aud`/`iss` pinned, a valid
        # token minted by any *other* Supabase project would verify here.
        # PyJWT does not check either by default.
        options={"require": ["exp", "sub", "aud", "iss"]},
    )

    # Supabase's anonymous sign-in issues a token with the same `authenticated`
    # audience as a real one, distinguished only by this claim. Without the
    # check, "answering requires a Google sign-in" would be satisfiable by a
    # single unauthenticated API call — which is the entire gate, gone.
    if claims.get("is_anonymous") is True:
        raise jwt.InvalidTokenError("anonymous sessions may not ask questions")

    metadata = claims.get("user_metadata") or {}
    return Identity(
        sub=claims["sub"],
        # `email` at the top level is the account's email as Supabase holds it;
        # user_metadata's copy is whatever the OAuth provider last sent, which
        # can lag. Top level wins — this is what distinguishes two accounts
        # that share a display name, and the fallback order used to be able to
        # attribute one person's answers to the other's row when their
        # provider metadata had not refreshed.
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
    except KeysUnavailable as exc:
        # Our failure, not theirs. Logged loudly (this is an outage of a
        # dependency, not routine traffic) and reported as 503 so the client
        # retries instead of bouncing the user back through a sign-in that
        # would have worked fine.
        log.error("signing keys unavailable, cannot verify tokens: %s", exc)
        raise _KEYS_UNAVAILABLE from exc
    except Exception as exc:                     # noqa: BLE001
        # Logged at warning, not exception: an expired or malformed token is
        # normal traffic (a stale browser tab), not a server fault. The type
        # name is enough to distinguish expiry from tampering in the log
        # without recording the token itself.
        log.warning("rejected token: %s", type(exc).__name__)
        raise _UNAUTHORIZED from exc
