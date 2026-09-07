"""
Who is asking. Google sign-in via Supabase, verified server-side.

Verification is LOCAL, against the signing key. A network round trip per
question would add latency to the one path users wait on and make Supabase's
availability a hard dependency of answering at all. Signature verification
gives the same guarantee without either cost.

Two failures that look alike and must not be treated alike:

    the token is bad          (expired, forged, wrong project)   -> 401
    we could not CHECK it     (JWKS unreachable)                 -> 503

Answering 401 when Supabase's key endpoint blinked tells the user to sign in
again — which cannot possibly help, so they loop forever.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import jwt
from fastapi import Header, HTTPException

import config

log = logging.getLogger(__name__)

# Listing algorithms explicitly is the defence against `alg: none` and the
# HS256-signed-with-the-public-key confusion attack: an attacker cannot talk
# this into a weaker algorithm, because anything else is rejected before a key
# is even chosen.
ASYMMETRIC = ("ES256", "RS256")
SYMMETRIC = ("HS256",)

UNAUTHORIZED = HTTPException(
    status_code=401, detail="sign in to ask a question",
    headers={"WWW-Authenticate": "Bearer"})

KEYS_UNAVAILABLE = HTTPException(
    status_code=503,
    detail="could not reach the sign-in service to verify your session; "
           "please retry in a moment",
    headers={"Retry-After": "5"})

_jwks_client: jwt.PyJWKClient | None = None


class KeysUnavailable(Exception):
    """The signing key could not be fetched. Says nothing about the token."""


def _jwks() -> jwt.PyJWKClient:
    global _jwks_client
    if _jwks_client is None:
        # An explicit timeout: the default is 30s, and a stalled JWKS fetch
        # holding a request thread for half a minute is its own outage.
        _jwks_client = jwt.PyJWKClient(config.SUPABASE_JWKS_URL,
                                       cache_keys=True, lifespan=600, timeout=5)
    return _jwks_client


@dataclass(frozen=True)
class Identity:
    """A verified caller.

    `sub` is Supabase's stable user id — the only field safe to key records on,
    since a person can change their email. `session_id` is Supabase's own
    `auth.sessions.id`, reused rather than minted so every store groups a
    sign-in on the same UUID with no extra state anywhere.
    """
    sub: str
    email: str = ""
    name: str = ""
    session_id: str = ""


def verify(token: str) -> Identity:
    """Decode and validate, or raise. The caller turns that into a status."""
    # Reading the header is not trusting it: it selects which key to verify
    # AGAINST, and `algorithms=` below still constrains what is acceptable.
    algorithm = jwt.get_unverified_header(token).get("alg")

    if algorithm in ASYMMETRIC:
        try:
            key = _jwks().get_signing_key_from_jwt(token).key
        except jwt.exceptions.PyJWKClientConnectionError as exc:
            raise KeysUnavailable(str(exc)) from exc
        except jwt.exceptions.PyJWKSetError as exc:
            # A JWKS document that parsed but held no usable key. On Supabase
            # this is a project still on the legacy secret somehow issuing an
            # asymmetric token — a misconfiguration here, not a forged token.
            raise KeysUnavailable(str(exc)) from exc
        algorithms = list(ASYMMETRIC)
    elif algorithm in SYMMETRIC:
        # Supabase is mid-migration between signing schemes and their docs are
        # explicit that the JWKS endpoint returns NO keys on a project still
        # using the legacy secret. Supporting only one fails 100% of logins on
        # the other, and they do not document which way a new project defaults.
        if not config.SUPABASE_JWT_SECRET:
            raise ValueError(
                "this token is signed with HS256 but SUPABASE_JWT_SECRET is "
                "not set — copy it from Project Settings -> API -> JWT "
                "Settings, or migrate the project to asymmetric signing keys")
        key = config.SUPABASE_JWT_SECRET
        algorithms = list(SYMMETRIC)
    else:
        raise jwt.InvalidAlgorithmError(f"unsupported algorithm {algorithm!r}")

    claims = jwt.decode(
        token, key, algorithms=algorithms,
        audience=config.SUPABASE_AUDIENCE, issuer=config.SUPABASE_ISSUER,
        # Small clock skew is normal; without leeway it surfaces to the user
        # as a login that silently did not take.
        leeway=config.JWT_LEEWAY,
        # Signature alone is not enough. Without `aud`/`iss` pinned, a valid
        # token minted by ANY OTHER Supabase project would verify here. PyJWT
        # checks neither by default.
        options={"require": ["exp", "sub", "aud", "iss"]})

    # Supabase's anonymous sign-in issues a token with the same `authenticated`
    # audience as a real one. Without this check, "answering requires a Google
    # sign-in" is satisfiable by a single unauthenticated API call — the entire
    # gate, gone, with no symptom.
    if claims.get("is_anonymous") is True:
        raise jwt.InvalidTokenError("anonymous sessions may not ask questions")

    metadata = claims.get("user_metadata") or {}
    return Identity(
        sub=claims["sub"],
        # Top-level email wins: user_metadata's copy is whatever the OAuth
        # provider last sent and can lag, and this is the field that
        # distinguishes two accounts sharing a display name.
        email=claims.get("email") or metadata.get("email") or "",
        name=metadata.get("full_name") or metadata.get("name") or "",
        session_id=claims.get("session_id") or "")


async def require_user(authorization: str = Header(default="")) -> Identity:
    """FastAPI dependency for every route that answers questions.

    `/api/health` and `/api/config` deliberately do not use it: an
    infrastructure probe must not need a credential, and the frontend has no
    credential of its own with which to fetch its configuration.
    """
    if not config.REQUIRE_AUTH:
        # Local development against a corpus with no auth provider. Never a
        # default, and reported by /api/health so it cannot pass unnoticed.
        return Identity(sub="dev-local", email="dev@localhost", name="Local dev")

    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise UNAUTHORIZED

    try:
        return verify(token.strip())
    except KeysUnavailable as exc:
        # Our failure, not theirs. Logged at error because it is an outage of
        # a dependency, not routine traffic.
        log.error("signing keys unavailable, cannot verify tokens: %s", exc)
        raise KEYS_UNAVAILABLE from exc
    except Exception as exc:                               # noqa: BLE001
        # Warning, not exception: an expired or malformed token is normal
        # traffic (a stale browser tab), not a server fault. The type name
        # distinguishes expiry from tampering without logging the token.
        log.warning("rejected token: %s", type(exc).__name__)
        raise UNAUTHORIZED from exc
