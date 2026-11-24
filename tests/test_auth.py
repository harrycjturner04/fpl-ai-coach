import time
from types import SimpleNamespace

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import ec
from fastapi import HTTPException

from presentation.api.auth import RateLimiter, Verifier

URL = "https://proj.supabase.co"
KEY = ec.generate_private_key(ec.SECP256R1())


def token(key=KEY, **claims):
    body = {"sub": "u1", "aud": "authenticated", "iss": f"{URL}/auth/v1", "exp": time.time() + 600} | claims
    return jwt.encode(body, key, algorithm="ES256", headers={"kid": "k1"})


def verifier(key=KEY):
    class Jwks:
        def get_signing_key_from_jwt(self, _):
            return SimpleNamespace(key=key.public_key())
    return Verifier(URL, jwk_client=Jwks())


def test_valid_token_gives_sub():
    assert verifier().user_id(token()) == "u1"


@pytest.mark.parametrize("claims", [{"exp": time.time() - 10}, {"aud": "other"}, {"iss": "https://evil.example/auth/v1"}])
def test_bad_claims_401(claims):
    with pytest.raises(HTTPException) as e:
        verifier().user_id(token(**claims))
    assert e.value.status_code == 401 and e.value.detail == "Sign in again."


def test_other_key_401():
    with pytest.raises(HTTPException) as e:
        verifier().user_id(token(key=ec.generate_private_key(ec.SECP256R1())))
    assert e.value.status_code == 401


def test_garbage_token_401():
    with pytest.raises(HTTPException) as e:
        verifier().user_id("not.a.jwt")
    assert e.value.status_code == 401


def test_rate_limiter_window():
    now = [0.0]
    lim = RateLimiter(3, 60, clock=lambda: now[0])
    assert [lim.allow("a") for _ in range(4)] == [True, True, True, False]
    assert lim.allow("b")
    now[0] = 61
    assert lim.allow("a")
