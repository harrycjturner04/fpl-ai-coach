import base64
import hmac
import json
import time
from types import SimpleNamespace

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
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


def test_token_without_exp_401():
    body = {"sub": "u1", "aud": "authenticated", "iss": f"{URL}/auth/v1"}
    with pytest.raises(HTTPException) as e:
        verifier().user_id(jwt.encode(body, KEY, algorithm="ES256", headers={"kid": "k1"}))
    assert e.value.status_code == 401


def b64(d):
    return base64.urlsafe_b64encode(json.dumps(d).encode()).rstrip(b"=").decode()


def test_hs256_signed_with_public_key_401():
    """Algorithm confusion: PyJWT refuses to sign this itself, so forge the HMAC by hand."""
    pem = KEY.public_key().public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
    body = {"sub": "u1", "aud": "authenticated", "iss": f"{URL}/auth/v1", "exp": time.time() + 600}
    signing = f"{b64({'alg': 'HS256', 'typ': 'JWT', 'kid': 'k1'})}.{b64(body)}"
    sig = base64.urlsafe_b64encode(hmac.new(pem, signing.encode(), "sha256").digest()).rstrip(b"=").decode()
    with pytest.raises(HTTPException) as e:
        verifier().user_id(f"{signing}.{sig}")
    assert e.value.status_code == 401


def test_alg_none_401():
    forged = f"{b64({'alg': 'none', 'typ': 'JWT'})}.{b64({'sub': 'u1', 'aud': 'authenticated', 'iss': f'{URL}/auth/v1', 'exp': time.time() + 600})}."
    with pytest.raises(HTTPException) as e:
        verifier().user_id(forged)
    assert e.value.status_code == 401


def test_jwks_unreachable_401():
    class Down:
        def get_signing_key_from_jwt(self, _):
            raise jwt.PyJWKClientConnectionError("down")
    with pytest.raises(HTTPException) as e:
        Verifier(URL, jwk_client=Down()).user_id(token())
    assert e.value.status_code == 401


def test_trailing_slash_url_keeps_issuer():
    class Jwks:
        def get_signing_key_from_jwt(self, _):
            return SimpleNamespace(key=KEY.public_key())
    assert Verifier(URL + "/", jwk_client=Jwks()).user_id(token()) == "u1"
