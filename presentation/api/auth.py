"""Supabase JWT verification and a small sliding-window rate limiter."""
from __future__ import annotations

import time
from collections import defaultdict, deque

import jwt
from fastapi import HTTPException


class Verifier:
    def __init__(self, supabase_url: str, jwk_client=None):
        self.issuer = f"{supabase_url}/auth/v1"
        self.jwk_client = jwk_client or jwt.PyJWKClient(f"{self.issuer}/.well-known/jwks.json")

    def user_id(self, token: str) -> str:
        try:
            key = self.jwk_client.get_signing_key_from_jwt(token).key
            claims = jwt.decode(token, key, algorithms=["ES256", "RS256"], audience="authenticated", issuer=self.issuer)
            return claims["sub"]
        except (jwt.PyJWTError, KeyError):
            raise HTTPException(401, "Sign in again.") from None


class RateLimiter:
    def __init__(self, limit: int, window_s: float, clock=time.monotonic):
        self.limit, self.window_s, self.clock = limit, window_s, clock
        self.hits: dict[str, deque] = defaultdict(deque)

    def allow(self, key: str) -> bool:
        now, hits = self.clock(), self.hits[key]
        while hits and hits[0] <= now - self.window_s:
            hits.popleft()
        if len(hits) >= self.limit:
            return False
        hits.append(now)
        return True
