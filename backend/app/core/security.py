from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import os
from typing import Any, Dict, Optional

import jwt

_SECRET = os.getenv("APP_SECRET", "dev-secret-change-me")
_logger = logging.getLogger(__name__)


def _signing_key(secret: str) -> bytes:
    """
    HS256 recommends >= 32-byte key material.
    If APP_SECRET is shorter, derive a fixed 32-byte key via SHA-256.
    """
    raw = (secret or "").encode("utf-8")
    if len(raw) >= 32:
        return raw

    _logger.warning(
        "APP_SECRET is shorter than 32 bytes; deriving 32-byte signing key via SHA-256. "
        "Set APP_SECRET to a strong 32+ byte secret in production."
    )
    return hashlib.sha256(raw).digest()


_SIGNING_KEY = _signing_key(_SECRET)


def pbkdf2_hash_password(password: str, salt: Optional[bytes] = None) -> str:
    salt = salt or os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 200_000)
    return (
        "pbkdf2$"
        + base64.b64encode(salt).decode()
        + "$"
        + base64.b64encode(dk).decode()
    )


def pbkdf2_verify_password(password: str, stored: str) -> bool:
    try:
        scheme, b64salt, b64dk = stored.split("$", 2)
        if scheme != "pbkdf2":
            return False
        salt = base64.b64decode(b64salt.encode())
        dk = base64.b64decode(b64dk.encode())
        test = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 200_000)
        return hmac.compare_digest(test, dk)
    except Exception:
        return False


def sign_token(payload: Dict[str, Any]) -> str:
    token = jwt.encode(payload, _SIGNING_KEY, algorithm="HS256")
    # PyJWT may return bytes in older versions; normalize to str.
    if isinstance(token, bytes):
        return token.decode("utf-8")
    return str(token)


def verify_token(token: str) -> Optional[Dict[str, Any]]:
    try:
        payload = jwt.decode(token, _SIGNING_KEY, algorithms=["HS256"])
        if isinstance(payload, dict):
            return payload
        return None
    except Exception:
        return None
