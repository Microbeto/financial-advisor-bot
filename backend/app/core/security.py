from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import os
from typing import Any, Dict, Optional

import jwt

# Reads the application secret used for token signing.
_SECRET = os.getenv("APP_SECRET", "dev-secret-change-me")
# Creates a module logger for security-related warnings.
_logger = logging.getLogger(__name__)


def _signing_key(secret: str) -> bytes:
    """
    HS256 recommends >= 32-byte key material.
    If APP_SECRET is shorter, derive a fixed 32-byte key via SHA-256.
    """
    # Encodes the configured secret into raw bytes.
    raw = (secret or "").encode("utf-8")
    # Uses the secret directly when it already has enough entropy length.
    if len(raw) >= 32:
        return raw

    # Warns and derives a stable 32-byte fallback key for undersized secrets.
    _logger.warning(
        "APP_SECRET is shorter than 32 bytes; deriving 32-byte signing key via SHA-256. "
        "Set APP_SECRET to a strong 32+ byte secret in production."
    )
    return hashlib.sha256(raw).digest()


# Initializes the module-level signing key once at import time.
_SIGNING_KEY = _signing_key(_SECRET)


def pbkdf2_hash_password(password: str, salt: Optional[bytes] = None) -> str:
    # Generates a random salt when one is not provided.
    salt = salt or os.urandom(16)
    # Derives a password hash using PBKDF2-HMAC-SHA256 and 200k iterations.
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 200_000)
    # Returns a portable string containing scheme, salt, and derived key.
    return (
        "pbkdf2$"
        + base64.b64encode(salt).decode()
        + "$"
        + base64.b64encode(dk).decode()
    )


def pbkdf2_verify_password(password: str, stored: str) -> bool:
    # Parses stored hash components and validates expected hashing scheme.
    try:
        scheme, b64salt, b64dk = stored.split("$", 2)
        if scheme != "pbkdf2":
            return False
        # Decodes stored salt/hash and recomputes hash for comparison.
        salt = base64.b64decode(b64salt.encode())
        dk = base64.b64decode(b64dk.encode())
        test = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 200_000)
        # Uses constant-time comparison to avoid timing attacks.
        return hmac.compare_digest(test, dk)
    except Exception:
        # Treats malformed or unexpected stored values as verification failure.
        return False


def sign_token(payload: Dict[str, Any]) -> str:
    # Signs a JWT payload with HS256 using the derived signing key.
    token = jwt.encode(payload, _SIGNING_KEY, algorithm="HS256")
    # PyJWT may return bytes in older versions; normalize to str.
    if isinstance(token, bytes):
        return token.decode("utf-8")
    return str(token)


def verify_token(token: str) -> Optional[Dict[str, Any]]:
    # Verifies JWT signature/claims and returns payload only when it is a dict.
    try:
        payload = jwt.decode(token, _SIGNING_KEY, algorithms=["HS256"])
        if isinstance(payload, dict):
            return payload
        return None
    except Exception:
        # Fails closed by returning None for invalid or expired tokens.
        return None
