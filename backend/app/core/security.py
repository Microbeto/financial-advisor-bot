from __future__ import annotations

import base64
import hashlib
import hmac
import os
from typing import Any, Dict, Optional

_SECRET = os.getenv("APP_SECRET", "dev-secret-change-me")


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
    msg = repr(sorted(payload.items())).encode("utf-8")
    sig = hmac.new(_SECRET.encode("utf-8"), msg, hashlib.sha256).digest()
    return (
        base64.urlsafe_b64encode(msg).decode().rstrip("=")
        + "."
        + base64.urlsafe_b64encode(sig).decode().rstrip("=")
    )


def verify_token(token: str) -> Optional[Dict[str, Any]]:
    try:
        msg_b64, sig_b64 = token.split(".", 1)
        msg = base64.urlsafe_b64decode(msg_b64 + "==")
        sig = base64.urlsafe_b64decode(sig_b64 + "==")
        expected = hmac.new(_SECRET.encode("utf-8"), msg, hashlib.sha256).digest()
        if not hmac.compare_digest(sig, expected):
            return None
        s = msg.decode("utf-8")
        items = eval(s, {"__builtins__": {}})
        return dict(items)
    except Exception:
        return None
