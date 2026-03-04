from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Dict

from fastapi import HTTPException, status

from .. import db
from ..core.security import pbkdf2_hash_password, pbkdf2_verify_password, sign_token, verify_token as _verify
from ..models import AuthResponse, Role, UserPublic

DEFAULT_ADMIN_LOGIN = "admin"
DEFAULT_ADMIN_PASSWORD = "1234"


def _ensure_default_admin_credentials() -> None:
    users = db.require_col(db.users_col, "users")
    login = DEFAULT_ADMIN_LOGIN.lower().strip()
    now = datetime.now(timezone.utc)

    doc = users.find_one({"email": login})
    if not doc:
        users.insert_one(
            {
                "email": login,
                "password_hash": pbkdf2_hash_password(DEFAULT_ADMIN_PASSWORD),
                "role": "admin",
                "created_at": now,
            }
        )
        return

    updates: dict[str, object] = {}
    if str(doc.get("role", "user")) != "admin":
        updates["role"] = "admin"
    if not pbkdf2_verify_password(DEFAULT_ADMIN_PASSWORD, str(doc.get("password_hash", ""))):
        updates["password_hash"] = pbkdf2_hash_password(DEFAULT_ADMIN_PASSWORD)

    if updates:
        users.update_one({"_id": doc["_id"]}, {"$set": updates})


def bootstrap_admin_if_configured() -> None:
    try:
        _ensure_default_admin_credentials()
    except Exception:
        pass

    email = os.getenv("ADMIN_BOOTSTRAP_EMAIL")
    password = os.getenv("ADMIN_BOOTSTRAP_PASSWORD")
    if not email or not password:
        return
    try:
        create_user(email=email, password=password, role="admin")
    except HTTPException:
        return


def create_user(email: str, password: str, role: Role = "user") -> UserPublic:
    users = db.require_col(db.users_col, "users")
    now = datetime.now(timezone.utc)
    doc = {
        "email": email.lower().strip(),
        "password_hash": pbkdf2_hash_password(password),
        "role": role,
        "created_at": now,
    }
    try:
        res = users.insert_one(doc)
    except Exception:
        raise HTTPException(status_code=400, detail="Email already exists")
    return UserPublic(user_id=str(res.inserted_id), email=doc["email"], role=role, created_at=now)


def login_user(email: str, password: str) -> AuthResponse:
    users = db.require_col(db.users_col, "users")
    login_id = email.lower().strip()
    if "@" not in login_id and login_id != DEFAULT_ADMIN_LOGIN:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

    doc = users.find_one({"email": login_id})
    if not doc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    if not pbkdf2_verify_password(password, doc.get("password_hash", "")):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

    payload = {"uid": str(doc["_id"]), "role": doc.get("role", "user"), "email": doc.get("email", "")}
    token = sign_token(payload)
    return AuthResponse(user_id=payload["uid"], email=payload["email"], role=payload["role"], token=token)


def verify_token(token: str) -> Dict[str, str]:
    payload = _verify(token)
    if not payload:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    uid = str(payload.get("uid", ""))
    role = str(payload.get("role", "user"))
    email = str(payload.get("email", ""))
    if not uid:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    return {"uid": uid, "role": role, "email": email}


def require_admin(token: str) -> Dict[str, str]:
    p = verify_token(token)
    if p.get("role") != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin only")
    return p


def require_roles(payload: Dict[str, str], allowed: tuple[str, ...]) -> Dict[str, str]:
    role = str(payload.get("role", "user"))
    if role not in allowed:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient permissions")
    return payload
