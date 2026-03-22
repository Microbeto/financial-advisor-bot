# Structure: user service exposing role/public-profile lookups and user-list retrieval helpers.
from __future__ import annotations

from typing import List, Optional

from .. import db
from ..models import Role, UserPublic


# get_user_public service logic.
def get_user_public(user_id: str) -> Optional[UserPublic]:
    users = db.require_col(db.users_col, "users")
    oid = _obj_id(user_id)
    if oid is None:
        return None

    doc = users.find_one({"_id": oid})
    if not doc:
        return None

    return UserPublic(
        user_id=str(doc["_id"]),
        email=str(doc.get("email", "")),
        role=_normalize_role(doc.get("role")),
        created_at=doc.get("created_at"),
    )


# get_user_role service logic.
def get_user_role(user_id: str) -> Role:
    p = get_user_public(user_id)
    if not p:
        return "user"
    return _normalize_role(p.role)


# list_users service logic.
def list_users(limit: int = 200) -> List[UserPublic]:
    users = db.require_col(db.users_col, "users")
    docs = list(users.find({}).sort([("created_at", -1)]).limit(max(1, int(limit))))
    out: List[UserPublic] = []
    for d in docs:
        out.append(
            UserPublic(
                user_id=str(d.get("_id")),
                email=str(d.get("email", "")),
                role=_normalize_role(d.get("role")),
                created_at=d.get("created_at"),
            )
        )
    return out


# _normalize_role service logic.
def _normalize_role(value: object) -> Role:
    role = str(value or "").strip().lower()
    if role == "premium":
        return "premium"
    if role == "admin":
        return "admin"
    if role == "manager":
        return "manager"
    return "user"


# update_user_role service logic.
def update_user_role(user_id: str, role: Role) -> Optional[UserPublic]:
    users = db.require_col(db.users_col, "users")
    oid = _obj_id(user_id)
    if oid is None:
        return None

    users.update_one({"_id": oid}, {"$set": {"role": role}})
    return get_user_public(user_id)


# delete_user service logic.
def delete_user(user_id: str) -> bool:
    users = db.require_col(db.users_col, "users")
    oid = _obj_id(user_id)
    if oid is None:
        return False

    res = users.delete_one({"_id": oid})
    return bool(res.deleted_count > 0)


# _obj_id service logic.
def _obj_id(s: str):
    try:
        from bson import ObjectId
        return ObjectId(s)
    except Exception:
        return None
