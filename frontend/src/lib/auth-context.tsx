// frontend/src/lib/auth-context.tsx
"use client";

import {
  createContext,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import type { UserPublic, UserCreate, UserLogin, UserRole } from "@/lib/types";
import * as api from "@/lib/api-client";

type Role = "guest" | UserRole;

type AuthContextValue = {
  role: Role;
  isSignedIn: boolean;
  user: UserPublic | null;

  // Local testing (no backend)
  signInUser: () => void;
  signInAdmin: () => void;

  // Real auth (backend)
  register: (payload: UserCreate) => Promise<UserPublic>;
  login: (payload: UserLogin) => Promise<UserPublic>;

  signOut: () => void;
};

const AuthContext = createContext<AuthContextValue | undefined>(undefined);

const ROLE_KEY = "advisor_role";
const USER_KEY = "advisor_user";
const TOKEN_KEY = "advisor_token";

function roleFromUser(u: UserPublic | null): Role {
  if (!u) return "guest";
  const r = (u.role || "user").toString().toLowerCase();
  if (r === "admin" || r === "premium" || r === "manager" || r === "user") {
    return r as Role;
  }
  return "user";
}

function safeParseUser(raw: string | null): UserPublic | null {
  if (!raw) return null;
  try {
    const u = JSON.parse(raw) as UserPublic;
    if (!u || typeof u !== "object") return null;

    const userId = typeof (u as any).user_id === "string" ? (u as any).user_id : undefined;
    const legacyId = typeof (u as any).id === "string" ? (u as any).id : undefined;
    if (!userId && !legacyId) return null;

    const normalized: UserPublic = {
      ...u,
      user_id: userId || legacyId || "",
      role: (u.role || "user") as string,
    };

    return normalized;
  } catch {
    return null;
  }
}

function loadStored(): { role: Role; user: UserPublic | null; token: string | null } {
  if (typeof window === "undefined") return { role: "guest", user: null, token: null };

  let role: Role = "guest";
  let user: UserPublic | null = null;
  let token: string | null = null;

  try {
    const r = window.localStorage.getItem(ROLE_KEY);
    if (r === "user" || r === "premium" || r === "admin" || r === "manager") {
      role = r;
    }
  } catch {}

  try {
    token = window.localStorage.getItem(TOKEN_KEY);
  } catch {}

  try {
    const raw = window.localStorage.getItem(USER_KEY);
    user = safeParseUser(raw);
  } catch {}

  // If a user exists, derive role from user data.
  if (user) role = roleFromUser(user);

  return { role, user, token };
}

function persist(role: Role, user: UserPublic | null, token: string | null) {
  if (typeof window === "undefined") return;

  try {
    window.localStorage.setItem(ROLE_KEY, role);
  } catch {}

  try {
    if (user) window.localStorage.setItem(USER_KEY, JSON.stringify(user));
    else window.localStorage.removeItem(USER_KEY);
  } catch {}

  try {
    if (token) window.localStorage.setItem(TOKEN_KEY, token);
    else window.localStorage.removeItem(TOKEN_KEY);
  } catch {}
}

function randomId(): string {
  try {
    if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
      return crypto.randomUUID();
    }
  } catch {}

  return `local_${Date.now()}_${Math.random().toString(16).slice(2)}`;
}

function makeLocalUser(role: Role): UserPublic {
  const user_id = randomId();

  return {
    user_id,
    email: role === "admin" ? "admin@local" : "user@local",
    role,
  };
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const [{ role, user, token }, setState] = useState(() => ({
    role: "guest" as Role,
    user: null as UserPublic | null,
    token: null as string | null,
  }));

  useEffect(() => {
    setState(loadStored());
  }, []);

  useEffect(() => {
    persist(role, user, token);
  }, [role, user, token]);

  const isSignedIn = role !== "guest";

  function signInUser() {
    const u = makeLocalUser("user");
    setState({ role: roleFromUser(u), user: u, token: `local-token-${u.user_id}` });
  }

  function signInAdmin() {
    const u = makeLocalUser("admin");
    setState({ role: roleFromUser(u), user: u, token: `local-token-${u.user_id}` });
  }

  async function hydrateUserFromToken() {
    if (!token) return;
    if (user) return;
    try {
      const me = await api.me();
      setState((prev) => ({ role: roleFromUser(me), user: me, token: prev.token }));
    } catch {
      setState({ role: "guest", user: null, token: null });
    }
  }

  useEffect(() => {
    hydrateUserFromToken();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [token]);

  async function register(payload: UserCreate) {
    // Your backend may not implement /auth/register yet.
    // If it fails, fall back to a local user so the app remains usable.
    try {
      const auth = await api.register(payload);
      const u: UserPublic = {
        user_id: auth.user_id,
        email: auth.email,
        role: auth.role,
      };
      setState({ role: roleFromUser(u), user: u, token: auth.token });
      return u;
    } catch {
      const u: UserPublic = {
        user_id: randomId(),
        email: payload.email,
        role: "user",
      } as UserPublic;
      setState({ role: roleFromUser(u), user: u, token: `local-token-${u.user_id}` });
      return u;
    }
  }

  async function login(payload: UserLogin) {
    // If backend login fails, do not silently sign in.
    const auth = await api.login(payload);
    const u: UserPublic = {
      user_id: auth.user_id,
      email: auth.email,
      role: auth.role,
    };
    setState({ role: roleFromUser(u), user: u, token: auth.token });
    return u;
  }

  function signOut() {
    setState({ role: "guest", user: null, token: null });
  }

  const value = useMemo<AuthContextValue>(
    () => ({
      role,
      isSignedIn,
      user,
      signInUser,
      signInAdmin,
      register,
      login,
      signOut,
    }),
    [role, isSignedIn, user]
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within AuthProvider");
  return ctx;
}
