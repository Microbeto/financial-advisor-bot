"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { useAuth } from "@/lib/auth-context";
import {
  adminClearCache,
  adminDeleteUser,
  adminExportUser,
  adminGetCacheStats,
  adminGetUserDetail,
  adminListUsers,
  adminPurgeUser,
  adminPruneCache,
  adminRefreshNews,
  adminResetUserState,
  adminRunDaily,
  adminSetUserStatus,
  adminUpdatePolicyConstraints,
  adminUpdateRateLimit,
  adminUpdateUserRole,
  getDailySignals,
  getMlRuntimeSettings,
  updateMlRuntimeSettings,
} from "@/lib/api-client";
import type {
  AccountStatus,
  AdminUserDetail,
  MLRuntimeSettings,
  SignalTopItem,
  UserPublic,
} from "@/lib/types";

type Candidate = {
  symbol: string;
  action: "BUY" | "SELL";
  size: number;
  confidence: number;
  rationale: string;
};

export default function AdminPage() {
  const { isSignedIn, role } = useAuth();
  const router = useRouter();

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);

  const [candidates, setCandidates] = useState<Candidate[]>([]);
  const [basket, setBasket] = useState<Set<string>>(new Set());

  const [users, setUsers] = useState<UserPublic[]>([]);
  const [userLoading, setUserLoading] = useState(false);
  const [selectedUserId, setSelectedUserId] = useState<string | null>(null);
  const [userDetail, setUserDetail] = useState<AdminUserDetail | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [statusDraft, setStatusDraft] = useState<AccountStatus>("active");
  const [lockMinutes, setLockMinutes] = useState(30);

  const [mlSettings, setMlSettings] = useState<MLRuntimeSettings | null>(null);
  const [mlDraft, setMlDraft] = useState<MLRuntimeSettings | null>(null);
  const [mlLoading, setMlLoading] = useState(false);
  const [mlSaving, setMlSaving] = useState(false);
  const [mlMessage, setMlMessage] = useState<string | null>(null);
  const [mlError, setMlError] = useState<string | null>(null);

  const [cacheStats, setCacheStats] = useState<Record<string, unknown> | null>(null);
  const [opsLoading, setOpsLoading] = useState(false);
  const [rateLimitKey, setRateLimitKey] = useState("api_read");
  const [rateLimitValue, setRateLimitValue] = useState(120);

  const [policyAdd, setPolicyAdd] = useState("");
  const [policyRemove, setPolicyRemove] = useState("");
  const [policyConstraints, setPolicyConstraints] = useState<string[]>([]);

  // Guard: only admins allowed
  useEffect(() => {
    if (!isSignedIn) {
      router.replace("/");
      return;
    }
    if (role !== "admin") {
      router.replace("/dashboard");
      return;
    }
  }, [isSignedIn, role, router]);

  useEffect(() => {
    if (!isSignedIn || role !== "admin") return;

    let cancelled = false;

    async function load() {
      try {
        setLoading(true);
        setError(null);

        const [signals, userRes, settings, cache] = await Promise.all([
          getDailySignals(undefined, true),
          adminListUsers(),
          getMlRuntimeSettings(),
          adminGetCacheStats(),
        ]);

        if (cancelled) return;

        const toCandidate = (it: SignalTopItem, action: "BUY" | "SELL"): Candidate => ({
          symbol: it.symbol,
          action,
          size: 1,
          confidence: Math.max(0, Math.min(1, Math.abs(Number(it.score || 0)))),
          rationale: String(it.reason || "Signal from ranking engine"),
        });

        const up = (signals.top_up || []).map((it) => toCandidate(it, "BUY"));
        const down = (signals.top_down || []).map((it) => toCandidate(it, "SELL"));
        const merged = [...up, ...down];
        setCandidates(merged);
        setBasket(new Set(up.map((x) => x.symbol)));

        setUsers(userRes.items || []);
        setMlSettings(settings);
        setMlDraft(settings);
        setCacheStats(cache);
      } catch (err) {
        console.error(err);
        if (!cancelled) {
          setError("Failed to load admin control panel.");
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    load();
    return () => {
      cancelled = true;
    };
  }, [isSignedIn, role]);

  useEffect(() => {
    if (!selectedUserId) {
      setUserDetail(null);
      return;
    }

    let cancelled = false;

    async function loadDetail() {
      try {
        setDetailLoading(true);
        const detail = await adminGetUserDetail(selectedUserId);
        if (cancelled) return;
        setUserDetail(detail);
        setStatusDraft(detail.security.status);
      } catch (err) {
        console.error(err);
        if (!cancelled) setError("Failed to load selected user details.");
      } finally {
        if (!cancelled) setDetailLoading(false);
      }
    }

    loadDetail();

    return () => {
      cancelled = true;
    };
  }, [selectedUserId]);

  function toggleInBasket(symbol: string) {
    setBasket((prev) => {
      const next = new Set(prev);
      if (next.has(symbol)) {
        next.delete(symbol);
      } else {
        next.add(symbol);
      }
      return next;
    });
  }

  function updateDraft<K extends keyof MLRuntimeSettings>(
    key: K,
    value: MLRuntimeSettings[K],
  ) {
    setMlDraft((prev) => {
      if (!prev) return prev;
      return { ...prev, [key]: value };
    });
  }

  async function saveMlSettings() {
    if (!mlDraft) return;
    try {
      setMlSaving(true);
      setMlMessage(null);
      setMlError(null);
      const saved = await updateMlRuntimeSettings(mlDraft);
      setMlSettings(saved);
      setMlDraft(saved);
      setMlMessage("ML runtime settings saved.");
    } catch (err) {
      console.error(err);
      setMlError("Failed to save ML runtime settings.");
    } finally {
      setMlSaving(false);
    }
  }

  async function refreshCacheStats() {
    try {
      const data = await adminGetCacheStats();
      setCacheStats(data);
    } catch (err) {
      console.error(err);
    }
  }

  async function changeUserRole(userId: string, newRole: string) {
    try {
      setUserLoading(true);
      await adminUpdateUserRole(userId, newRole);
      const data = await adminListUsers();
      setUsers(data.items || []);
      setMessage("User role updated.");
    } catch (err) {
      console.error(err);
      setError("Failed to update user role.");
    } finally {
      setUserLoading(false);
    }
  }

  async function removeUser(userId: string) {
    try {
      setUserLoading(true);
      await adminDeleteUser(userId);
      const data = await adminListUsers();
      setUsers(data.items || []);
      setMessage("User deleted.");
    } catch (err) {
      console.error(err);
      setError("Failed to delete user.");
    } finally {
      setUserLoading(false);
    }
  }

  async function refreshUserDetail(userId: string) {
    const detail = await adminGetUserDetail(userId);
    setUserDetail(detail);
    setStatusDraft(detail.security.status);
  }

  async function applyUserStatus() {
    if (!selectedUserId) return;
    try {
      setUserLoading(true);
      await adminSetUserStatus(selectedUserId, statusDraft, lockMinutes);
      await refreshUserDetail(selectedUserId);
      setMessage("User account status updated.");
    } catch (err) {
      console.error(err);
      setError("Failed to update user account status.");
    } finally {
      setUserLoading(false);
    }
  }

  async function resetUserState() {
    if (!selectedUserId) return;
    try {
      setUserLoading(true);
      const out = await adminResetUserState(selectedUserId);
      await refreshUserDetail(selectedUserId);
      setMessage(
        `User state reset (signals=${out.cleared_daily_signals}, cache=${out.cleared_dashboard_cache}).`
      );
    } catch (err) {
      console.error(err);
      setError("Failed to reset user state.");
    } finally {
      setUserLoading(false);
    }
  }

  async function exportUserData() {
    if (!selectedUserId) return;
    try {
      setUserLoading(true);
      const payload = await adminExportUser(selectedUserId);
      const blob = new Blob([JSON.stringify(payload, null, 2)], {
        type: "application/json;charset=utf-8",
      });
      const href = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = href;
      anchor.download = `user-export-${selectedUserId}.json`;
      anchor.click();
      URL.revokeObjectURL(href);
      setMessage("User export downloaded.");
    } catch (err) {
      console.error(err);
      setError("Failed to export user data.");
    } finally {
      setUserLoading(false);
    }
  }

  async function purgeUserData() {
    if (!selectedUserId) return;
    const ok = window.confirm(
      "Permanently purge this user and related records? This cannot be undone."
    );
    if (!ok) return;

    try {
      setUserLoading(true);
      await adminPurgeUser(selectedUserId);
      const data = await adminListUsers();
      setUsers(data.items || []);
      setSelectedUserId(null);
      setUserDetail(null);
      setMessage("User and related records purged.");
    } catch (err) {
      console.error(err);
      setError("Failed to purge user records.");
    } finally {
      setUserLoading(false);
    }
  }

  async function runOps(action: "daily" | "news" | "prune" | "clear") {
    try {
      setOpsLoading(true);
      setMessage(null);
      if (action === "daily") {
        await adminRunDaily();
        setMessage("Daily scheduler run completed.");
      } else if (action === "news") {
        const out = await adminRefreshNews();
        setMessage(`News refreshed (${out.count} items).`);
      } else if (action === "prune") {
        await adminPruneCache({ dry_run: false });
        setMessage("Cache prune finished.");
      } else if (action === "clear") {
        await adminClearCache({ scope: "all", dry_run: false });
        setMessage("Cache clear finished.");
      }
      await refreshCacheStats();
    } catch (err) {
      console.error(err);
      setError("Failed to run admin operation.");
    } finally {
      setOpsLoading(false);
    }
  }

  async function saveRateLimit() {
    try {
      setOpsLoading(true);
      await adminUpdateRateLimit(rateLimitKey, rateLimitValue);
      setMessage(`Rate limit updated: ${rateLimitKey}=${rateLimitValue}/min`);
    } catch (err) {
      console.error(err);
      setError("Failed to update rate limit.");
    } finally {
      setOpsLoading(false);
    }
  }

  async function savePolicyConstraints() {
    const add = policyAdd
      .split(",")
      .map((x) => x.trim())
      .filter(Boolean);
    const remove = policyRemove
      .split(",")
      .map((x) => x.trim())
      .filter(Boolean);
    try {
      setOpsLoading(true);
      const out = await adminUpdatePolicyConstraints(add, remove);
      setPolicyConstraints(out.constraints || []);
      setMessage("Policy constraints updated.");
    } catch (err) {
      console.error(err);
      setError("Failed to update policy constraints.");
    } finally {
      setOpsLoading(false);
    }
  }

  function resetMlSettings() {
    if (!mlSettings) return;
    setMlDraft(mlSettings);
    setMlMessage(null);
    setMlError(null);
  }

  const selectedCount = basket.size;

  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-xl font-semibold tracking-tight">Admin Control Panel</h1>
        <p className="max-w-2xl text-sm text-slate-300">
          Central place to manage users, machine-learning runtime configuration,
          policy constraints, and operational site/database actions.
        </p>
      </div>

      {error && (
        <div className="rounded-md border border-red-500/40 bg-red-950/40 px-3 py-2 text-xs text-red-200">
          {error}
        </div>
      )}
      {message && (
        <div className="rounded-md border border-emerald-500/40 bg-emerald-950/40 px-3 py-2 text-xs text-emerald-200">
          {message}
        </div>
      )}

      <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-4 text-xs">
        <div className="mb-3 flex items-center justify-between">
          <div className="text-[11px] text-slate-300">
            Candidates from current policy engine
          </div>
          <div className="text-[11px] text-slate-400">
            Selected in basket:{" "}
            <span className="font-mono text-sky-300">{selectedCount}</span>
          </div>
        </div>

        {loading ? (
          <p className="text-[11px] text-slate-300">Loading…</p>
        ) : candidates.length === 0 ? (
          <p className="text-[11px] text-slate-300">
            No candidates available. Check portfolio and recommendations.
          </p>
        ) : (
          <div className="overflow-x-auto">
            <table className="min-w-full table-auto text-left text-[11px]">
              <thead>
                <tr className="border-b border-slate-700 bg-slate-900/80 text-slate-300">
                  <th className="px-2 py-1">In basket</th>
                  <th className="px-2 py-1">Symbol</th>
                  <th className="px-2 py-1">Action</th>
                  <th className="px-2 py-1">Confidence</th>
                  <th className="px-2 py-1 w-1/2">Rationale</th>
                </tr>
              </thead>
              <tbody>
                {candidates.map((r) => {
                  const checked = basket.has(r.symbol);
                  return (
                    <tr
                      key={r.symbol}
                      className="border-b border-slate-800/80 hover:bg-slate-900"
                    >
                      <td className="px-2 py-1 align-top">
                        <input
                          type="checkbox"
                          checked={checked}
                          onChange={() => toggleInBasket(r.symbol)}
                          aria-label={`Toggle ${r.symbol} in basket`}
                          title={`Toggle ${r.symbol} in basket`}
                          className="h-3 w-3 rounded border-slate-500 bg-slate-900 accent-sky-500"
                        />
                      </td>
                      <td className="px-2 py-1 align-top font-mono text-slate-100">
                        {r.symbol}
                      </td>
                      <td className="px-2 py-1 align-top">
                        <span
                          className={[
                            "inline-flex rounded-full px-2 py-[1px] text-[10px] font-medium",
                            r.action === "BUY"
                              ? "bg-emerald-900/60 text-emerald-200 border border-emerald-500/50"
                              : r.action === "SELL"
                              ? "bg-rose-900/60 text-rose-200 border border-rose-500/50"
                              : "bg-slate-800 text-slate-200 border border-slate-600",
                          ].join(" ")}
                        >
                          {r.action}
                        </span>
                      </td>
                      <td className="px-2 py-1 align-top text-slate-200">
                        {(r.confidence * 100).toFixed(0)}%
                      </td>
                      <td className="px-2 py-1 align-top text-slate-300">
                        {r.rationale}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}

        <div className="mt-3 flex items-center justify-between text-[11px] text-slate-400">
          <p>
            Basket is currently client-side selection of live signal candidates.
          </p>
          <button
            type="button"
              onClick={() => setMessage(`Basket prepared with ${selectedCount} symbols.`)}
            className="rounded-md border border-sky-600 bg-sky-600 px-3 py-1 text-[11px] font-medium text-slate-50 hover:bg-sky-500"
          >
            Save basket
          </button>
        </div>
      </div>

      <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-4 text-xs">
        <div className="mb-3 flex items-center justify-between">
          <div className="text-[11px] text-slate-300">User Management</div>
          <div className="text-[11px] text-slate-400">Users: {users.length}</div>
        </div>
        {userLoading ? (
          <p className="text-[11px] text-slate-300">Updating users…</p>
        ) : users.length === 0 ? (
          <p className="text-[11px] text-slate-300">No users found.</p>
        ) : (
          <div className="overflow-x-auto">
            <table className="min-w-full table-auto text-left text-[11px]">
              <thead>
                <tr className="border-b border-slate-700 bg-slate-900/80 text-slate-300">
                  <th className="px-2 py-1">User ID</th>
                  <th className="px-2 py-1">Email</th>
                  <th className="px-2 py-1">Role</th>
                  <th className="px-2 py-1">Actions</th>
                </tr>
              </thead>
              <tbody>
                {users.map((u) => (
                  <tr
                    key={u.user_id}
                    className={[
                      "border-b border-slate-800/80 hover:bg-slate-900",
                      selectedUserId === u.user_id ? "bg-sky-950/20" : "",
                    ].join(" ")}
                  >
                    <td className="px-2 py-1 font-mono text-slate-100">{u.user_id}</td>
                    <td className="px-2 py-1 text-slate-300">{u.email}</td>
                    <td className="px-2 py-1">
                      <select
                        value={String(u.role || "user")}
                        onChange={(e) => changeUserRole(u.user_id, e.target.value)}
                        className="rounded border border-slate-700 bg-slate-950 px-2 py-1 text-[11px] text-slate-100"
                        title="Change role"
                      >
                        <option value="user">user</option>
                        <option value="premium">premium</option>
                        <option value="manager">manager</option>
                        <option value="admin">admin</option>
                      </select>
                    </td>
                    <td className="px-2 py-1">
                      <button
                        type="button"
                        onClick={() => setSelectedUserId(u.user_id)}
                        className="mr-2 rounded border border-sky-700 bg-sky-900/30 px-2 py-1 text-[11px] text-sky-200 hover:bg-sky-900/50"
                      >
                        View
                      </button>
                      <button
                        type="button"
                        onClick={() => removeUser(u.user_id)}
                        className="rounded border border-rose-700 bg-rose-900/40 px-2 py-1 text-[11px] text-rose-200 hover:bg-rose-900/60"
                      >
                        Delete
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        <div className="mt-4 rounded-lg border border-slate-800 bg-slate-950/60 p-3">
          <div className="mb-2 flex items-center justify-between">
            <div className="text-[11px] text-slate-300">Selected User Account</div>
            <div className="text-[11px] text-slate-400">
              {selectedUserId ? `User: ${selectedUserId}` : "No user selected"}
            </div>
          </div>

          {detailLoading ? (
            <p className="text-[11px] text-slate-300">Loading user detail…</p>
          ) : !userDetail ? (
            <p className="text-[11px] text-slate-400">Choose a user from the table to inspect account detail.</p>
          ) : (
            <div className="space-y-3">
              <div className="grid gap-2 md:grid-cols-2">
                <div className="rounded border border-slate-800 bg-slate-950/60 p-2 text-[11px] text-slate-300">
                  <div className="mb-1 text-slate-200">Identity & Security</div>
                  <div>Email: {userDetail.email}</div>
                  <div>Role: {String(userDetail.role || "user")}</div>
                  <div>Status: {userDetail.security.status}</div>
                  <div>Failed logins: {userDetail.security.failed_login_attempts}</div>
                  <div>Locked until: {userDetail.security.locked_until || "-"}</div>
                </div>
                <div className="rounded border border-slate-800 bg-slate-950/60 p-2 text-[11px] text-slate-300">
                  <div className="mb-1 text-slate-200">Risk & Constraints</div>
                  <div>Risk tolerance: {userDetail.financial_context.risk_tolerance}</div>
                  <div>Horizon years: {String(userDetail.financial_context.horizon_years ?? "-")}</div>
                  <div>Max drawdown: {String(userDetail.financial_context.max_drawdown_pct ?? "-")}</div>
                  <div>
                    Constraints: {userDetail.financial_context.constraints.length ? userDetail.financial_context.constraints.join(", ") : "-"}
                  </div>
                  <div>
                    Universe: {userDetail.financial_context.custom_universe.length ? userDetail.financial_context.custom_universe.join(", ") : "-"}
                  </div>
                </div>
              </div>

              <div className="grid gap-2 md:grid-cols-2">
                <div className="rounded border border-slate-800 bg-slate-950/60 p-2 text-[11px] text-slate-300">
                  <div className="mb-1 text-slate-200">Usage Activity</div>
                  <div>Created at: {userDetail.created_at || "-"}</div>
                  <div>Last login: {userDetail.activity.last_login_at || "-"}</div>
                  <div>Last dashboard: {userDetail.activity.last_dashboard_at || "-"}</div>
                </div>
                <div className="rounded border border-slate-800 bg-slate-950/60 p-2 text-[11px] text-slate-300">
                  <div className="mb-1 text-slate-200">Rate Limit</div>
                  <div>Currently limited: {String(Boolean(userDetail.rate_limit.currently_limited))}</div>
                  <pre className="mt-1 max-h-28 overflow-auto rounded border border-slate-800 bg-slate-950 p-2 text-[10px] text-slate-300">
                    {JSON.stringify(userDetail.rate_limit.buckets || {}, null, 2)}
                  </pre>
                </div>
              </div>

              <div className="rounded border border-slate-800 bg-slate-950/60 p-2 text-[11px] text-slate-300">
                <div className="mb-1 text-slate-200">Recent Errors</div>
                {userDetail.error_logs.length === 0 ? (
                  <div className="text-slate-400">No recent user-linked errors.</div>
                ) : (
                  <div className="max-h-40 overflow-auto space-y-1">
                    {userDetail.error_logs.map((item, idx) => (
                      <div key={`${item.timestamp || "ts"}-${idx}`} className="rounded border border-slate-800 bg-slate-950 px-2 py-1">
                        <div className="text-slate-200">{item.timestamp || "-"}</div>
                        <div>{item.method} {item.path}</div>
                        <div className="text-slate-400">{item.message}</div>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              <div className="rounded border border-slate-800 bg-slate-950/60 p-2 text-[11px]">
                <div className="mb-2 text-slate-200">Admin Actions</div>
                <div className="grid gap-2 md:grid-cols-4">
                  <select
                    value={statusDraft}
                    onChange={(e) => setStatusDraft(e.target.value as AccountStatus)}
                    className="rounded border border-slate-700 bg-slate-950 px-2 py-1 text-[11px] text-slate-100"
                    title="User status"
                  >
                    <option value="active">active</option>
                    <option value="suspended">suspended</option>
                    <option value="locked">locked</option>
                  </select>
                  <input
                    type="number"
                    min={1}
                    value={lockMinutes}
                    onChange={(e) => setLockMinutes(Number(e.target.value) || 1)}
                    className="rounded border border-slate-700 bg-slate-950 px-2 py-1 text-[11px] text-slate-100"
                    placeholder="Lock minutes"
                    title="Lock minutes"
                  />
                  <button
                    type="button"
                    onClick={applyUserStatus}
                    disabled={userLoading}
                    className="rounded border border-sky-700 bg-sky-900/30 px-2 py-1 text-[11px] text-sky-200 hover:bg-sky-900/50 disabled:opacity-60"
                  >
                    Set status
                  </button>
                  <button
                    type="button"
                    onClick={resetUserState}
                    disabled={userLoading}
                    className="rounded border border-slate-700 bg-slate-900 px-2 py-1 text-[11px] text-slate-200 hover:bg-slate-800 disabled:opacity-60"
                  >
                    Reset state
                  </button>
                </div>

                <div className="mt-2 flex flex-wrap gap-2">
                  <button
                    type="button"
                    onClick={exportUserData}
                    disabled={userLoading}
                    className="rounded border border-emerald-700 bg-emerald-900/30 px-2 py-1 text-[11px] text-emerald-200 hover:bg-emerald-900/50 disabled:opacity-60"
                  >
                    Export user data
                  </button>
                  <button
                    type="button"
                    onClick={purgeUserData}
                    disabled={userLoading}
                    className="rounded border border-rose-700 bg-rose-900/40 px-2 py-1 text-[11px] text-rose-200 hover:bg-rose-900/60 disabled:opacity-60"
                  >
                    Purge user records
                  </button>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-4 text-xs">
        <div className="mb-3 flex items-center justify-between">
          <div className="text-[11px] text-slate-300">ML Pipeline Settings</div>
          <div className="text-[11px] text-slate-400">
            Runtime tuning for labeling and walk-forward validation
          </div>
        </div>

        {mlLoading ? (
          <p className="text-[11px] text-slate-300">Loading settings…</p>
        ) : !mlDraft ? (
          <p className="text-[11px] text-slate-300">No settings available.</p>
        ) : (
          <div className="grid gap-3 md:grid-cols-2">
            <label className="space-y-1">
              <span className="text-[11px] text-slate-300">Take-profit barrier</span>
              <input
                type="number"
                min={0.001}
                step={0.001}
                value={mlDraft.tp_barrier}
                onChange={(e) => updateDraft("tp_barrier", Number(e.target.value))}
                className="w-full rounded-md border border-slate-700 bg-slate-950 px-2 py-1 text-[11px] text-slate-100"
              />
            </label>

            <label className="space-y-1">
              <span className="text-[11px] text-slate-300">Stop-loss barrier</span>
              <input
                type="number"
                min={0.001}
                step={0.001}
                value={mlDraft.sl_barrier}
                onChange={(e) => updateDraft("sl_barrier", Number(e.target.value))}
                className="w-full rounded-md border border-slate-700 bg-slate-950 px-2 py-1 text-[11px] text-slate-100"
              />
            </label>

            <label className="space-y-1">
              <span className="text-[11px] text-slate-300">Time barrier (days)</span>
              <input
                type="number"
                min={1}
                step={1}
                value={mlDraft.time_barrier_days}
                onChange={(e) => updateDraft("time_barrier_days", Number(e.target.value))}
                className="w-full rounded-md border border-slate-700 bg-slate-950 px-2 py-1 text-[11px] text-slate-100"
              />
            </label>

            <label className="space-y-1">
              <span className="text-[11px] text-slate-300">Walk-forward splits</span>
              <input
                type="number"
                min={2}
                step={1}
                value={mlDraft.walk_forward_splits}
                onChange={(e) => updateDraft("walk_forward_splits", Number(e.target.value))}
                className="w-full rounded-md border border-slate-700 bg-slate-950 px-2 py-1 text-[11px] text-slate-100"
              />
            </label>

            <label className="space-y-1 md:col-span-2">
              <span className="text-[11px] text-slate-300">Walk-forward minimum training rows</span>
              <input
                type="number"
                min={40}
                step={1}
                value={mlDraft.walk_forward_min_train}
                onChange={(e) => updateDraft("walk_forward_min_train", Number(e.target.value))}
                className="w-full rounded-md border border-slate-700 bg-slate-950 px-2 py-1 text-[11px] text-slate-100"
              />
            </label>
          </div>
        )}

        {mlError && (
          <div className="mt-3 rounded-md border border-red-500/40 bg-red-950/40 px-3 py-2 text-[11px] text-red-200">
            {mlError}
          </div>
        )}
        {mlMessage && (
          <div className="mt-3 rounded-md border border-emerald-500/40 bg-emerald-950/40 px-3 py-2 text-[11px] text-emerald-200">
            {mlMessage}
          </div>
        )}

        <div className="mt-3 flex items-center justify-end gap-2">
          <button
            type="button"
            onClick={resetMlSettings}
            disabled={mlLoading || mlSaving || !mlSettings}
            className="rounded-md border border-slate-600 bg-slate-800 px-3 py-1 text-[11px] font-medium text-slate-100 disabled:opacity-50"
          >
            Reset
          </button>
          <button
            type="button"
            onClick={saveMlSettings}
            disabled={mlLoading || mlSaving || !mlDraft}
            className="rounded-md border border-sky-600 bg-sky-600 px-3 py-1 text-[11px] font-medium text-slate-50 hover:bg-sky-500 disabled:opacity-50"
          >
            {mlSaving ? "Saving…" : "Save ML settings"}
          </button>
        </div>
      </div>

      <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-4 text-xs space-y-4">
        <div className="flex items-center justify-between">
          <div className="text-[11px] text-slate-300">Policy Constraints</div>
          <div className="text-[11px] text-slate-400">Global rule controls</div>
        </div>
        <div className="grid gap-3 md:grid-cols-2">
          <label className="space-y-1">
            <span className="text-[11px] text-slate-300">Add constraints (comma-separated)</span>
            <input
              type="text"
              value={policyAdd}
              onChange={(e) => setPolicyAdd(e.target.value)}
              className="w-full rounded-md border border-slate-700 bg-slate-950 px-2 py-1 text-[11px] text-slate-100"
              placeholder="no_penny_stocks,max_sector_30"
            />
          </label>
          <label className="space-y-1">
            <span className="text-[11px] text-slate-300">Remove constraints (comma-separated)</span>
            <input
              type="text"
              value={policyRemove}
              onChange={(e) => setPolicyRemove(e.target.value)}
              className="w-full rounded-md border border-slate-700 bg-slate-950 px-2 py-1 text-[11px] text-slate-100"
              placeholder="old_constraint"
            />
          </label>
        </div>
        <div className="flex items-center justify-between">
          <div className="text-[11px] text-slate-400">
            Active constraints: {policyConstraints.length ? policyConstraints.join(", ") : "(load/update to view)"}
          </div>
          <button
            type="button"
            onClick={savePolicyConstraints}
            disabled={opsLoading}
            className="rounded-md border border-sky-600 bg-sky-600 px-3 py-1 text-[11px] font-medium text-slate-50 hover:bg-sky-500 disabled:opacity-60"
          >
            Save policy constraints
          </button>
        </div>
      </div>

      <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-4 text-xs space-y-4">
        <div className="flex items-center justify-between">
          <div className="text-[11px] text-slate-300">Site & Database Operations</div>
          <div className="text-[11px] text-slate-400">Cache, scheduler, news, rate limits</div>
        </div>

        <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-4">
          <button type="button" onClick={() => runOps("daily")} disabled={opsLoading} className="rounded-md border border-slate-600 bg-slate-800 px-3 py-1 text-[11px] text-slate-100 hover:bg-slate-700 disabled:opacity-60">Run daily job</button>
          <button type="button" onClick={() => runOps("news")} disabled={opsLoading} className="rounded-md border border-slate-600 bg-slate-800 px-3 py-1 text-[11px] text-slate-100 hover:bg-slate-700 disabled:opacity-60">Refresh news</button>
          <button type="button" onClick={() => runOps("prune")} disabled={opsLoading} className="rounded-md border border-slate-600 bg-slate-800 px-3 py-1 text-[11px] text-slate-100 hover:bg-slate-700 disabled:opacity-60">Prune cache</button>
          <button type="button" onClick={() => runOps("clear")} disabled={opsLoading} className="rounded-md border border-rose-700 bg-rose-900/40 px-3 py-1 text-[11px] text-rose-200 hover:bg-rose-900/60 disabled:opacity-60">Clear cache</button>
        </div>

        <div className="rounded-lg border border-slate-800 bg-slate-950/60 p-3">
          <div className="mb-2 text-[11px] text-slate-300">Rate Limit Update</div>
          <div className="grid gap-2 md:grid-cols-3">
            <input
              value={rateLimitKey}
              onChange={(e) => setRateLimitKey(e.target.value)}
              className="rounded border border-slate-700 bg-slate-950 px-2 py-1 text-[11px] text-slate-100"
              placeholder="api_read"
              title="Rate-limit key"
              aria-label="Rate-limit key"
            />
            <input
              type="number"
              value={rateLimitValue}
              onChange={(e) => setRateLimitValue(Number(e.target.value))}
              className="rounded border border-slate-700 bg-slate-950 px-2 py-1 text-[11px] text-slate-100"
              min={1}
              placeholder="120"
              title="Per-minute limit"
              aria-label="Per-minute limit"
            />
            <button type="button" onClick={saveRateLimit} disabled={opsLoading} className="rounded-md border border-sky-600 bg-sky-600 px-3 py-1 text-[11px] font-medium text-slate-50 hover:bg-sky-500 disabled:opacity-60">Save rate limit</button>
          </div>
        </div>

        <div className="rounded-lg border border-slate-800 bg-slate-950/60 p-3">
          <div className="mb-2 text-[11px] text-slate-300">Cache stats</div>
          <pre className="overflow-x-auto text-[11px] text-slate-300">{JSON.stringify(cacheStats || {}, null, 2)}</pre>
        </div>
      </div>
    </div>
  );
}
