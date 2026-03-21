// User settings and preferences page for managing notifications and display options.
"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { useAuth } from "@/lib/auth-context";

export default function SettingsPage() {
  const router = useRouter();
  const { role, isSignedIn, user } = useAuth();
  const isAdmin = isSignedIn && role === "admin";

  const [dailyDigest, setDailyDigest] = useState(true);
  const [marketAlerts, setMarketAlerts] = useState(true);

  useEffect(() => {
    if (isAdmin) {
      router.replace("/admin");
      return;
    }

    try {
      const rawDigest = window.localStorage.getItem("settings_daily_digest");
      const rawAlerts = window.localStorage.getItem("settings_market_alerts");
      if (rawDigest !== null) setDailyDigest(rawDigest === "1");
      if (rawAlerts !== null) setMarketAlerts(rawAlerts === "1");
    } catch {
      // Ignore storage failures in constrained environments.
    }
  }, [isAdmin, router]);

  useEffect(() => {
    if (isAdmin) return;
    try {
      window.localStorage.setItem("settings_daily_digest", dailyDigest ? "1" : "0");
      window.localStorage.setItem("settings_market_alerts", marketAlerts ? "1" : "0");
    } catch {
      // Ignore storage failures in constrained environments.
    }
  }, [dailyDigest, marketAlerts, isAdmin]);

  if (isAdmin) {
    return (
      <div className="space-y-4">
        <h1 className="text-xl font-semibold tracking-tight">Settings</h1>
        <p className="text-sm text-slate-300">Opening Admin panel...</p>
      </div>
    );
  }

  return (
    <div className="space-y-5">
      <h1 className="text-xl font-semibold tracking-tight">Settings</h1>

      <section className="rounded-xl border border-slate-800 bg-slate-900/60 p-4">
        <h2 className="text-sm font-semibold text-slate-100">Account</h2>
        <p className="mt-1 text-sm text-slate-300">
          Signed in as {user?.email ?? "Guest"}
          {isSignedIn ? ` (${role})` : ""}
        </p>
      </section>

      <section className="rounded-xl border border-slate-800 bg-slate-900/60 p-4">
        <h2 className="text-sm font-semibold text-slate-100">Preferences</h2>
        <div className="mt-3 space-y-3">
          <label className="flex items-center justify-between gap-3 rounded-lg border border-slate-800 bg-slate-950/50 px-3 py-2">
            <span className="text-sm text-slate-200">Daily market digest</span>
            <input
              type="checkbox"
              checked={dailyDigest}
              onChange={(e) => setDailyDigest(e.target.checked)}
              className="h-4 w-4 accent-sky-500"
            />
          </label>

          <label className="flex items-center justify-between gap-3 rounded-lg border border-slate-800 bg-slate-950/50 px-3 py-2">
            <span className="text-sm text-slate-200">Price movement alerts</span>
            <input
              type="checkbox"
              checked={marketAlerts}
              onChange={(e) => setMarketAlerts(e.target.checked)}
              className="h-4 w-4 accent-sky-500"
            />
          </label>
        </div>
      </section>

      <p className="text-xs text-slate-400">
        Preferences are saved locally on this browser.
      </p>
    </div>
  );
}
