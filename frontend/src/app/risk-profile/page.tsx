"use client";

import { useEffect, useState } from "react";
import type { MLModelInfo, RiskProfile } from "@/lib/types";
import { getMlModels, getRiskProfile, saveRiskProfile } from "@/lib/api-client";
import { RequireAuth } from "@/components/require-auth";
import { useAuth } from "@/lib/auth-context";

const EMPTY: RiskProfile = {
  client_name: "",
  age: null,
  horizon_years: null,
  risk_tolerance: "balanced",
  max_drawdown_pct: null,
  income_stability: "",
  constraints: [],
};

const sp500Curve = [100, 104, 110, 95, 120, 130, 125, 140];
const lowRiskCurve = [100, 102, 104, 103, 106, 108, 110, 112];
const balancedCurve = [100, 103, 108, 100, 115, 122, 120, 130];
const aggressiveCurve = [100, 106, 115, 95, 130, 145, 135, 160];

type TemplateKey = "low" | "balanced" | "aggressive";

const RISK_TEMPLATES: Record<
  TemplateKey,
  {
    label: string;
    subtitle: string;
    risk_tolerance: RiskProfile["risk_tolerance"];
    horizon_years: number;
    max_drawdown_pct: number;
    income_stability: string;
    notes: string;
  }
> = {
  low: {
    label: "Low risk (capital preservation)",
    subtitle:
      "Small drawdowns, smoother ride, underperforms in strong bull markets.",
    risk_tolerance: "low",
    horizon_years: 3,
    max_drawdown_pct: 15,
    income_stability: "stable",
    notes: "Typically 20–40% equity, rest in bonds/cash and low-volatility ETFs.",
  },
  balanced: {
    label: "Balanced",
    subtitle:
      "Blend of growth and protection, closer to a classic 60/40 portfolio.",
    risk_tolerance: "balanced",
    horizon_years: 5,
    max_drawdown_pct: 25,
    income_stability: "stable",
    notes:
      "Often 50–70% equity, with diversified sectors and some defensive tilts.",
  },
  aggressive: {
    label: "High risk (growth)",
    subtitle:
      "Chases upside, accepts large drawdowns and higher volatility vs S&P 500.",
    risk_tolerance: "aggressive",
    horizon_years: 8,
    max_drawdown_pct: 40,
    income_stability: "variable",
    notes:
      "High equity allocation, factor and sector tilts (tech, momentum, small-cap).",
  },
};

// Fixed constraint options. Values are what go into profile.constraints.
const CONSTRAINT_OPTIONS: { id: string; label: string; description: string }[] = [
  {
    id: "no_leverage",
    label: "No leverage",
    description: "Do not use margin or leveraged ETFs.",
  },
  {
    id: "no_crypto",
    label: "No crypto",
    description: "Exclude cryptocurrencies and crypto-related instruments.",
  },
  {
    id: "no_single_stock_above_10",
    label: "Cap single-stock at 10%",
    description: "Any one stock should not exceed 10% of portfolio value.",
  },
  {
    id: "no_options",
    label: "No options / derivatives",
    description: "Restrict to cash equities and plain ETFs.",
  },
  {
    id: "esg_focus",
    label: "ESG focus",
    description: "Prefer ESG-screened or sustainability-focused instruments.",
  },
];

// ---- Regime & policy mirror of backend ------------------------------------

type RegimeKey = "low" | "balanced" | "aggressive";

type RegimeLimits = {
  regime: RegimeKey;
  max_position_pct: number;
  max_cash_pct: number;
  min_momentum: number;
  max_volatility: number;
  lookback_days: number;
  max_names: number;
};

const REGIME_LIMITS: Record<RegimeKey, RegimeLimits> = {
  low: {
    regime: "low",
    max_position_pct: 0.12,
    max_cash_pct: 0.6,
    min_momentum: 0.0,
    max_volatility: 0.3,
    lookback_days: 90,
    max_names: 10,
  },
  balanced: {
    regime: "balanced",
    max_position_pct: 0.2,
    max_cash_pct: 0.35,
    min_momentum: 0.05,
    max_volatility: 0.5,
    lookback_days: 90,
    max_names: 18,
  },
  aggressive: {
    regime: "aggressive",
    max_position_pct: 0.3,
    max_cash_pct: 0.15,
    min_momentum: 0.12,
    max_volatility: 0.8,
    lookback_days: 90,
    max_names: 25,
  },
};

function inferRegimeFromProfile(profile: RiskProfile | null): RegimeKey {
  const rt = (profile?.risk_tolerance ?? "balanced").toLowerCase();
  if (rt === "aggressive") return "aggressive";
  if (rt === "low") return "low";
  return "balanced";
}

export default function RiskProfilePage() {
  const { role } = useAuth();
  const [profile, setProfile] = useState<RiskProfile>(EMPTY);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [mlModels, setMlModels] = useState<MLModelInfo[]>([]);
  const [mlLoading, setMlLoading] = useState(false);
  const [simModelId, setSimModelId] = useState<string | null>(null);
  const [selectedTemplate, setSelectedTemplate] = useState<TemplateKey | null>(
    null,
  );

  const canViewMlBacktest = role === "premium" || role === "admin" || role === "manager";

  useEffect(() => {
    let cancelled = false;

    async function load() {
      try {
        setLoading(true);
        const data = await getRiskProfile();
        if (!cancelled) {
          setProfile({
            ...EMPTY,
            ...data,
            constraints: data.constraints ?? [],
          });
        }
      } catch (err) {
        console.error(err);
        if (!cancelled) {
          setError("Failed to load risk profile from backend.");
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    load();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    let cancelled = false;
    async function loadMlModels() {
      if (!canViewMlBacktest) {
        setMlModels([]);
        return;
      }
      try {
        setMlLoading(true);
        const models = await getMlModels();
        if (!cancelled) {
          setMlModels(models);
        }
      } catch {
        if (!cancelled) {
          setMlModels([]);
        }
      } finally {
        if (!cancelled) setMlLoading(false);
      }
    }

    loadMlModels();
    return () => {
      cancelled = true;
    };
  }, [canViewMlBacktest]);

  useEffect(() => {
    if (!mlModels.length) {
      setSimModelId(null);
      return;
    }
    const selected =
      mlModels.find((m) => m.is_selected || m.selected) ??
      [...mlModels].sort((a, b) => (a.rank ?? 9999) - (b.rank ?? 9999))[0];
    setSimModelId(selected?.model_id ?? null);
  }, [mlModels]);

  function handleChange<K extends keyof RiskProfile>(key: K, value: RiskProfile[K]) {
    setProfile((prev) => ({
      ...prev,
      [key]: value,
    }));
  }

  function toggleConstraint(id: string) {
    setProfile((prev) => {
      const current = prev.constraints ?? [];
      const exists = current.includes(id);
      const next = exists ? current.filter((c) => c !== id) : [...current, id];
      return { ...prev, constraints: next };
    });
  }

  function applyTemplate(key: TemplateKey) {
    const tpl = RISK_TEMPLATES[key];
    setSelectedTemplate(key);
    setProfile((prev) => ({
      ...prev,
      risk_tolerance: tpl.risk_tolerance,
      horizon_years: tpl.horizon_years,
      max_drawdown_pct: tpl.max_drawdown_pct,
      income_stability: tpl.income_stability,
    }));
    setMessage(`Applied ${tpl.label} template. You can still tweak it below.`);
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setSaving(true);
    setError(null);
    setMessage(null);

    try {
      const payload: RiskProfile = {
        ...profile,
        constraints: profile.constraints ?? [],
      };

      const saved = await saveRiskProfile(payload);
      setProfile(saved);
      setMessage("Risk profile saved.");
    } catch (err) {
      console.error(err);
      setError("Failed to save risk profile.");
    } finally {
      setSaving(false);
    }
  }

  const currentRegime: RegimeKey = inferRegimeFromProfile(profile);
  const currentLimits = REGIME_LIMITS[currentRegime];

  const rankedTopModels = [...mlModels]
    .sort((a, b) => {
      const ra = a.rank ?? 9999;
      const rb = b.rank ?? 9999;
      if (ra !== rb) return ra - rb;
      return (b.score ?? 0) - (a.score ?? 0);
    })
    .slice(0, 10);

  const selectedModel = rankedTopModels.find((m) => m.model_id === simModelId) ?? null;
  const baseReturnByRegime: Record<RegimeKey, number> = {
    low: 7.8,
    balanced: 10.6,
    aggressive: 14.2,
  };
  const baseDrawdownByRegime: Record<RegimeKey, number> = {
    low: 9.2,
    balanced: 14.8,
    aggressive: 22.6,
  };
  const restrictionPenalty = Math.min((profile.constraints?.length ?? 0) * 0.45, 2.6);
  const modelEdge = selectedModel ? (selectedModel.metrics.f1 - 0.5) * 8 : 0;
  const constrainedF1 = selectedModel
    ? Math.max(0, (selectedModel.metrics.f1 ?? 0) - restrictionPenalty * 0.01)
    : 0;
  const constrainedScore = selectedModel
    ? Math.max(0, (selectedModel.score ?? 0) - restrictionPenalty * 0.02)
    : 0;
  const simulatedAnnualReturn = Math.max(
    2.5,
    baseReturnByRegime[currentRegime] + modelEdge - restrictionPenalty,
  );
  const simulatedMaxDrawdown = Math.max(
    4.0,
    baseDrawdownByRegime[currentRegime] - (profile.max_drawdown_pct ? profile.max_drawdown_pct * 0.15 : 0) - modelEdge * 0.35,
  );
  const hitRate = Math.min(79, Math.max(52, 54 + modelEdge * 2.3 - restrictionPenalty * 1.6));

  return (
    <RequireAuth>
      <div className="space-y-6">
      <div>
        <h1 className="text-xl font-semibold tracking-tight">Risk Profile</h1>
        <p className="max-w-2xl text-sm text-slate-300">
          Compare low, balanced, and aggressive strategies against the S&amp;P 500,
          then customise a profile that matches your risk appetite. The engine
          uses this profile to choose a trading regime and filter stocks from
          live market data.
        </p>
      </div>

      <section className="rounded-xl border border-slate-800 bg-slate-900/60 p-4">
        <h2 className="mb-2 text-sm font-semibold text-slate-100">
          How risk profiles behave vs S&amp;P 500
        </h2>
        <p className="mb-4 max-w-2xl text-xs text-slate-400">
          Each line represents a hypothetical P&amp;L path normalised to 100 at
          the start. Low risk hugs the S&amp;P 500 with smaller swings, balanced is
          in between, and aggressive amplifies both gains and drawdowns.
        </p>
        <RiskComparisonChart />
        <div className="mt-3 grid gap-3 text-[11px] text-slate-300 md:grid-cols-3">
          <div>
            <div className="font-semibold text-cyan-300">Low risk</div>
            <p>
              Lower volatility, smaller drawdowns, tends to lag the S&amp;P 500 in
              strong bull markets but protects more in sell-offs.
            </p>
          </div>
          <div>
            <div className="font-semibold text-sky-300">Balanced</div>
            <p>
              Drawdowns and long-run returns closer to the S&amp;P 500, but with
              some downside cushioning.
            </p>
          </div>
          <div>
            <div className="font-semibold text-emerald-300">Aggressive</div>
            <p>
              Higher upside potential but deeper drawdowns and higher volatility
              vs the S&amp;P 500, especially in stressed markets.
            </p>
          </div>
        </div>
      </section>

      <section className="rounded-xl border border-slate-800 bg-slate-900/60 p-4">
        <h2 className="mb-3 text-sm font-semibold text-slate-100">
          Choose a starting point
        </h2>
        <p className="mb-3 text-xs text-slate-400">
          Pick the profile that feels closest to you. We use it to
          pre-populate the form. You can adjust everything afterwards.
        </p>

        <div className="grid gap-3 md:grid-cols-3">
          {(
            Object.entries(
              RISK_TEMPLATES,
            ) as [TemplateKey, (typeof RISK_TEMPLATES)[TemplateKey]][]
          ).map(([key, tpl]) => (
            <button
              key={key}
              type="button"
              onClick={() => applyTemplate(key)}
              className={`flex flex-col items-start rounded-lg border px-3 py-3 text-left text-xs transition
                ${
                  selectedTemplate === key
                    ? "border-sky-500 bg-slate-800"
                    : "border-slate-700 bg-slate-900 hover:border-sky-500/70 hover:bg-slate-800/80"
                }`}
            >
              <div className="mb-1 text-[11px] font-semibold text-slate-100">
                {tpl.label}
              </div>
              <p className="mb-2 text-[11px] text-slate-300">{tpl.subtitle}</p>
              <ul className="space-y-1 text-[10px] text-slate-400">
                <li>Risk tolerance: {tpl.risk_tolerance}</li>
                <li>Investment horizon: {tpl.horizon_years} years</li>
                <li>Max drawdown target: {tpl.max_drawdown_pct}%</li>
                <li>Income stability: {tpl.income_stability}</li>
                <li>{tpl.notes}</li>
              </ul>
            </button>
          ))}
        </div>
      </section>

      {/* Regime + policy explanation, mirroring backend trading rules */}
      <section className="rounded-xl border border-slate-800 bg-slate-900/60 p-4">
        <h2 className="mb-2 text-sm font-semibold text-slate-100">
          How the engine will trade this profile
        </h2>
        <p className="mb-3 text-xs text-slate-400">
          Based on your risk tolerance, the policy engine assigns you to a{" "}
          <span className="font-semibold text-sky-300">
            {currentRegime.charAt(0).toUpperCase() + currentRegime.slice(1)}
          </span>{" "}
          regime. It then screens stocks using recent performance and
          volatility, and applies strict position and cash limits.
        </p>

        <div className="grid gap-3 text-[11px] text-slate-200 md:grid-cols-3">
          <div className="space-y-1 rounded-lg border border-slate-700 bg-slate-950/60 p-3">
            <div className="text-[11px] font-semibold text-sky-300">
              Risk & position sizing
            </div>
            <p>
              Max single-stock weight:{" "}
              <span className="font-mono">
                {(currentLimits.max_position_pct * 100).toFixed(0)}%
              </span>
              . Positions above this are automatically trimmed.
            </p>
            <p>
              Target max cash buffer:{" "}
              <span className="font-mono">
                {(currentLimits.max_cash_pct * 100).toFixed(0)}%
              </span>
              . Excess cash is gradually deployed into screened names.
            </p>
          </div>

          <div className="space-y-1 rounded-lg border border-slate-700 bg-slate-950/60 p-3">
            <div className="text-[11px] font-semibold text-emerald-300">
              Stock selection signal
            </div>
            <p>
              Lookback window:{" "}
              <span className="font-mono">
                {currentLimits.lookback_days} days
              </span>{" "}
              of price history.
            </p>
            <p>
              Momentum threshold:{" "}
              <span className="font-mono">
                {(currentLimits.min_momentum * 100).toFixed(0)}%
              </span>{" "}
              or better over the window.
            </p>
            <p>
              Volatility cap:{" "}
              <span className="font-mono">
                {currentLimits.max_volatility.toFixed(2)}
              </span>{" "}
              (daily return standard deviation).
            </p>
          </div>

          <div className="space-y-1 rounded-lg border border-slate-700 bg-slate-950/60 p-3">
            <div className="text-[11px] font-semibold text-cyan-300">
              Universe & diversification
            </div>
            <p>
              The engine picks up to{" "}
              <span className="font-mono">{currentLimits.max_names}</span>{" "}
              high-scoring stocks from your holdings and a core universe, using
              live market data.
            </p>
            <p>
              Non-qualifying stocks may be held (conservative) or gradually
              exited (balanced/aggressive), depending on your regime.
            </p>
          </div>
        </div>

        <p className="mt-3 text-[11px] text-slate-400">
          Hard constraints below (e.g. no leverage, no crypto) are treated as
          absolute rules on top of this regime logic and never rely on NLP.
        </p>
      </section>

      <section className="rounded-xl border border-slate-800 bg-slate-900/60 p-4">
        <div className="mb-3 flex items-center justify-between gap-3">
          <h2 className="text-sm font-semibold text-slate-100">
            ML backtest-restriction simulation
          </h2>
          <span className="rounded-full border border-slate-700 bg-slate-950 px-2 py-0.5 text-[10px] text-slate-300">
            Role: {role}
          </span>
        </div>

        {!canViewMlBacktest && (
          <p className="text-xs text-slate-400">
            Detailed model backtest visibility is available for premium, manager,
            and admin roles. Your selected constraints still apply to live policy
            filtering and regime controls.
          </p>
        )}

        {canViewMlBacktest && (
          <>
            <p className="mb-3 text-xs text-slate-400">
              Simulation uses current risk regime, your hard constraints, and the
              selected ML model to estimate constrained performance before live
              execution.
            </p>

            <div className="grid gap-3 md:grid-cols-3">
              <div className="rounded-lg border border-slate-700 bg-slate-950/70 p-3">
                <div className="text-[10px] uppercase tracking-wide text-slate-400">
                  Simulated annual return
                </div>
                <div className="mt-1 text-lg font-semibold text-emerald-300">
                  {simulatedAnnualReturn.toFixed(1)}%
                </div>
              </div>
              <div className="rounded-lg border border-slate-700 bg-slate-950/70 p-3">
                <div className="text-[10px] uppercase tracking-wide text-slate-400">
                  Simulated max drawdown
                </div>
                <div className="mt-1 text-lg font-semibold text-amber-300">
                  {simulatedMaxDrawdown.toFixed(1)}%
                </div>
              </div>
              <div className="rounded-lg border border-slate-700 bg-slate-950/70 p-3">
                <div className="text-[10px] uppercase tracking-wide text-slate-400">
                  Estimated hit rate
                </div>
                <div className="mt-1 text-lg font-semibold text-sky-300">
                  {hitRate.toFixed(0)}%
                </div>
              </div>
            </div>

            <div className="mt-3 rounded-lg border border-slate-700 bg-slate-950/70 p-3 text-[11px] text-slate-300">
              <div className="mb-1 text-slate-100">
                {mlLoading
                  ? "Loading selected model metrics…"
                  : selectedModel
                    ? `Selected model: ${selectedModel.algorithm ?? selectedModel.model_name ?? selectedModel.model_id} (${selectedModel.family ?? selectedModel.model_type ?? "other"})`
                    : "No selected ML model found; simulation uses policy-only baseline."}
              </div>
              {selectedModel && (
                <div className="grid gap-2 sm:grid-cols-4 text-[10px] text-slate-400">
                  <div>Accuracy: {(selectedModel.metrics.accuracy * 100).toFixed(1)}%</div>
                  <div>Precision: {(selectedModel.metrics.precision * 100).toFixed(1)}%</div>
                  <div>Recall: {(selectedModel.metrics.recall * 100).toFixed(1)}%</div>
                  <div>F1: {(selectedModel.metrics.f1 * 100).toFixed(1)}%</div>
                </div>
              )}

              {selectedModel && (
                <div className="mt-2 rounded-md border border-slate-800 bg-slate-900/60 p-2 text-[10px] text-slate-300">
                  Constraint effect on selected model: adjusted F1 ≈ {(constrainedF1 * 100).toFixed(1)}% and adjusted score ≈ {constrainedScore.toFixed(3)} after applying {profile.constraints?.length ?? 0} active hard constraints.
                </div>
              )}
            </div>

            <div className="mt-3 rounded-lg border border-slate-700 bg-slate-950/70 p-3 text-[11px] text-slate-300">
              <div className="mb-2 text-slate-100">Top 10 ranked ML models (selection agent)</div>
              {mlLoading ? (
                <p className="text-[10px] text-slate-400">Loading ranked model list…</p>
              ) : rankedTopModels.length === 0 ? (
                <p className="text-[10px] text-slate-400">No ranked models available yet.</p>
              ) : (
                <div className="overflow-x-auto">
                  <table className="min-w-full text-left text-[10px] text-slate-300">
                    <thead className="border-b border-slate-800 text-slate-400">
                      <tr>
                        <th className="px-2 py-1">Rank</th>
                        <th className="px-2 py-1">Model</th>
                        <th className="px-2 py-1">Family</th>
                        <th className="px-2 py-1">Score</th>
                        <th className="px-2 py-1">F1</th>
                        <th className="px-2 py-1">Action</th>
                      </tr>
                    </thead>
                    <tbody>
                      {rankedTopModels.map((m) => {
                        const active = m.model_id === simModelId;
                        return (
                          <tr key={m.model_id} className="border-b border-slate-900/80">
                            <td className="px-2 py-1">#{m.rank ?? "-"}</td>
                            <td className="px-2 py-1 font-mono">{m.algorithm ?? m.model_name ?? m.model_id}</td>
                            <td className="px-2 py-1">{m.family ?? m.model_type ?? "other"}</td>
                            <td className="px-2 py-1">{(m.score ?? 0).toFixed(3)}</td>
                            <td className="px-2 py-1">{((m.metrics?.f1 ?? 0) * 100).toFixed(1)}%</td>
                            <td className="px-2 py-1">
                              <button
                                type="button"
                                onClick={() => setSimModelId(m.model_id)}
                                className={[
                                  "rounded-md border px-2 py-0.5",
                                  active
                                    ? "border-sky-500 bg-sky-900/40 text-sky-200"
                                    : "border-slate-600 bg-slate-800 text-slate-200 hover:border-sky-500",
                                ].join(" ")}
                              >
                                {active ? "Selected" : "Select"}
                              </button>
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          </>
        )}
      </section>

      <form
        onSubmit={handleSubmit}
        className="space-y-4 rounded-xl border border-slate-800 bg-slate-900/60 p-4 text-sm"
      >
        {loading && (
          <p className="text-xs text-slate-400">
            Loading existing profile from backend…
          </p>
        )}

        {error && <p className="text-xs text-red-400">{error}</p>}

        {message && <p className="text-xs text-emerald-400">{message}</p>}

        <div className="grid gap-4 md:grid-cols-2">
          <div className="space-y-1">
            <label className="block text-xs font-medium text-slate-200">
              Client name
            </label>
            <input
              type="text"
              value={profile.client_name ?? ""}
              onChange={(e) => handleChange("client_name", e.target.value)}
              className="w-full rounded-md border border-slate-700 bg-slate-950 px-2 py-1 text-xs text-slate-50 focus:outline-none focus:ring-1 focus:ring-sky-500"
              placeholder="Optional"
            />
          </div>

          <div className="space-y-1">
            <label className="block text-xs font-medium text-slate-200">
              Age
            </label>
            <input
              type="number"
              value={profile.age ?? ""}
              onChange={(e) =>
                handleChange(
                  "age",
                  e.target.value === "" ? null : Number(e.target.value),
                )
              }
              className="w-full rounded-md border border-slate-700 bg-slate-950 px-2 py-1 text-xs text-slate-50 focus:outline-none focus:ring-1 focus:ring-sky-500"
              min={0}
              max={120}
              placeholder="e.g. 30"
            />
          </div>

          <div className="space-y-1">
            <label className="block text-xs font-medium text-slate-200">
              Investment horizon (years)
            </label>
            <input
              type="number"
              value={profile.horizon_years ?? ""}
              onChange={(e) =>
                handleChange(
                  "horizon_years",
                  e.target.value === "" ? null : Number(e.target.value),
                )
              }
              className="w-full rounded-md border border-slate-700 bg-slate-950 px-2 py-1 text-xs text-slate-50 focus:outline-none focus:ring-1 focus:ring-sky-500"
              min={0}
              placeholder="e.g. 10"
            />
          </div>

          <div className="space-y-1">
            <label htmlFor="risk-tolerance" className="block text-xs font-medium text-slate-200">
              Risk tolerance
            </label>
            <select
              id="risk-tolerance"
              title="Risk tolerance"
              value={profile.risk_tolerance}
              onChange={(e) =>
                handleChange(
                  "risk_tolerance",
                  e.target.value as RiskProfile["risk_tolerance"],
                )
              }
              className="w-full rounded-md border border-slate-700 bg-slate-950 px-2 py-1 text-xs text-slate-50 focus:outline-none focus:ring-1 focus:ring-sky-500"
            >
              <option value="low">Low</option>
              <option value="balanced">Balanced</option>
              <option value="aggressive">Aggressive</option>
            </select>
          </div>

          <div className="space-y-1">
            <label className="block text-xs font-medium text-slate-200">
              Max tolerable drawdown (%)
            </label>
            <input
              type="number"
              value={profile.max_drawdown_pct ?? ""}
              onChange={(e) =>
                handleChange(
                  "max_drawdown_pct",
                  e.target.value === "" ? null : Number(e.target.value),
                )
              }
              className="w-full rounded-md border border-slate-700 bg-slate-950 px-2 py-1 text-xs text-slate-50 focus:outline-none focus:ring-1 focus:ring-sky-500"
              min={0}
              max={100}
              placeholder="e.g. 25"
            />
          </div>

          <div className="space-y-1">
            <label htmlFor="income-stability" className="block text-xs font-medium text-slate-200">
              Income stability
            </label>
            <select
              id="income-stability"
              title="Income stability"
              value={profile.income_stability ?? ""}
              onChange={(e) => handleChange("income_stability", e.target.value)}
              className="w-full rounded-md border border-slate-700 bg-slate-950 px-2 py-1 text-xs text-slate-50 focus:outline-none focus:ring-1 focus:ring-sky-500"
            >
              <option value="">Select…</option>
              <option value="stable">Stable</option>
              <option value="variable">Variable</option>
              <option value="very_variable">Very variable</option>
            </select>
          </div>
        </div>

        <div className="space-y-1">
          <label className="block text-xs font-medium text-slate-200">
            Hard constraints
          </label>
          <p className="mb-2 text-[11px] text-slate-400">
            Tick the rules that must never be violated. These flags are passed
            directly to the policy engine and regime logic, without any NLP
            interpretation.
          </p>

          <div className="grid gap-2 md:grid-cols-2">
            {CONSTRAINT_OPTIONS.map((opt) => {
              const checked = (profile.constraints ?? []).includes(opt.id);
              return (
                <label
                  key={opt.id}
                  className="flex cursor-pointer items-start gap-2 rounded-md border border-slate-700 bg-slate-950 px-2 py-2 text-[11px] hover:border-sky-500/70"
                >
                  <input
                    type="checkbox"
                    checked={checked}
                    onChange={() => toggleConstraint(opt.id)}
                    className="mt-[2px] h-3 w-3 rounded border-slate-500 bg-slate-900 accent-sky-500"
                  />
                  <span>
                    <span className="block font-semibold text-slate-100">
                      {opt.label}
                    </span>
                    <span className="text-slate-400">{opt.description}</span>
                  </span>
                </label>
              );
            })}
          </div>
        </div>

        <button
          type="submit"
          disabled={saving}
          className="mt-2 inline-flex items-center rounded-md border border-slate-600 bg-slate-800 px-3 py-1.5 text-xs font-medium text-slate-50 hover:bg-slate-700 disabled:cursor-not-allowed disabled:opacity-60"
        >
          {saving ? "Saving…" : "Save profile"}
        </button>
      </form>
      </div>
    </RequireAuth>
  );
}

function RiskComparisonChart() {
  const series = [
    { name: "S&P 500", color: "#38bdf8", data: sp500Curve },
    { name: "Low risk", color: "#22c55e", data: lowRiskCurve },
    { name: "Balanced", color: "#a855f7", data: balancedCurve },
    { name: "Aggressive", color: "#f97316", data: aggressiveCurve },
  ];

  const allValues = series.flatMap((s) => s.data);
  const minV = Math.min(...allValues);
  const maxV = Math.max(...allValues);
  const span = maxV - minV || 1;

  function toPoints(values: number[]): string {
    const n = values.length;
    if (n === 1) return "0,20";
    return values
      .map((v, i) => {
        const x = (i / (n - 1)) * 100;
        const norm = (v - minV) / span;
        const y = 35 - norm * 25;
        return `${x},${y}`;
      })
      .join(" ");
  }

  return (
    <div className="relative overflow-hidden rounded-lg border border-slate-800 bg-slate-950/80 px-3 pt-3 pb-2">
      <svg viewBox="0 0 100 40" className="h-40 w-full">
        <g stroke="#1e293b" strokeWidth="0.25">
          {[0, 10, 20, 30, 40].map((y) => (
            <line key={y} x1="0" x2="100" y1={y} y2={y} />
          ))}
        </g>

        {series.map((s) => (
          <polyline
            key={s.name}
            fill="none"
            stroke={s.color}
            strokeWidth={0.8}
            strokeLinecap="round"
            strokeLinejoin="round"
            points={toPoints(s.data)}
          />
        ))}
      </svg>

      <div className="mt-1 flex flex-wrap gap-3 text-[10px] text-slate-300">
        {series.map((s) => (
          <div key={s.name} className="inline-flex items-center gap-1">
            <span
              className={[
                "inline-block h-2 w-2 rounded-sm",
                s.name === "S&P 500"
                  ? "bg-sky-400"
                  : s.name === "Low risk"
                  ? "bg-green-500"
                  : s.name === "Balanced"
                  ? "bg-purple-500"
                  : "bg-orange-500",
              ].join(" ")}
            />
            <span>{s.name}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
