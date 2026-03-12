// frontend/src/app/dashboard/page.tsx
"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import { getDailySignals, getMarketHistory } from "@/lib/api-client";
import type { DailySignals, SignalTopItem, NewsItem } from "@/lib/types";

type HistoryPoint = { t: string; o?: number; h?: number; l?: number; c: number; v?: number };
type History = { symbol: string; points: HistoryPoint[] };

function clamp(n: number, lo: number, hi: number) {
  return Math.max(lo, Math.min(hi, n));
}

function stripTrailingPunctuation(s: string) {
  return (s || "").replace(/[.\s]+$/, "");
}

function formatMaybeDate(s?: string | null) {
  if (!s) return null;
  // backend may return ISO string or null; keep simple and safe
  const d = new Date(s);
  if (Number.isNaN(d.getTime())) return s;
  return d.toLocaleString();
}

function CandlesChart({ points }: { points: HistoryPoint[] }) {
  const w = 760;
  const h = 360;
  const padX = 18;
  const topPad = 12;
  const priceAreaH = 260;
  const volumeAreaTop = topPad + priceAreaH + 12;
  const volumeAreaH = h - volumeAreaTop - 18;

  const normalized = points.map((p, i) => {
    const prev = i > 0 ? points[i - 1] : null;
    const close = Number(p.c || 0);
    const openRaw = Number(p.o ?? NaN);
    const highRaw = Number(p.h ?? NaN);
    const lowRaw = Number(p.l ?? NaN);
    const open = Number.isFinite(openRaw) ? openRaw : Number(prev?.c ?? close);
    const high = Number.isFinite(highRaw) ? highRaw : Math.max(open, close);
    const low = Number.isFinite(lowRaw) ? lowRaw : Math.min(open, close);
    const volume = Math.max(0, Number(p.v ?? 0));
    return { t: p.t, o: open, h: high, l: low, c: close, v: volume };
  });

  const highs = normalized.map((p) => p.h);
  const lows = normalized.map((p) => p.l);
  const minY = Math.min(...lows);
  const maxY = Math.max(...highs);
  const spanY = Math.max(1e-9, maxY - minY);
  const maxV = Math.max(1, ...normalized.map((p) => p.v));

  const left = padX;
  const right = w - padX;
  const plotW = right - left;
  const step = plotW / Math.max(1, normalized.length);
  const bodyW = Math.max(2, Math.min(8, step * 0.65));

  const yPrice = (v: number) => topPad + (1 - (v - minY) / spanY) * priceAreaH;
  const yVol = (v: number) => volumeAreaTop + volumeAreaH - (v / maxV) * volumeAreaH;

  const first = normalized[0]?.t ?? "";
  const last = normalized[normalized.length - 1]?.t ?? "";

  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="w-full">
      {normalized.map((p, i) => {
        const x = left + i * step + step / 2;
        const yo = yPrice(p.o);
        const yc = yPrice(p.c);
        const yh = yPrice(p.h);
        const yl = yPrice(p.l);
        const up = p.c >= p.o;
        const color = up ? "#10b981" : "#f43f5e";
        const bodyTop = Math.min(yo, yc);
        const bodyH = Math.max(1.2, Math.abs(yc - yo));

        return (
          <g key={`${p.t}-${i}`}>
            <line x1={x} y1={yh} x2={x} y2={yl} stroke={color} strokeWidth={1} opacity={0.9} />
            <rect x={x - bodyW / 2} y={bodyTop} width={bodyW} height={bodyH} fill={color} opacity={0.85} rx={0.5} />
            <rect
              x={x - bodyW / 2}
              y={yVol(p.v)}
              width={Math.max(1.5, bodyW * 0.85)}
              height={Math.max(1, volumeAreaTop + volumeAreaH - yVol(p.v))}
              fill={color}
              opacity={0.5}
            />
          </g>
        );
      })}

      <text x={left} y={h - 4} fontSize="11" opacity="0.7" fill="currentColor">
        {first} → {last}
      </text>
      <text x={right} y={topPad + 10} fontSize="11" opacity="0.7" textAnchor="end" fill="currentColor">
        {minY.toFixed(2)} – {maxY.toFixed(2)}
      </text>
      <text x={left} y={volumeAreaTop - 2} fontSize="10" opacity="0.6" fill="currentColor">
        Volume
      </text>
    </svg>
  );
}

function Modal({
  title,
  children,
  onClose,
}: {
  title: string;
  children: React.ReactNode;
  onClose: () => void;
}) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4">
      <div className="w-full max-w-3xl rounded-2xl border border-slate-800 bg-slate-950/95 shadow-xl">
        <div className="flex items-center justify-between border-b border-slate-800 px-4 py-3">
          <div className="text-sm font-semibold text-slate-100">{title}</div>
          <button
            onClick={onClose}
            className="rounded-lg border border-slate-800 bg-slate-900/60 px-3 py-1 text-xs text-slate-200 hover:bg-slate-900"
          >
            Close
          </button>
        </div>
        <div className="p-4 text-slate-200">{children}</div>
      </div>
    </div>
  );
}

function InfoPopover({ rec }: { rec: SignalTopItem }) {
  const [open, setOpen] = useState(false);
  const wrapRef = useRef<HTMLSpanElement | null>(null);

  useEffect(() => {
    function onDocClick(e: MouseEvent) {
      if (!open) return;
      const target = e.target as Node;
      if (wrapRef.current && !wrapRef.current.contains(target)) setOpen(false);
    }

    function onKeyDown(e: KeyboardEvent) {
      if (!open) return;
      if (e.key === "Escape") setOpen(false);
    }

    document.addEventListener("mousedown", onDocClick);
    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("mousedown", onDocClick);
      document.removeEventListener("keydown", onKeyDown);
    };
  }, [open]);

  const title = useMemo(() => {
    const side = rec.direction === "up" ? "Up list" : "Down list";
    return `${rec.symbol} • ${side}`;
  }, [rec.symbol, rec.direction]);

  const explanation = useMemo(() => {
    const side = rec.direction === "up" ? "Up list" : "Down list";
    return {
      title,
      bullets: [
        `${side} ranks symbols by a score that blends momentum, volatility, and a light news signal.`,
        `Momentum is recent price change; volatility is how much price swings day to day.`,
        `A higher score means a stronger trend signal, not a guarantee.`,
        `Open the glossary to learn the terms used across the dashboard.`,
      ],
    };
  }, [title, rec.direction]);

  return (
    <span ref={wrapRef} className="relative inline-flex items-center">
      <button
        type="button"
        onClick={(e) => {
          e.stopPropagation();
          // Do not toggle closed by clicking the icon again.
          // Close happens only on outside click or Esc.
          setOpen(true);
        }}
        className="ml-2 inline-flex h-6 w-6 items-center justify-center rounded-full border border-slate-700 bg-slate-900/60 text-slate-200 hover:bg-slate-900"
        aria-label="Explain signal"
        title="Explain"
      >
        <svg viewBox="0 0 24 24" width="14" height="14" fill="none" aria-hidden="true">
          <circle cx="12" cy="12" r="9" stroke="currentColor" strokeWidth="2" />
          <path d="M12 10.5v6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
          <path d="M12 7.2h.01" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
        </svg>
      </button>

      {open ? (
        <div
          className="absolute right-0 top-7 z-50 w-[min(420px,86vw)] rounded-xl border border-slate-800 bg-slate-950/95 p-3 text-xs text-slate-200 shadow-xl"
          onClick={(e) => e.stopPropagation()}
          role="dialog"
          aria-label="Signal explanation"
        >
          <div className="mb-2 text-xs font-semibold text-slate-100">{explanation.title}</div>
          <ul className="space-y-1">
            {explanation.bullets.map((b) => (
              <li key={b} className="leading-relaxed text-slate-300">
                {b}
              </li>
            ))}
          </ul>

          <div className="mt-3 flex items-center justify-between gap-2">
            <div className="text-[11px] text-slate-400">Closes only on outside click or Esc.</div>
            <Link
              href="/glossary"
              className="rounded-lg border border-slate-800 bg-slate-900/60 px-2 py-1 text-[11px] text-slate-200 hover:bg-slate-900"
            >
              Explore lingo
            </Link>
          </div>
        </div>
      ) : null}
    </span>
  );
}

function RecCard({
  rec,
  tone,
  onOpenChart,
}: {
  rec: SignalTopItem;
  tone: "up" | "down";
  onOpenChart: () => void;
}) {
  const score = Number.isFinite(rec.score) ? rec.score : 0;
  const scoreLabel = score.toFixed(3);
  const rationale = stripTrailingPunctuation(rec.reason ?? "");

  return (
    <button
      onClick={onOpenChart}
      className={`w-full rounded-xl border px-4 py-3 text-left transition ${
        tone === "up"
          ? "border-emerald-700/40 bg-emerald-950/20 hover:bg-emerald-950/30"
          : "border-rose-700/40 bg-rose-950/20 hover:bg-rose-950/30"
      }`}
    >
      <div className="flex items-start justify-between gap-4">
        <div className="min-w-0">
          <div className="text-sm font-semibold tracking-tight text-slate-100">{rec.symbol}</div>
          <div className="mt-1 flex flex-wrap items-center gap-1 text-xs text-slate-300">
            <span className="break-words">{rationale || "No rationale provided."}</span>
            <InfoPopover rec={rec} />
          </div>
        </div>

        <div className="shrink-0 text-right">
          <div className="text-sm font-semibold text-slate-100">{scoreLabel}</div>
          <div className="text-xs text-slate-300">score</div>
        </div>
      </div>
    </button>
  );
}

function threeLineSummary(text: string): string[] {
  const clean = (text || "").trim();
  if (!clean) return [];

  const sentences = clean
    .replace(/\s+/g, " ")
    .split(/(?<=[.!?])\s+/)
    .map((s) => stripTrailingPunctuation(s))
    .filter(Boolean);

  const out: string[] = [];
  for (const s of sentences) {
    if (out.length >= 3) break;
    out.push(s);
  }

  return out.slice(0, 3);
}

function NewsCard({ item }: { item: NewsItem }) {
  const title = item.title || "Untitled";
  const url = item.url;
  const publisher = item.publisher ? stripTrailingPunctuation(String(item.publisher)) : null;

  const maybeSymbol = item.symbol ? String(item.symbol).toUpperCase() : null;
  const published = formatMaybeDate(item.published_at);

  const summaryText =
    (typeof (item as any).summary === "string" ? ((item as any).summary as string) : "") ||
    (typeof (item as any).reason === "string" ? ((item as any).reason as string) : "");

  const lines = threeLineSummary(summaryText);

  return (
    <a
      href={url}
      target="_blank"
      rel="noreferrer"
      className="block rounded-xl border border-slate-800 bg-slate-900/40 px-4 py-3 text-sm text-slate-100 hover:bg-slate-900/55"
    >
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="text-sm font-semibold text-slate-100">{title}</div>

          <div className="mt-1 flex flex-wrap items-center gap-x-2 gap-y-1 text-xs text-slate-300">
            {publisher ? (
              <span className="text-slate-300">{publisher}</span>
            ) : (
              <span className="text-slate-400">News</span>
            )}
            {maybeSymbol ? <span className="text-slate-400">• {maybeSymbol}</span> : null}
            {published ? <span className="text-slate-500">• {published}</span> : null}
          </div>
        </div>

        <div className="shrink-0">
          <span className="rounded-lg border border-slate-800 bg-slate-950/60 px-2 py-1 text-[11px] text-slate-300">
            Open
          </span>
        </div>
      </div>

      <div className="mt-2 space-y-1 text-xs text-slate-300">
        {lines.length ? (
          lines.map((l, i) => (
            <div key={`${item.id}-s${i}`} className="line-clamp-1 leading-relaxed">
              {l}
            </div>
          ))
        ) : (
          <div className="text-slate-400">
            No summary available. Ensure the backend returns a <span className="text-slate-300">summary</span> field on
            each news item.
          </div>
        )}
      </div>
    </a>
  );
}

function FeatureImportanceChart({
  labels,
  values,
}: {
  labels: string[];
  values: number[];
}) {
  function barWidthClass(pct: number): string {
    if (pct <= 5) return "w-1/12";
    if (pct <= 15) return "w-2/12";
    if (pct <= 25) return "w-3/12";
    if (pct <= 35) return "w-4/12";
    if (pct <= 45) return "w-5/12";
    if (pct <= 55) return "w-6/12";
    if (pct <= 65) return "w-7/12";
    if (pct <= 75) return "w-8/12";
    if (pct <= 85) return "w-9/12";
    if (pct <= 95) return "w-10/12";
    return "w-full";
  }

  if (!labels.length || !values.length || labels.length !== values.length) {
    return (
      <div className="text-sm text-slate-300">
        Feature importances are not available for the currently selected model.
      </div>
    );
  }

  const pairs = labels.map((label, i) => ({ label, value: Number(values[i] ?? 0) }));
  pairs.sort((a, b) => b.value - a.value);
  const top = pairs.slice(0, 12);
  const maxVal = Math.max(1e-9, ...top.map((x) => x.value));

  return (
    <div className="space-y-2">
      {top.map((item) => {
        const widthPct = Math.max(2, Math.min(100, (item.value / maxVal) * 100));
        return (
          <div key={item.label} className="space-y-1">
            <div className="flex items-center justify-between gap-3 text-xs text-slate-300">
              <span className="truncate">{item.label}</span>
              <span className="font-mono text-slate-200">{item.value.toFixed(4)}</span>
            </div>
            <div className="h-2 w-full overflow-hidden rounded-full bg-slate-800">
              <div
                className={["h-full rounded-full bg-cyan-500", barWidthClass(widthPct)].join(" ")}
              />
            </div>
          </div>
        );
      })}
    </div>
  );
}

export default function DashboardPage() {
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState<string | null>(null);
  const [signals, setSignals] = useState<DailySignals | null>(null);

  const [activeSymbol, setActiveSymbol] = useState<string | null>(null);
  const [history, setHistory] = useState<History | null>(null);
  const [historyErr, setHistoryErr] = useState<string | null>(null);
  const [historyLoading, setHistoryLoading] = useState(false);

  useEffect(() => {
    let cancelled = false;

    async function load() {
      try {
        setLoading(true);
        setErr(null);
        const data = await getDailySignals();
        console.log("daily signals payload:", data);
        console.log("news count:", Array.isArray((data as any)?.news) ? (data as any).news.length : "no news field");
        if (!cancelled) setSignals(data);
      } catch (e) {
        console.error(e);
        if (!cancelled) setErr("Unable to load dashboard data from backend.");
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    load();
    return () => {
      cancelled = true;
    };
  }, []);

  async function openChart(symbol: string) {
    setActiveSymbol(symbol);
    setHistory(null);
    setHistoryErr(null);
    setHistoryLoading(true);

    try {
      const h = (await getMarketHistory(symbol, 180)) as unknown;
      const parsed = h as any;
      const pointsFromPayload = Array.isArray(parsed?.points) ? (parsed.points as HistoryPoint[]) : [];
      const pointsFromBars = Array.isArray(parsed?.bars)
        ? (parsed.bars as Array<{ t?: number | string; o?: number; h?: number; l?: number; c?: number; v?: number }>).map((b) => {
            const rawT = b?.t;
            let t = "";
            if (typeof rawT === "number" && Number.isFinite(rawT)) {
              t = new Date(rawT * 1000).toISOString().slice(0, 10);
            } else {
              t = String(rawT ?? "");
            }
            return {
              t,
              o: Number(b?.o ?? b?.c ?? 0),
              h: Number(b?.h ?? b?.c ?? 0),
              l: Number(b?.l ?? b?.c ?? 0),
              c: Number(b?.c ?? 0),
              v: Number(b?.v ?? 0),
            };
          })
        : [];
      const points = pointsFromPayload.length > 0 ? pointsFromPayload : pointsFromBars;

      // Keep modal open even if empty; show a clear message.
      setHistory({ symbol, points });
      if (!points.length) setHistoryErr("No history points returned.");
    } catch (e) {
      console.error(e);
      setHistoryErr("Failed to load price history.");
    } finally {
      setHistoryLoading(false);
    }
  }

  const dateLabel = signals?.date ?? null;
  const up = signals?.top_up ?? [];
  const down = signals?.top_down ?? [];
  const news = signals?.news ?? [];
  const regime = typeof signals?.regime === "string" ? signals?.regime : null;
  const systemCapability =
    signals?.system_capability === "high" ||
    signals?.system_capability === "medium" ||
    signals?.system_capability === "low"
      ? signals.system_capability
      : "low";
  const llmSummaryEnabled = Boolean(signals?.llm_summary_enabled);
  const marketSummary = typeof signals?.market_summary === "string" ? signals.market_summary.trim() : "";
  const refreshStats = signals?.news_refresh_stats;
  const refreshHit = Number.isFinite(Number(refreshStats?.hit)) ? Number(refreshStats?.hit) : 0;
  const refreshMiss = Number.isFinite(Number(refreshStats?.miss)) ? Number(refreshStats?.miss) : 0;
  const refreshDisabled = Number.isFinite(Number(refreshStats?.disabled)) ? Number(refreshStats?.disabled) : 0;
  const mlModelId = typeof signals?.ml_model_id === "string" ? signals.ml_model_id : null;
  const mlWinner = typeof signals?.ml_winner_name === "string" ? signals.ml_winner_name : null;
  const mlLabels = Array.isArray(signals?.ml_feature_labels)
    ? (signals?.ml_feature_labels as string[])
    : [];
  const mlImportances = Array.isArray(signals?.ml_feature_importances)
    ? (signals?.ml_feature_importances as number[])
    : [];

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-semibold tracking-tight">Market radar</h1>
        <p className="max-w-3xl text-sm text-slate-300">
          These are the top ideas from the engine: three names it prefers to accumulate and three it prefers to avoid,
          plus a quick snapshot of market headlines.
        </p>

        <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-slate-300">
          {dateLabel ? (
            <div>
              Signals date: <span className="text-cyan-300">{dateLabel}</span>
            </div>
          ) : null}
          {regime ? (
            <div>
              Regime: <span className="text-slate-200">{regime}</span>
            </div>
          ) : null}
          <div>
            News refresh: <span className="text-slate-200">hit {refreshHit}</span>,{" "}
            <span className="text-slate-200">miss {refreshMiss}</span>,{" "}
            <span className="text-slate-200">disabled {refreshDisabled}</span>
          </div>
        </div>

        <div className="mt-3">
          <Link
            href="/glossary"
            className="inline-flex items-center rounded-lg border border-slate-800 bg-slate-900/60 px-3 py-1 text-xs text-slate-200 hover:bg-slate-900"
          >
            View glossary
          </Link>
        </div>
      </div>

      {err ? (
        <div className="rounded-xl border border-rose-800/60 bg-rose-950/30 px-4 py-3 text-sm text-rose-200">{err}</div>
      ) : null}

      <div className="grid gap-4 md:grid-cols-2">
        <section className="rounded-2xl border border-emerald-800/40 bg-slate-950/30 p-4">
          <h2 className="text-sm font-semibold text-slate-100">Top 3 signals: up</h2>
          <p className="mt-1 text-xs text-slate-300">
            Ranked by the backend signal score. Click a card to open a price chart.
          </p>

          <div className="mt-3 space-y-3">
            {loading ? (
              <div className="text-sm text-slate-300">Loading…</div>
            ) : up.length ? (
              up.map((r) => <RecCard key={`up-${r.symbol}`} rec={r} tone="up" onOpenChart={() => openChart(r.symbol)} />)
            ) : (
              <div className="text-sm text-slate-300">No up signals.</div>
            )}
          </div>
        </section>

        <section className="rounded-2xl border border-rose-800/40 bg-slate-950/30 p-4">
          <h2 className="text-sm font-semibold text-slate-100">Top 3 signals: down</h2>
          <p className="mt-1 text-xs text-slate-300">
            Ranked by the backend signal score. Click a card to open a price chart.
          </p>

          <div className="mt-3 space-y-3">
            {loading ? (
              <div className="text-sm text-slate-300">Loading…</div>
            ) : down.length ? (
              down.map((r) => (
                <RecCard key={`down-${r.symbol}`} rec={r} tone="down" onOpenChart={() => openChart(r.symbol)} />
              ))
            ) : (
              <div className="text-sm text-slate-300">No down signals.</div>
            )}
          </div>
        </section>
      </div>

      <section className="rounded-2xl border border-slate-800 bg-slate-950/30 p-4">
        <h2 className="text-sm font-semibold text-slate-100">Market regime summary</h2>
        <p className="mt-1 text-xs text-slate-300">
          Capability-aware LLM narration. High-tier systems generate this from today&apos;s headlines; lower tiers skip it.
        </p>

        <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-slate-300">
          <div>
            Router tier: <span className="text-slate-200">{systemCapability}</span>
          </div>
          <div>
            LLM enabled: <span className="text-slate-200">{llmSummaryEnabled ? "yes" : "no"}</span>
          </div>
        </div>

        <div className="mt-3 rounded-xl border border-slate-800 bg-slate-900/40 px-4 py-3 text-sm leading-relaxed text-slate-200">
          {loading ? "Loading..." : marketSummary || "Summary not generated for this capability tier."}
        </div>
      </section>

      <section className="rounded-2xl border border-slate-800 bg-slate-950/30 p-4">
        <h2 className="text-sm font-semibold text-slate-100">Financial news snapshot</h2>
        <p className="mt-1 text-xs text-slate-300">
          Financial news of the day. Click to open the source.
        </p>

        <div className="mt-3 space-y-3">
          {loading ? (
            <div className="text-sm text-slate-300">Loading…</div>
          ) : news.length ? (
            news.map((n) => <NewsCard key={n.id} item={n} />)
          ) : (
            <div className="text-sm text-slate-300">No news items.</div>
          )}
        </div>
      </section>

      <section className="rounded-2xl border border-slate-800 bg-slate-950/30 p-4">
        <h2 className="text-sm font-semibold text-slate-100">ML feature importance</h2>
        <p className="mt-1 text-xs text-slate-300">
          Visual explanation of which features mattered most for the currently selected model.
        </p>

        <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-slate-300">
          {mlModelId ? (
            <div>
              Model: <span className="font-mono text-slate-200">{mlModelId}</span>
            </div>
          ) : null}
          {mlWinner ? (
            <div>
              Tournament winner: <span className="text-slate-200">{mlWinner}</span>
            </div>
          ) : null}
        </div>

        <div className="mt-3">
          <FeatureImportanceChart labels={mlLabels} values={mlImportances} />
        </div>
      </section>

      {activeSymbol ? (
        <Modal title={`${activeSymbol} price history`} onClose={() => setActiveSymbol(null)}>
          {historyErr ? (
            <div className="rounded-xl border border-rose-800/60 bg-rose-950/30 px-3 py-2 text-sm text-rose-200">
              {historyErr}
            </div>
          ) : null}

          {historyLoading ? (
            <div className="text-sm text-slate-300">Loading…</div>
          ) : history?.points?.length ? (
            <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-3 text-slate-200">
              <CandlesChart points={history.points} />
            </div>
          ) : (
            <div className="text-sm text-slate-300">
              No chart data to render. If this persists, verify the backend route{" "}
              <span className="text-slate-200">/market/annotated/{activeSymbol}</span> returns a{" "}
              <span className="text-slate-200">points</span> array with <span className="text-slate-200">t</span> and{" "}
              <span className="text-slate-200">c</span>.
            </div>
          )}
        </Modal>
      ) : null}
    </div>
  );
}
