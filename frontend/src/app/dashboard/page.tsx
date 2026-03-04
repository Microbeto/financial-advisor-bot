// frontend/src/app/dashboard/page.tsx
"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import { getDailySignals, getMarketHistory } from "@/lib/api-client";
import type { DailySignals, SignalTopItem, NewsItem } from "@/lib/types";

type HistoryPoint = { t: string; c: number };
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

function Sparkline({ points }: { points: HistoryPoint[] }) {
  const w = 640;
  const h = 220;
  const pad = 16;

  const ys = points.map((p) => p.c);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const spanY = Math.max(1e-9, maxY - minY);

  const toX = (i: number) => pad + (i / Math.max(1, points.length - 1)) * (w - pad * 2);
  const toY = (v: number) => pad + (1 - (v - minY) / spanY) * (h - pad * 2);

  const d = points
    .map((p, i) => `${i === 0 ? "M" : "L"} ${toX(i).toFixed(2)} ${toY(p.c).toFixed(2)}`)
    .join(" ");

  const first = points[0]?.t ?? "";
  const last = points[points.length - 1]?.t ?? "";

  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="w-full">
      <path d={d} fill="none" stroke="currentColor" strokeWidth="2" opacity="0.9" />
      <text x={pad} y={h - 6} fontSize="11" opacity="0.7">
        {first} → {last}
      </text>
      <text x={w - pad} y={pad + 10} fontSize="11" opacity="0.7" textAnchor="end">
        {minY.toFixed(2)} – {maxY.toFixed(2)}
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
      const points = Array.isArray(parsed?.points) ? (parsed.points as HistoryPoint[]) : [];

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
              <Sparkline points={history.points} />
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
