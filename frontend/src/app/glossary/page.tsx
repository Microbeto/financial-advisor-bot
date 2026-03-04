// frontend/src/app/glossary/page.tsx
"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { getGlossary } from "@/lib/api-client";

type GlossaryMap = Record<string, string>;

function normalize(s: string) {
  return (s || "").toLowerCase().trim();
}

export default function GlossaryPage() {
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState<string | null>(null);
  const [terms, setTerms] = useState<GlossaryMap>({});
  const [query, setQuery] = useState("");

  useEffect(() => {
    let cancelled = false;

    async function load() {
      try {
        setLoading(true);
        setErr(null);
        const res = await getGlossary();
        if (!cancelled) {
          setTerms(res?.terms || {});
        }
      } catch (e) {
        console.error(e);
        if (!cancelled) setErr("Unable to load glossary from backend.");
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    load();
    return () => {
      cancelled = true;
    };
  }, []);

  const entries = useMemo(() => {
    const q = normalize(query);
    const all = Object.entries(terms || {}).map(([k, v]) => ({
      term: k,
      def: String(v),
    }));

    if (!q) {
      return all.sort((a, b) => a.term.localeCompare(b.term));
    }

    return all
      .filter(
        (e) =>
          normalize(e.term).includes(q) ||
          normalize(e.def).includes(q)
      )
      .sort((a, b) => a.term.localeCompare(b.term));
  }, [terms, query]);

  return (
    <div className="mx-auto max-w-5xl space-y-6">
      <header className="space-y-2">
        <h1 className="text-xl font-semibold tracking-tight">Financial glossary</h1>
        <p className="max-w-3xl text-sm text-slate-300">
          Plain-language explanations of terms used across the dashboard,
          trend analysis, and news summaries. This list updates automatically
          as new terms appear in daily market coverage.
        </p>

        <div className="flex flex-wrap items-center gap-3">
          <Link
            href="/dashboard"
            className="inline-flex items-center rounded-lg border border-slate-800 bg-slate-900/60 px-3 py-1 text-xs text-slate-200 hover:bg-slate-900"
          >
            Back to dashboard
          </Link>

          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search terms or definitions…"
            className="w-full max-w-sm rounded-lg border border-slate-800 bg-slate-950/60 px-3 py-2 text-sm text-slate-200 placeholder:text-slate-500 focus:outline-none focus:ring-2 focus:ring-cyan-500/40"
          />
        </div>
      </header>

      {err ? (
        <div className="rounded-xl border border-rose-800/60 bg-rose-950/30 px-4 py-3 text-sm text-rose-200">
          {err}
        </div>
      ) : null}

      <section className="rounded-2xl border border-slate-800 bg-slate-950/30 p-4">
        {loading ? (
          <div className="text-sm text-slate-300">Loading glossary…</div>
        ) : entries.length ? (
          <ul className="divide-y divide-slate-800">
            {entries.map(({ term, def }) => (
              <li key={term} className="py-3">
                <div className="text-sm font-semibold text-slate-100">
                  {term}
                </div>
                <div className="mt-1 text-sm leading-relaxed text-slate-300">
                  {def}
                </div>
              </li>
            ))}
          </ul>
        ) : (
          <div className="text-sm text-slate-300">
            No matching terms found.
          </div>
        )}
      </section>

      <section className="rounded-2xl border border-slate-800 bg-slate-950/30 p-4">
        <h2 className="text-sm font-semibold text-slate-100">
          How this glossary is built
        </h2>
        <p className="mt-2 text-sm text-slate-300">
          The glossary combines a permanent core of common financial concepts
          with dynamic terms extracted from daily news summaries and trend
          explanations. When unfamiliar language appears in the dashboard,
          it is added here automatically so future views remain consistent.
        </p>
        <p className="mt-2 text-sm text-slate-300">
          Definitions are intentionally short and practical. They describe how
          the term is used inside this system, not every possible academic
          interpretation.
        </p>
      </section>
    </div>
  );
}
