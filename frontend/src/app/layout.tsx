"use client";

import "./globals.css";
import type { ReactNode } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { AuthProvider, useAuth } from "@/lib/auth-context";
import { StockBackground } from "@/components/stock-background";

function SettingsIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg
      viewBox="0 0 24 24"
      aria-hidden="true"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.9"
      strokeLinecap="round"
      strokeLinejoin="round"
      {...props}
    >
      <path
        d="M14.5 3.5 15 5.2a1 1 0 0 0 .8.7l1.8.3a1 1 0 0 1 .7.4l1.2 1.6a1 1 0 0 1 .1.8l-.7 1.7a1 1 0 0 0 0 .9l.7 1.7a1 1 0 0 1-.1.8l-1.2 1.6a1 1 0 0 1-.7.4l-1.8.3a1 1 0 0 0-.8.7l-.5 1.7a1 1 0 0 1-.6.6l-1.7.5a1 1 0 0 1-.8-.1l-1.6-1.2a1 1 0 0 0-.9 0l-1.6 1.2a1 1 0 0 1-.8.1l-1.7-.5a1 1 0 0 1-.6-.6l-.5-1.7a1 1 0 0 0-.8-.7l-1.8-.3a1 1 0 0 1-.7-.4l-1.2-1.6a1 1 0 0 1-.1-.8l.7-1.7a1 1 0 0 0 0-.9l-.7-1.7a1 1 0 0 1 .1-.8l1.2-1.6a1 1 0 0 1 .7-.4l1.8-.3a1 1 0 0 0 .8-.7l.5-1.7a1 1 0 0 1 .6-.6l1.7-.5a1 1 0 0 1 .8.1l1.6 1.2a1 1 0 0 0 .9 0l1.6-1.2a1 1 0 0 1 .8-.1l1.7.5a1 1 0 0 1 .6.6Z"
      />
      <circle cx="12" cy="12" r="3" />
    </svg>
  );
}

function TopNav() {
  const pathname = usePathname();
  const { role, isSignedIn, signOut } = useAuth();

  const mainItems = [
    { href: "/dashboard", label: "Dashboard" },
    { href: "/risk-profile", label: "Risk Profile" },
    { href: "/portfolio", label: "Portfolio" },
    { href: "/goals", label: "Goals" },
    { href: "/audit-log", label: "Audit Log" },
  ];

  return (
    <header className="glass-backdrop border-b border-slate-800 bg-slate-950/90">
      <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-4">
        <Link
          href={isSignedIn ? "/dashboard" : "/"}
          className="text-base font-semibold tracking-tight text-slate-50"
        >
          Advisor Bot
        </Link>

        <nav className="flex items-center gap-4 text-sm">
          {/* After sign-in: full app nav */}
          {isSignedIn && (
            <>
              {mainItems.map((item) => {
                const active = pathname === item.href;
                return (
                  <Link
                    key={item.href}
                    href={item.href}
                    className={[
                      "relative rounded-full px-4 py-2 font-medium transition-colors",
                      active
                        ? "bg-sky-600 text-white shadow-sm"
                        : "text-slate-200 hover:bg-slate-800 hover:text-slate-50",
                    ].join(" ")}
                  >
                    {item.label}
                    {active && (
                      <span className="absolute inset-x-3 -bottom-1 h-[2px] rounded-full bg-cyan-300" />
                    )}
                  </Link>
                );
              })}

              {/* Settings / preferences */}
              <Link
                href="/settings"
                aria-label="Preferences"
                className={[
                  "flex h-9 w-9 items-center justify-center rounded-full border text-slate-200 transition-colors",
                  pathname === "/settings"
                    ? "border-sky-600 bg-slate-800"
                    : "border-slate-700 bg-slate-900 hover:border-sky-500 hover:bg-slate-800",
                ].join(" ")}
              >
                <SettingsIcon className="h-5 w-5" />
              </Link>

              {/* Admin panel link: only if signed in AND role is admin */}
              {isSignedIn && role === "admin" && (
                <Link
                  href="/admin"
                  className={[
                    "hidden rounded-full px-3 py-1 text-xs font-medium transition-colors md:inline-block",
                    pathname === "/admin"
                      ? "bg-amber-500/20 text-amber-200 border border-amber-400/60"
                      : "border border-slate-700 text-slate-300 hover:border-amber-400/60 hover:text-amber-200",
                  ].join(" ")}
                >
                  Admin panel
                </Link>
              )}

              <button
                type="button"
                onClick={signOut}
                className="ml-2 rounded-full border border-slate-600 px-4 py-2 text-xs font-medium text-slate-200 hover:bg-slate-800"
              >
                Sign out
              </button>
            </>
          )}
        </nav>
      </div>
    </header>
  );
}

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body className="bg-slate-950 text-slate-50">
        <AuthProvider>
          <StockBackground />
          <TopNav />
          <main className="mx-auto max-w-6xl px-6 py-6">{children}</main>
        </AuthProvider>
      </body>
    </html>
  );
}
