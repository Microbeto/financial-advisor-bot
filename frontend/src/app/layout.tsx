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
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      {...props}
    >
      <line x1="4" y1="6" x2="20" y2="6" />
      <circle cx="9" cy="6" r="2" />
      <line x1="4" y1="12" x2="20" y2="12" />
      <circle cx="15" cy="12" r="2" />
      <line x1="4" y1="18" x2="20" y2="18" />
      <circle cx="11" cy="18" r="2" />
    </svg>
  );
}

function TopNav() {
  const pathname = usePathname();
  const { isSignedIn, signOut } = useAuth();

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
