// Authentication guard component that redirects unauthenticated users to sign-in page.
"use client";

import type { ReactNode } from "react";
import Link from "next/link";
import { useAuth } from "@/lib/auth-context";

// Wrapper component that conditionally renders children only when user is signed in.
export function RequireAuth({ children }: { children: ReactNode }) {
  const { isSignedIn } = useAuth();

  if (!isSignedIn) {
    return (
      <main className="flex min-h-[70vh] flex-col items-center justify-center text-slate-50">
        <h1 className="mb-2 text-2xl font-semibold">Sign in required</h1>
        <p className="mb-4 max-w-md text-center text-sm text-slate-300">
          You need to sign in on the home page before you can use the advisor.
        </p>
        <Link
          href="/"
          className="rounded-md border border-slate-600 bg-slate-800 px-4 py-2 text-sm font-medium hover:bg-slate-700"
        >
          Go to sign-in
        </Link>
      </main>
    );
  }

  return <>{children}</>;
}
