// frontend/src/lib/types.ts

export interface Holding {
  symbol: string;
  quantity: number;
  avg_price: number;
}

export interface PortfolioSnapshot {
  cash: number;
  holdings: Holding[];
}

export interface Recommendation {
  symbol: string;
  action: "BUY" | "SELL" | "HOLD";
  size: number;
  confidence: number;
  rationale: string;
}

export interface RiskProfile {
  client_name: string | null;
  age: number | null;
  horizon_years: number | null;
  risk_tolerance: "conservative" | "moderate" | "aggressive" | string;
  max_drawdown_pct: number | null;
  income_stability: string | null;
  constraints: string[];
}

export interface Goal {
  id: number;
  name: string;
  target_amount: number;
  target_date: string | null;
  priority: "low" | "medium" | "high" | string;
  risk_bucket: "capital_preservation" | "balanced" | "growth" | string;
}

export interface AuditLogEntry {
  id: number;
  timestamp: string;
  description: string;
  num_recommendations: number;

  regime?: string | null;
  user_id?: string | null;
  model_id?: string | null;
}

export type AdminClient = {
  id: string;
  name: string;
  risk_profile: RiskProfile;
  portfolio: PortfolioSnapshot;
};

export interface AdminStats {
  total_clients: number;
  active_clients: number;
  average_portfolio_value: number;
}

/**
 * Auth models
 */
export type UserRole = "user" | "premium" | "admin" | "manager";

export interface UserPublic {
  user_id: string;
  email: string;
  role: UserRole | string;
  created_at?: string;

  [k: string]: unknown;
}

export interface AuthResponse {
  user_id: string;
  email: string;
  role: UserRole | string;
  token: string;
}

export interface UserCreate {
  email: string;
  password: string;
  name?: string;
}

export interface UserLogin {
  email: string;
  password: string;
}

/**
 * Signals + Dashboard types aligned to FastAPI responses.
 */

export type SignalDirection = "up" | "down";

export interface SignalTopItem {
  symbol: string;
  direction: SignalDirection;
  score: number;

  reason?: string | null;
  link?: string | null;

  // allow backend to add more fields without breaking builds
  [k: string]: unknown;
}

/**
 * News items
 * Backend now adds `summary` (3-line summary) when available.
 */
export interface NewsItem {
  id: string;
  symbol?: string | null;
  title: string;
  url: string;

  publisher?: string | null;
  published_at?: string | null;
  score?: number | null;
  tags?: string[];

  // new: optional 3-line summary text from backend
  summary?: string | null;

  [k: string]: unknown;
}

/**
 * /signals/today
 */
export interface DailySignals {
  date: string; // "YYYY-MM-DD"
  top_up: SignalTopItem[];
  top_down: SignalTopItem[];
  news: NewsItem[];
  universe: string[];
  regime?: string | null;

  [k: string]: unknown;
}

/**
 * /dashboard/public (and /dashboard)
 */
export interface DashboardSignalItem {
  symbol: string;
  direction: SignalDirection;
  score: number;

  mom_5d?: number | null;
  mom_20d?: number | null;
  vol_20d?: number | null;

  reason?: string | null;
  link?: string | null;

  [k: string]: unknown;
}

export interface GlossaryTerms {
  [term: string]: string;
}

export interface DashboardResponse {
  date: string; // "YYYY-MM-DD"
  regime?: string | null;

  sp500_up: DashboardSignalItem[];
  sp500_down: DashboardSignalItem[];
  dow_up: DashboardSignalItem[];
  dow_down: DashboardSignalItem[];

  top_news: NewsItem[];
  glossary: GlossaryTerms;

  [k: string]: unknown;
}

/**
 * Market annotated history (backend: /market/annotated/{symbol})
 */
export interface AnnotatedHistoryPoint {
  t: string; // ISO date or YYYY-MM-DD (backend-defined)
  c: number; // close

  // Optional extra annotation fields if backend adds them later:
  // event?: "news" | "earnings" | ...
  // headline?: string
  // url?: string
  [k: string]: unknown;
}

export interface AnnotatedHistoryResponse {
  symbol: string;
  points?: AnnotatedHistoryPoint[];

  [k: string]: unknown;
}

export interface MLModelMetric {
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
}

export interface MLModelInfo {
  model_id: string;
  model_name: string;
  model_type: string;
  trained_on: string;
  selected: boolean;
  deployed: boolean;
  metrics: MLModelMetric;
}

export interface MLModelListResponse {
  items: MLModelInfo[];
}

export interface MLRuntimeSettings {
  tp_barrier: number;
  sl_barrier: number;
  time_barrier_days: number;
  walk_forward_splits: number;
  walk_forward_min_train: number;
}

export interface MLRuntimeSettingsUpdate {
  tp_barrier?: number;
  sl_barrier?: number;
  time_barrier_days?: number;
  walk_forward_splits?: number;
  walk_forward_min_train?: number;
}
