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
  risk_tolerance: "low" | "balanced" | "aggressive";
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
  ml_model_id?: string | null;
  ml_winner_name?: string | null;
  ml_feature_labels?: string[];
  ml_feature_importances?: number[];

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
  roc_auc?: number;
}

export interface MLModelInfo {
  model_id: string;
  algorithm?: string;
  family?: string;
  feature_type?: "numeric" | "text";
  rank?: number;
  score?: number;
  sample_count?: number;
  created_at?: string;
  is_selected?: boolean;
  is_deployed?: boolean;
  underperforming?: boolean;

  model_name?: string;
  model_type?: string;
  trained_on?: string;
  selected?: boolean;
  deployed?: boolean;
  metrics: MLModelMetric;
}

export interface MLModelListResponse {
  items: MLModelInfo[];
}

export interface MLPredictionItem {
  symbol: string;
  prediction: "up" | "down";
  probability_up: number;
}

export interface MLPredictionResponse {
  model_id: string;
  items: MLPredictionItem[];
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

export interface MLTournamentCompetitorStat {
  sharpe: number;
  max_drawdown: number;
  mean_return: number;
  volatility: number;
  sample_count: number;
}

export interface MLTournamentStatsResponse {
  model_id?: string | null;
  algorithm?: string | null;
  winner_name?: string | null;
  competitor_stats: Record<string, MLTournamentCompetitorStat>;
}

export type AccountStatus = "active" | "suspended" | "locked";

export interface AdminUserErrorItem {
  timestamp?: string | null;
  path: string;
  method: string;
  message: string;
}

export interface AdminUserDetail {
  user_id: string;
  email: string;
  role: UserRole | string;
  created_at?: string;
  security: {
    status: AccountStatus;
    failed_login_attempts: number;
    locked_until?: string | null;
  };
  financial_context: {
    risk_tolerance: string;
    horizon_years?: number | null;
    max_drawdown_pct?: number | null;
    constraints: string[];
    custom_universe: string[];
  };
  activity: {
    last_login_at?: string | null;
    last_dashboard_at?: string | null;
  };
  rate_limit: {
    currently_limited?: boolean;
    buckets?: Record<string, unknown>;
  };
  error_logs: AdminUserErrorItem[];
}
