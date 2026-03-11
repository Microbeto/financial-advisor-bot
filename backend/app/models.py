from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


Role = Literal["user", "premium", "admin", "manager"]


class RiskToleranceEnum(str, Enum):
    LOW = "low"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"


RiskTolerance = RiskToleranceEnum
Regime = Literal["risk_on", "risk_off", "neutral"]
TrendDirection = Literal["up", "down"]


class UserPublic(BaseModel):
    user_id: str
    email: str
    role: Role
    created_at: datetime


class AuthResponse(BaseModel):
    user_id: str
    email: str
    role: Role
    token: str


class LoginRequest(BaseModel):
    email: str
    password: str


class RegisterRequest(BaseModel):
    email: str
    password: str


class RoleUpdateRequest(BaseModel):
    role: Role


AccountStatus = Literal["active", "suspended", "locked"]


class AdminUserStatusUpdateRequest(BaseModel):
    status: AccountStatus
    lock_minutes: int = 30


class AdminUserErrorItem(BaseModel):
    timestamp: Optional[datetime] = None
    path: str = ""
    method: str = ""
    message: str = ""


class AdminUserActivity(BaseModel):
    last_login_at: Optional[datetime] = None
    last_dashboard_at: Optional[datetime] = None


class AdminUserSecurity(BaseModel):
    status: AccountStatus = "active"
    failed_login_attempts: int = 0
    locked_until: Optional[datetime] = None


class AdminUserFinancialContext(BaseModel):
    risk_tolerance: RiskTolerance = RiskTolerance.BALANCED
    horizon_years: Optional[int] = None
    max_drawdown_pct: Optional[float] = None
    constraints: List[str] = Field(default_factory=list)
    custom_universe: List[str] = Field(default_factory=list)


class AdminUserDetailResponse(BaseModel):
    user_id: str
    email: str
    role: Role
    created_at: datetime
    security: AdminUserSecurity = Field(default_factory=AdminUserSecurity)
    financial_context: AdminUserFinancialContext = Field(default_factory=AdminUserFinancialContext)
    activity: AdminUserActivity = Field(default_factory=AdminUserActivity)
    rate_limit: Dict[str, Any] = Field(default_factory=dict)
    error_logs: List[AdminUserErrorItem] = Field(default_factory=list)


class AdminUserStateResetResponse(BaseModel):
    ok: bool = True
    cleared_daily_signals: int = 0
    cleared_dashboard_cache: int = 0


class AdminUserExportResponse(BaseModel):
    user: Dict[str, Any] = Field(default_factory=dict)
    risk_profile: Dict[str, Any] = Field(default_factory=dict)
    portfolio: Dict[str, Any] = Field(default_factory=dict)
    custom_universe: List[str] = Field(default_factory=list)
    daily_signals: List[Dict[str, Any]] = Field(default_factory=list)
    dashboard_cache: List[Dict[str, Any]] = Field(default_factory=list)
    error_logs: List[Dict[str, Any]] = Field(default_factory=list)


class UserListResponse(BaseModel):
    items: List[UserPublic] = Field(default_factory=list)


class RateLimitUpdateRequest(BaseModel):
    key: str
    per_minute: int


class UniverseUpdateRequest(BaseModel):
    key: Literal["sp500", "dow"]
    symbols: List[str] = Field(default_factory=list)


class CustomUniverseRequest(BaseModel):
    symbols: List[str] = Field(default_factory=list)


class PolicyConstraintRequest(BaseModel):
    add: List[str] = Field(default_factory=list)
    remove: List[str] = Field(default_factory=list)


class PolicyConstraintResponse(BaseModel):
    constraints: List[str] = Field(default_factory=list)


class GlossaryTermUpdateRequest(BaseModel):
    key: str
    value: str


class RiskProfile(BaseModel):
    user_id: Optional[str] = None
    client_name: str = ""
    age: Optional[int] = None
    horizon_years: Optional[int] = None
    risk_tolerance: RiskTolerance = "balanced"
    max_drawdown_pct: Optional[float] = None
    income_stability: str = ""
    constraints: List[str] = Field(default_factory=list)
    updated_at: Optional[datetime] = None


class AdminBasketDecision(BaseModel):
    date: str
    candidates: List[str] = Field(default_factory=list)
    selected: List[str] = Field(default_factory=list)


class NewsItem(BaseModel):
    """
    News items support two modes:
    - symbol-linked items (symbol="AAPL")
    - market-wide items (symbol may be None)
    """
    id: str
    symbol: Optional[str] = None
    title: str
    url: str
    publisher: str = ""
    published_at: Optional[datetime] = None
    score: float = 0.0
    tags: List[str] = Field(default_factory=list)
    summary: Optional[str] = None  # needed by frontend


class SignalItem(BaseModel):
    symbol: str
    name: Optional[str] = None  # full company/ETF name for UI
    direction: TrendDirection
    score: float
    reason: str
    link: Optional[str] = None


class SignalsToday(BaseModel):
    date: str
    top_up: List[SignalItem] = Field(default_factory=list)
    top_down: List[SignalItem] = Field(default_factory=list)
    news: List[NewsItem] = Field(default_factory=list)
    universe: List[str] = Field(default_factory=list)
    regime: Regime = "neutral"
    system_capability: Literal["low", "medium", "high"] = "low"
    llm_summary_enabled: bool = False
    market_summary: Optional[str] = None
    ml_model_id: Optional[str] = None
    ml_winner_name: Optional[str] = None
    ml_feature_labels: List[str] = Field(default_factory=list)
    ml_feature_importances: List[float] = Field(default_factory=list)


class MarketBar(BaseModel):
    t: int
    o: float
    h: float
    l: float
    c: float
    v: float


class MarketHistoryResponse(BaseModel):
    symbol: str
    bars: List[MarketBar] = Field(default_factory=list)


class Holding(BaseModel):
    symbol: str
    quantity: float = 0.0
    avg_price: float = 0.0


class PortfolioSnapshot(BaseModel):
    cash: float = 0.0
    holdings: List[Holding] = Field(default_factory=list)


class Annotation(BaseModel):
    t: int
    direction: TrendDirection
    title: str
    url: str
    why: str


class AnnotatedPoint(BaseModel):
    t: str
    o: Optional[float] = None
    h: Optional[float] = None
    l: Optional[float] = None
    c: float
    v: Optional[float] = None


class AnnotatedHistoryResponse(BaseModel):
    symbol: str
    points: List[AnnotatedPoint] = Field(default_factory=list)
    news_markers: List[Dict[str, Any]] = Field(default_factory=list)
    bars: List[MarketBar] = Field(default_factory=list)
    annotations: List[Annotation] = Field(default_factory=list)


class TrendItem(BaseModel):
    symbol: str
    name: Optional[str] = None  # full company/ETF name for UI
    direction: TrendDirection
    score: float
    mom_5d: float
    mom_20d: float
    vol_20d: float
    reason: str
    link: Optional[str] = None


class DashboardResponse(BaseModel):
    date: str
    regime: Regime
    sp500_up: List[TrendItem] = Field(default_factory=list)
    sp500_down: List[TrendItem] = Field(default_factory=list)
    dow_up: List[TrendItem] = Field(default_factory=list)
    dow_down: List[TrendItem] = Field(default_factory=list)
    top_news: List[NewsItem] = Field(default_factory=list)
    system_capability: Literal["low", "medium", "high"] = "low"
    llm_summary_enabled: bool = False
    market_summary: Optional[str] = None
    glossary: Dict[str, str] = Field(default_factory=dict)


class GlossaryResponse(BaseModel):
    terms: Dict[str, str]


class GlossaryExplainRequest(BaseModel):
    term: str


class GlossaryExplainResponse(BaseModel):
    term: str
    definition: str
    generated: bool = False
    source: Literal["existing", "llm", "fallback"] = "existing"


class MLTrainingRequest(BaseModel):
    stock_basket: List[str] = Field(default_factory=list)
    lookback_days: int = 240
    label_horizon_days: int = 5
    test_size: float = 0.25
    random_seed: int = 42
    run_async: bool = True


class MLModelMetric(BaseModel):
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    roc_auc: float = 0.0


class MLModelInfo(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_id: str
    algorithm: str
    family: str
    feature_type: Literal["numeric", "text"] = "numeric"
    nlp_pipeline: Optional[Literal["lexicon", "finbert", "cascade"]] = None
    metrics: MLModelMetric = Field(default_factory=MLModelMetric)
    rank: int = 0
    score: float = 0.0
    sample_count: int = 0
    is_selected: bool = False
    is_deployed: bool = False
    underperforming: bool = False
    news_coverage_ratio: Optional[float] = None
    created_at: Optional[datetime] = None


class MLTrainResponse(BaseModel):
    run_id: str
    trained_models: List[MLModelInfo] = Field(default_factory=list)
    selected_model_id: Optional[str] = None
    deleted_underperforming: List[str] = Field(default_factory=list)


class MLModelListResponse(BaseModel):
    items: List[MLModelInfo] = Field(default_factory=list)


class MLSelectionRequest(BaseModel):
    top_k: int = 1
    nlp_pipeline: Optional[Literal["lexicon", "finbert", "cascade"]] = None


class MLSelectionResponse(BaseModel):
    selected_model_id: Optional[str] = None
    ranked_model_ids: List[str] = Field(default_factory=list)


class MLNlpPipelineMigrationRequest(BaseModel):
    dry_run: bool = True
    limit: int = 0
    rollout_iso: Optional[str] = None
    default_pipeline: Literal["lexicon", "finbert", "cascade"] = "finbert"


class MLPruneRequest(BaseModel):
    min_f1: float = 0.45
    min_accuracy: float = 0.45


class MLPruneResponse(BaseModel):
    deleted_model_ids: List[str] = Field(default_factory=list)


class MLTournamentCompetitorStat(BaseModel):
    sharpe: float = 0.0
    max_drawdown: float = 0.0
    mean_return: float = 0.0
    volatility: float = 0.0
    sample_count: float = 0.0


class MLTournamentStatsResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_id: Optional[str] = None
    algorithm: Optional[str] = None
    winner_name: Optional[str] = None
    competitor_stats: Dict[str, MLTournamentCompetitorStat] = Field(default_factory=dict)


class MLPredictionRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    stock_basket: List[str] = Field(default_factory=list)
    model_id: Optional[str] = None
    lookback_days: int = 120


class MLPredictionItem(BaseModel):
    symbol: str
    prediction: Literal["up", "down"]
    probability_up: float = 0.5


class MLPredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_id: str
    items: List[MLPredictionItem] = Field(default_factory=list)


class MLDeployRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_id: Optional[str] = None


class MLDeployResponse(BaseModel):
    deployed_model_id: Optional[str] = None


class MLRuntimeSettingsUpdateRequest(BaseModel):
    tp_barrier: Optional[float] = None
    sl_barrier: Optional[float] = None
    time_barrier_days: Optional[int] = None
    walk_forward_splits: Optional[int] = None
    walk_forward_min_train: Optional[int] = None


class MLRuntimeSettingsResponse(BaseModel):
    tp_barrier: float
    sl_barrier: float
    time_barrier_days: int
    walk_forward_splits: int
    walk_forward_min_train: int
