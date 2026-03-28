export interface PathwayResult {
  pathway: string;
  display_name: string;
  npv: number;
  recovery_rate: number;
  acceptance_prob: number;
  churn_reduction: number;
  composite: number;
  new_emi: number;
  new_tenure_months: number;
  description: string;
  action: string;
  immediate_relief: number;
  monthly_savings: number;
  explainability: string;
  short_explanation: string;
  audit?: Record<string, unknown>;
  mc_result?: MonteCarloResult;
  policy_checks?: Record<string, boolean>;
}

export interface MonteCarloResult {
  mean_recovery: number;
  std_recovery: number;
  percentile_5: number;
  percentile_25: number;
  percentile_50: number;
  percentile_75: number;
  percentile_95: number;
  simulations: number;
}

export interface SimulationResponse {
  customer_id: string;
  results: PathwayResult[];
  recommended: string;
  policy_checks: Record<string, boolean>;
  timestamp: string;
}

export interface SimulationRequest {
  customer_id: string;
  pathways?: string[];
  config?: Record<string, unknown>;
  customer?: CustomerInput;
}

export interface CustomerInput {
  customer_id: string;
  monthly_income: number;
  essential_expenses: number;
  loan: {
    principal: number;
    annual_rate: number;
    remaining_months: number;
    emi: number;
  };
  assets: Record<string, number>;
  other_debts?: { type: string; principal: number; rate: number }[];
  risk_band?: string;
  cibil_score?: number;
}

export interface Offer {
  offer_id: string;
  customer_id: string;
  pathway: string;
  status: 'DRAFT' | 'SENT' | 'ACCEPTED' | 'REJECTED';
  created_at: string;
  simulation_reference: string;
}
