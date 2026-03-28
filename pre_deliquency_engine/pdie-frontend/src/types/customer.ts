export type RiskCategory = 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';

export interface Customer {
  customer_id: string;
  risk_score: number;
  risk_category: RiskCategory;
  monthly_income: number;
  emi_amount: number;
  outstanding_principal: number;
  full_name?: string;
  city?: string;
  city_tier?: string;
  employment_type?: string;
  account_opening_date?: string;
  emi_day_of_month?: number;
  loan_type?: string;
  interest_rate?: number;
  remaining_months?: number;
  loan_status?: string;
  
  // Risk factors
  salary_delay_days?: number;
  savings_drawdown_rate_4w?: number;
  upi_lending_app_txn_count_30d?: number;
  bill_payment_delay_max?: number;
  atm_withdrawal_spike_pct?: number;
  emi_to_income_ratio?: number;
  emergency_fund_days?: number;
  current_savings?: number;
}

export interface CustomerSummary {
  total_customers: number;
  critical_count: number;
  high_risk_count: number;
  at_risk_count: number;
  avg_risk_score: number;
  total_exposure: number;
  avg_income_high_risk: number;
  estimated_savings: number;
}

export interface PortfolioStats {
  risk_distribution: Record<RiskCategory, number>;
  risk_by_employment: { employment_type: string; avg_risk: number; count: number }[];
  risk_by_city_tier: { city_tier: string; avg_risk: number; count: number }[];
}

export interface CustomerFilters {
  search?: string;
  risk_tiers?: RiskCategory[];
  city_tier?: string;
  employment_type?: string;
  page?: number;
  page_size?: number;
}

export interface PaginatedCustomers {
  customers: Customer[];
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
}
