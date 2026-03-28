import type { Customer, CustomerSummary, RiskCategory } from '../types';

export const mockCustomers: Customer[] = [
  {
    customer_id: 'CUST00009817',
    risk_score: 85.5,
    risk_category: 'CRITICAL',
    monthly_income: 95000,
    emi_amount: 28500,
    outstanding_principal: 450000,
    full_name: 'Rahul Sharma',
    city: 'Mumbai',
    city_tier: 'Tier 1',
    employment_type: 'Salaried',
  },
  {
    customer_id: 'CUST00009818',
    risk_score: 72.3,
    risk_category: 'HIGH',
    monthly_income: 75000,
    emi_amount: 22500,
    outstanding_principal: 380000,
    full_name: 'Priya Patel',
    city: 'Delhi',
    city_tier: 'Tier 1',
    employment_type: 'Self Employed',
  },
  {
    customer_id: 'CUST00009819',
    risk_score: 55.8,
    risk_category: 'MEDIUM',
    monthly_income: 65000,
    emi_amount: 19500,
    outstanding_principal: 320000,
    full_name: 'Amit Kumar',
    city: 'Bangalore',
    city_tier: 'Tier 1',
    employment_type: 'Salaried',
  },
  {
    customer_id: 'CUST00009820',
    risk_score: 35.2,
    risk_category: 'LOW',
    monthly_income: 55000,
    emi_amount: 16500,
    outstanding_principal: 280000,
    full_name: 'Sneha Reddy',
    city: 'Hyderabad',
    city_tier: 'Tier 2',
    employment_type: 'Salaried',
  },
  {
    customer_id: 'CUST00009821',
    risk_score: 78.9,
    risk_category: 'HIGH',
    monthly_income: 82000,
    emi_amount: 24600,
    outstanding_principal: 420000,
    full_name: 'Vikram Singh',
    city: 'Chennai',
    city_tier: 'Tier 1',
    employment_type: 'Self Employed',
  },
  {
    customer_id: 'CUST00009822',
    risk_score: 91.2,
    risk_category: 'CRITICAL',
    monthly_income: 55000,
    emi_amount: 22000,
    outstanding_principal: 550000,
    full_name: 'Anjali Gupta',
    city: 'Pune',
    city_tier: 'Tier 2',
    employment_type: 'Salaried',
  },
  {
    customer_id: 'CUST00009823',
    risk_score: 62.4,
    risk_category: 'MEDIUM',
    monthly_income: 70000,
    emi_amount: 21000,
    outstanding_principal: 350000,
    full_name: 'Raj Malhotra',
    city: 'Kolkata',
    city_tier: 'Tier 1',
    employment_type: 'Salaried',
  },
  {
    customer_id: 'CUST00009824',
    risk_score: 28.7,
    risk_category: 'LOW',
    monthly_income: 45000,
    emi_amount: 13500,
    outstanding_principal: 180000,
    full_name: 'Meera Nair',
    city: 'Kochi',
    city_tier: 'Tier 2',
    employment_type: 'Self Employed',
  },
];

export const mockPortfolioSummary: CustomerSummary = {
  total_customers: 10000,
  critical_count: 850,
  high_risk_count: 2200,
  at_risk_count: 4500,
  avg_risk_score: 52.3,
  total_exposure: 4500000000,
  avg_income_high_risk: 78000,
  estimated_savings: 18500000,
};

export function getRiskDistribution(): Record<RiskCategory, number> {
  return {
    LOW: 4500,
    MEDIUM: 2450,
    HIGH: 2200,
    CRITICAL: 850,
  };
}

export function getRiskByEmployment() {
  return [
    { employment_type: 'Salaried', avg_risk: 48.5, count: 6500 },
    { employment_type: 'Self Employed', avg_risk: 58.2, count: 2800 },
    { employment_type: 'Business Owner', avg_risk: 62.1, count: 700 },
  ];
}

export function getRiskByCityTier() {
  return [
    { city_tier: 'Tier 1', avg_risk: 54.2, count: 6000 },
    { city_tier: 'Tier 2', avg_risk: 48.5, count: 3000 },
    { city_tier: 'Tier 3', avg_risk: 42.1, count: 1000 },
  ];
}
