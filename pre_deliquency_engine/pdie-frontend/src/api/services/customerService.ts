import apiClient from '../client';
import type { 
  Customer, 
  CustomerFilters, 
  PaginatedCustomers, 
  CustomerSummary,
  PortfolioStats 
} from '../../types';

const API_BASE = '';

export const customerService = {
  async getCustomers(filters: CustomerFilters = {}): Promise<PaginatedCustomers> {
    const params = new URLSearchParams();
    
    if (filters.search) params.append('search', filters.search);
    if (filters.risk_tiers?.length) {
      params.append('risk_tiers', filters.risk_tiers.join(','));
    }
    if (filters.city_tier) params.append('city_tier', filters.city_tier);
    if (filters.employment_type) params.append('employment_type', filters.employment_type);
    if (filters.page) params.append('page', String(filters.page));
    if (filters.page_size) params.append('page_size', String(filters.page_size));

    const response = await apiClient.get<PaginatedCustomers>(`${API_BASE}/api/v1/customers?${params}`);
    return response.data;
  },

  async getCustomerById(customerId: string): Promise<Customer> {
    const response = await apiClient.get<Customer>(`${API_BASE}/api/v1/customers/${customerId}`);
    return response.data;
  },

  async getPortfolioSummary(): Promise<CustomerSummary> {
    const response = await apiClient.get<CustomerSummary>(`${API_BASE}/api/v1/portfolio/summary`);
    return response.data;
  },

  async getPortfolioStats(): Promise<PortfolioStats> {
    const response = await apiClient.get<PortfolioStats>(`${API_BASE}/api/v1/portfolio/stats`);
    return response.data;
  },

  async getCustomerTransactions(customerId: string): Promise<{ transactions: unknown[]; cash_flow: { month: string; income: number; expenses: number }[] }> {
    const response = await apiClient.get<{ transactions: unknown[]; cash_flow: { month: string; income: number; expenses: number }[] }>(
      `${API_BASE}/api/v1/customers/${customerId}/transactions`
    );
    return response.data;
  },

  async updateCustomerStatus(customerId: string, status: string): Promise<void> {
    await apiClient.patch(`${API_BASE}/api/v1/customers/${customerId}/status`, { status });
  },

  async assignCustomer(customerId: string, analystId: string): Promise<void> {
    await apiClient.post(`${API_BASE}/api/v1/customers/${customerId}/assign`, { analyst_id: analystId });
  },
};
