import apiClient from '../client';
import type { 
  SimulationRequest, 
  SimulationResponse, 
  Offer 
} from '../../types';

export const pathwayService = {
  async simulate(request: SimulationRequest): Promise<SimulationResponse> {
    const response = await apiClient.post<SimulationResponse>('/simulate', request);
    return response.data;
  },

  async createOffer(customerId: string, pathway: string, simulationId: string): Promise<Offer> {
    const response = await apiClient.post<Offer>('/offer', {
      customer_id: customerId,
      pathway,
      simulation_id: simulationId,
    });
    return response.data;
  },

  async getAuditLog(simulationId: string): Promise<Record<string, unknown>> {
    const response = await apiClient.get<Record<string, unknown>>(`/audit/${simulationId}`);
    return response.data;
  },
};
