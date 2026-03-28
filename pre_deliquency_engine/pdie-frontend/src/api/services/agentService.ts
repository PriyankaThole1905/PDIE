import apiClient from '../client';

const API_BASE = '';

export interface GenerateMessageRequest {
  customer_id: string;
  channel: 'SMS' | 'WHATSAPP' | 'VOICE' | 'EMAIL';
  message_type?: 'reminder' | 'awareness' | 'offer';
}

export interface GenerateMessageResponse {
  customer_id: string;
  channel: string;
  content: string;
  confidence: number;
  message_id: string;
}

export interface ScheduleInterventionRequest {
  customer_id: string;
  action_type: 'awareness_sms' | 'recovery_sms' | 'relationship_call';
  message?: string;
  scheduled_date?: string;
}

export interface ScheduleInterventionResponse {
  intervention_id: string;
  customer_id: string;
  action_type: string;
  message: string;
  scheduled_date: string;
  status: string;
  created_at: string;
}

export const agentService = {
  async generateMessage(request: GenerateMessageRequest): Promise<GenerateMessageResponse> {
    const response = await apiClient.post<GenerateMessageResponse>(
      `${API_BASE}/api/v1/ai/generate-message`,
      request
    );
    return response.data;
  },

  async scheduleIntervention(request: ScheduleInterventionRequest): Promise<ScheduleInterventionResponse> {
    const response = await apiClient.post<ScheduleInterventionResponse>(
      `${API_BASE}/api/v1/interventions/schedule`,
      request
    );
    return response.data;
  },
};
