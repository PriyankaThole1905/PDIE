export type MessageChannel = 'SMS' | 'WHATSAPP' | 'VOICE' | 'EMAIL';

export type RiskTier = 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';

export interface MessageTemplate {
  id: string;
  name: string;
  channel: MessageChannel;
  content: string;
  tone: 'formal' | 'friendly' | 'urgent';
}

export interface GeneratedMessage {
  content: string;
  channel: MessageChannel;
  confidence: number;
  reasoning: string;
}

export interface MessageRequest {
  customer_id: string;
  channel: MessageChannel;
  context?: string;
  tone?: 'formal' | 'friendly' | 'urgent';
}

export interface CommunicationLog {
  id: string;
  customer_id: string;
  channel: MessageChannel;
  content: string;
  sent_at: string;
  status: 'SENT' | 'DELIVERED' | 'FAILED';
}

export interface AgentTask {
  id: string;
  customer_id: string;
  task_type: string;
  status: 'PENDING' | 'RUNNING' | 'COMPLETED' | 'FAILED';
  result?: string;
  created_at: string;
  completed_at?: string;
}
