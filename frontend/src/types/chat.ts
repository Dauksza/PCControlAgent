export interface ConversationThread {
  id: string;
  conversation_id: string;
  name: string;
  summary?: string | null;
  created_at: string;
  updated_at: string;
}

export interface Conversation {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  threads: ConversationThread[];
}

export interface ToolCallPreview {
  id: string;
  name: string;
  arguments: Record<string, unknown>;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  createdAt: string;
  toolCalls?: ToolCallPreview[];
}
