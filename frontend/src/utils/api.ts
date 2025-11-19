import axios from 'axios';
import type { ChatMessage } from '@app-types/chat';

interface SendChatPayload {
  threadId: string;
  conversationId: string;
  content: string;
  model?: string;
}

export const sendChatMessage = async (payload: SendChatPayload): Promise<{ message: ChatMessage }> => {
  const { data } = await axios.post('/api/frontend/message', {
    conversation_id: payload.conversationId,
    thread_id: payload.threadId,
    content: payload.content,
    model: payload.model,
  });
  return data;
};
