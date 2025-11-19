import { useEffect, useState, useCallback } from 'react';
import axios from 'axios';
import { sendChatMessage } from '@utils/api';
import type { Conversation, ConversationThread, ChatMessage } from '@app-types/chat';

interface BootstrapResponse {
  conversations: Conversation[];
  activeThread?: ConversationThread;
  hasApiKey: boolean;
  models: string[];
}

const EMPTY_MESSAGE: ChatMessage[] = [];

export const useConversations = () => {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeThread, setActiveThread] = useState<ConversationThread | undefined>(undefined);
  const [messages, setMessages] = useState<ChatMessage[]>(EMPTY_MESSAGE);
  const [loading, setLoading] = useState(false);

  const bootstrap = useCallback(async () => {
    const { data } = await axios.get<BootstrapResponse>('/api/frontend/bootstrap');
    setConversations(data.conversations);
    if (data.activeThread) {
      setActiveThread(data.activeThread);
    }
  }, []);

  useEffect(() => {
    bootstrap().catch((error) => {
      console.error('Failed to bootstrap frontend', error);
    });
  }, [bootstrap]);

  const selectThread = useCallback((thread: ConversationThread) => {
    setActiveThread(thread);
    setMessages(EMPTY_MESSAGE);
  }, []);

  const sendMessage = useCallback(
    async (content: string) => {
      if (!activeThread) return;
      const optimistic: ChatMessage = {
        id: crypto.randomUUID(),
        role: 'user',
        content,
        createdAt: new Date().toISOString(),
      };
      setMessages((prev) => [...prev, optimistic]);
      setLoading(true);
      try {
        const response = await sendChatMessage({
          threadId: activeThread.id,
          conversationId: activeThread.conversation_id,
          content,
        });
        setMessages((prev) => [...prev, response.message]);
      } catch (error) {
        console.error('Message failed', error);
      } finally {
        setLoading(false);
      }
    },
    [activeThread]
  );

  return { conversations, activeThread, messages, selectThread, sendMessage, loading };
};
