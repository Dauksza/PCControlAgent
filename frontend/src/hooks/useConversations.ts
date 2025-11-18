import { useEffect, useState, useCallback } from 'react';
import axios from 'axios';
import type { Conversation, ConversationThread, ChatMessage } from '@/types';

const API_URL = 'http://localhost:8000';

interface BootstrapResponse {
  conversations: Conversation[];
  activeThread?: ConversationThread;
  hasApiKey: boolean;
  models: string[];
}

const EMPTY_MESSAGE: ChatMessage[] = [];

export const useConversations = () => {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeThread, setActiveThread] = useState<ConversationThread | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>(EMPTY_MESSAGE);
  const [loading, setLoading] = useState(false);
  const [hasApiKey, setHasApiKey] = useState(true);
  const [models, setModels] = useState<string[]>([]);

  useEffect(() => {
    const bootstrap = async () => {
      try {
        const response = await axios.get<BootstrapResponse>(`${API_URL}/api/conversations/bootstrap`);
        setConversations(response.data.conversations || []);
        setHasApiKey(response.data.hasApiKey);
        setModels(response.data.models || []);
        
        if (response.data.activeThread) {
          setActiveThread(response.data.activeThread);
          setMessages(response.data.activeThread.messages || []);
        }
      } catch (error) {
        console.error('Failed to bootstrap:', error);
      }
    };
    bootstrap();
  }, []);

  const selectThread = useCallback((thread: ConversationThread) => {
    setActiveThread(thread);
    setMessages(thread.messages || []);
  }, []);

  const sendMessage = useCallback(
    async (content: string) => {
      if (!activeThread) {
        console.error('No active thread');
        return;
      }

      const userMessage: ChatMessage = {
        id: crypto.randomUUID(),
        role: 'user',
        content,
        createdAt: new Date().toISOString(),
      };

      setMessages((prev) => [...prev, userMessage]);
      setLoading(true);

      try {
        const response = await axios.post<ChatMessage>(
          `${API_URL}/api/chat`,
          {
            threadId: activeThread.id,
            conversationId: activeThread.conversationId,
            content,
          },
          {
            headers: {
              'Content-Type': 'application/json',
            },
          }
        );

        setMessages((prev) => [...prev, response.data]);

        setConversations((prev) =>
          prev.map((conv) =>
            conv.id === activeThread.conversationId
              ? { ...conv, updatedAt: new Date().toISOString() }
              : conv
          )
        );
      } catch (error) {
        console.error('Failed to send message:', error);
        
        const errorMessage: ChatMessage = {
          id: crypto.randomUUID(),
          role: 'assistant',
          content: 'Sorry, I encountered an error. Please try again.',
          createdAt: new Date().toISOString(),
        };
        setMessages((prev) => [...prev, errorMessage]);
      } finally {
        setLoading(false);
      }
    },
    [activeThread]
  );

  return {
    conversations,
    activeThread,
    messages,
    loading,
    sendMessage,
    selectThread,
    hasApiKey,
    models,
  };
};
