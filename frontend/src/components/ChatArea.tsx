import MessageBubble from './MessageBubble';
import type { ChatMessage } from '@app-types/chat';

interface ChatAreaProps {
  messages: ChatMessage[];
  loading: boolean;
}

const ChatArea = ({ messages, loading }: ChatAreaProps) => {
  return (
    <section className="flex flex-1 flex-col space-y-4 overflow-y-auto px-8 py-6 scrollbar-hide">
      {messages.length === 0 && (
        <div className="mx-auto max-w-2xl rounded-2xl border border-dashed border-slate-300 bg-white/40 p-8 text-center text-slate-400 shadow-inner dark:border-slate-700 dark:bg-slate-800/40">
          <p className="text-lg font-medium">Ask anything, or run an automated workflow.</p>
          <p className="text-sm">The agent will plan, use tools, and report progress in real-time.</p>
        </div>
      )}

      {messages.map((message) => (
        <MessageBubble key={message.id} message={message} />
      ))}

      {loading && <div className="text-sm text-slate-500">Thinking...</div>}
    </section>
  );
};

export default ChatArea;
