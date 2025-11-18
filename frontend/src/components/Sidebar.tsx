import type { Conversation, ConversationThread } from '@app-types/chat';
import clsx from 'clsx';

interface SidebarProps {
  conversations: Conversation[];
  activeThreadId?: string;
  onSelectThread: (thread: ConversationThread) => void;
}

const Sidebar = ({ conversations, activeThreadId, onSelectThread }: SidebarProps) => {
  return (
    <aside className="w-80 border-r border-slate-200 bg-white/80 backdrop-blur dark:border-slate-800 dark:bg-slate-950/70">
      <div className="flex items-center justify-between px-4 py-3">
        <h2 className="text-sm font-semibold uppercase tracking-wide text-slate-500">Conversations</h2>
        <button className="rounded-full bg-primary px-3 py-1 text-xs font-semibold text-white shadow-card transition hover:bg-primary/90">
          New Chat
        </button>
      </div>
      <div className="space-y-6 overflow-y-auto px-4 pb-6 scrollbar-hide">
        {conversations.map((conversation) => (
          <div key={conversation.id}>
            <p className="text-xs uppercase tracking-wider text-slate-400">{conversation.title}</p>
            <div className="mt-2 space-y-1">
              {conversation.threads.map((thread) => (
                <button
                  key={thread.id}
                  onClick={() => onSelectThread(thread)}
                  className={clsx(
                    'w-full rounded-xl px-3 py-2 text-left text-sm transition',
                    activeThreadId === thread.id
                      ? 'bg-slate-900 text-white shadow-card dark:bg-white dark:text-slate-900'
                      : 'bg-slate-100 text-slate-600 hover:bg-slate-200 dark:bg-slate-900/50 dark:text-slate-300'
                  )}
                >
                  <div className="font-semibold">{thread.name}</div>
                  <div className="text-xs text-slate-400">{new Date(thread.updated_at).toLocaleString()}</div>
                </button>
              ))}
            </div>
          </div>
        ))}
      </div>
    </aside>
  );
};

export default Sidebar;
