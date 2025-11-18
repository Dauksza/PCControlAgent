import { useMemo } from 'react';
import Sidebar from '@components/Sidebar';
import ChatArea from '@components/ChatArea';
import InputBox from '@components/InputBox';
import { useConversations } from '@hooks/useConversations';

const HomePage = () => {
  const {
    conversations,
    activeThread,
    messages,
    selectThread,
    sendMessage,
    loading,
  } = useConversations();

  const heroCopy = useMemo(
    () => ({
      title: 'Your personal autonomous AI command center',
      subtitle: 'Chat, orchestrate tools, and control your system from one modern workspace.',
    }),
    []
  );

  return (
    <div className="flex h-screen">
      <Sidebar
        conversations={conversations}
        activeThreadId={activeThread?.id}
        onSelectThread={selectThread}
      />
      <main className="flex flex-1 flex-col bg-slate-50 dark:bg-slate-900">
        <div className="border-b border-slate-200 px-6 py-4 dark:border-slate-800">
          <h1 className="text-2xl font-semibold text-slate-900 dark:text-white">{heroCopy.title}</h1>
          <p className="text-sm text-slate-500 dark:text-slate-400">{heroCopy.subtitle}</p>
        </div>
        <ChatArea messages={messages} loading={loading} />
        <InputBox onSend={sendMessage} disabled={loading} />
      </main>
    </div>
  );
};

export default HomePage;
