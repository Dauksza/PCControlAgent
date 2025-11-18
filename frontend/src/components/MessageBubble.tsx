import type { ChatMessage } from '@app-types/chat';
import clsx from 'clsx';

interface Props {
  message: ChatMessage;
}

const MessageBubble = ({ message }: Props) => {
  const isAssistant = message.role !== 'user';

  return (
    <div
      className={clsx('flex w-full', {
        'justify-end': !isAssistant,
        'justify-start': isAssistant,
      })}
    >
      <div
        className={clsx(
          'max-w-3xl rounded-3xl px-5 py-3 text-sm shadow-card transition',
          isAssistant
            ? 'bg-white text-slate-900 dark:bg-slate-800 dark:text-slate-50'
            : 'bg-primary text-white'
        )}
      >
        <p className="whitespace-pre-line leading-relaxed">{message.content}</p>
        {message.toolCalls && message.toolCalls.length > 0 && (
          <div className="mt-3 space-y-2 text-xs text-slate-500 dark:text-slate-300">
            {message.toolCalls.map((call) => (
              <div key={call.id} className="rounded-xl bg-slate-100 p-2 dark:bg-slate-900">
                <span className="font-semibold">{call.name}</span>
                <pre className="mt-1 overflow-x-auto text-[11px]">{JSON.stringify(call.arguments, null, 2)}</pre>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default MessageBubble;
