import { FormEvent, useState, KeyboardEvent } from 'react';

interface Props {
  onSend: (content: string) => Promise<void>;
  disabled?: boolean;
}

const InputBox = ({ onSend, disabled }: Props) => {
  const [value, setValue] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleSubmit = async (event?: FormEvent) => {
    if (event) event.preventDefault();
    if (!value.trim() || isSubmitting || disabled) return;
    
    setIsSubmitting(true);
    try {
      await onSend(value.trim());
      setValue('');
    } catch (error) {
      console.error('Error sending message:', error);
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      handleSubmit();
    }
  };

  return (
    <form onSubmit={handleSubmit} className="border-t p-4 bg-white">
      <div className="flex gap-2">
        <textarea
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Type your message... (Press Enter to send, Shift+Enter for new line)"
          className="flex-1 p-3 border rounded-lg resize-none focus:outline-none focus:ring-2 focus:ring-blue-500"
          rows={3}
          disabled={disabled || isSubmitting}
        />
        <button
          type="submit"
          disabled={disabled || isSubmitting || !value.trim()}
          className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed font-medium"
        >
          {isSubmitting ? 'Sending...' : 'Send'}
        </button>
      </div>
    </form>
  );
};

export default InputBox;