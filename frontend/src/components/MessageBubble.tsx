import React from 'react';
import { User, Bot } from 'lucide-react';

interface MessageBubbleProps {
  role: 'user' | 'bot';
  content: string;
  isTyping?: boolean;
}

export const MessageBubble: React.FC<MessageBubbleProps> = ({ role, content, isTyping }) => {
  return (
    <div className={`message-wrapper ${role}`}>
      <div className="message-content">
        <div className={`avatar ${role}`}>
          {role === 'user' ? <User size={20} /> : <Bot size={20} />}
        </div>
        <div className="message-text">
          {isTyping ? (
            <div className="typing-indicator">
              <div className="typing-dot"></div>
              <div className="typing-dot"></div>
              <div className="typing-dot"></div>
            </div>
          ) : (
            content
          )}
        </div>
      </div>
    </div>
  );
};
