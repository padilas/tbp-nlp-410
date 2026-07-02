import React, { useState, useRef, useEffect } from 'react';
import { Send } from 'lucide-react';
import { MessageBubble } from './MessageBubble';

interface Message {
  id: string;
  role: 'user' | 'bot';
  content: string;
  sources?: { file: string; page: number }[];
}

interface ChatSession {
  id: string;
  title: string;
  messages: Message[];
}

const DEFAULT_MESSAGE: Message = {
  id: '1',
  role: 'bot',
  content: 'Halo! Saya adalah asisten AI Anda. Ada yang bisa saya bantu terkait dokumen yang Anda miliki?'
};

export const ChatInterface: React.FC = () => {
  const [sessions, setSessions] = useState<ChatSession[]>([
    {
      id: Date.now().toString(),
      title: 'Chat Baru',
      messages: [DEFAULT_MESSAGE]
    }
  ]);
  const [activeSessionId, setActiveSessionId] = useState<string>(sessions[0].id);
  const [inputValue, setInputValue] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const activeSession = sessions.find(s => s.id === activeSessionId) || sessions[0];
  const messages = activeSession.messages;

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isTyping]);

  const updateActiveSessionMessages = (newMessages: Message[]) => {
    setSessions(prev => prev.map(s => {
      if (s.id === activeSessionId) {
        let newTitle = s.title;
        // Jika ini adalah pesan pertama dari user, jadikan sebagai judul chat
        if (s.title === 'Chat Baru' && newMessages.length > 1) {
           const firstUser = newMessages.find(m => m.role === 'user');
           if (firstUser) {
             newTitle = firstUser.content.substring(0, 25);
             if (firstUser.content.length > 25) newTitle += '...';
           }
        }
        return { ...s, title: newTitle, messages: newMessages };
      }
      return s;
    }));
  };

  const handleNewChat = () => {
    // Jangan buat baru jika chat saat ini masih kosong (hanya berisi sapaan)
    if (messages.length === 1 && messages[0].id === '1') return;

    const newSession: ChatSession = {
      id: Date.now().toString(),
      title: 'Chat Baru',
      messages: [DEFAULT_MESSAGE]
    };
    setSessions([newSession, ...sessions]);
    setActiveSessionId(newSession.id);
  };

  const handleSendMessage = async () => {
    if (!inputValue.trim()) return;

    const newUserMsg: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: inputValue
    };

    const updatedMessages = [...messages, newUserMsg];
    updateActiveSessionMessages(updatedMessages);
    setInputValue('');
    setIsTyping(true);

    const historyToSend = updatedMessages.slice(-6, -1).map(m => ({
      role: m.role,
      content: m.content
    }));

    try {
      const response = await fetch('http://localhost:8000/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          question: newUserMsg.content,
          history: historyToSend
        }),
      });

      if (!response.ok) {
        throw new Error('Gagal menghubungi server AI');
      }

      const data = await response.json();
      
      const newBotMsg: Message = {
        id: (Date.now() + 1).toString(),
        role: 'bot',
        content: data.answer || "Maaf, respon kosong dari AI.",
        sources: data.sources
      };
      updateActiveSessionMessages([...updatedMessages, newBotMsg]);
    } catch (error) {
      console.error('Error fetching chat:', error);
      const errorMsg: Message = {
        id: (Date.now() + 1).toString(),
        role: 'bot',
        content: 'Maaf, terjadi kesalahan atau server RAG belum menyala. Pastikan backend (main.py) sudah dijalankan.'
      };
      updateActiveSessionMessages([...updatedMessages, errorMsg]);
    } finally {
      setIsTyping(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  return (
    <div className="app-container">
      {/* Sidebar Kiri */}
      <aside className="sidebar">
        <button className="sidebar-new-chat" onClick={handleNewChat}>
          <svg stroke="currentColor" fill="none" strokeWidth="2" viewBox="0 0 24 24" strokeLinecap="round" strokeLinejoin="round" height="16" width="16" xmlns="http://www.w3.org/2000/svg"><line x1="12" y1="5" x2="12" y2="19"></line><line x1="5" y1="12" x2="19" y2="12"></line></svg>
          Chat Baru
        </button>
        
        <div className="sidebar-history">
          {sessions.map(session => (
            <div 
              key={session.id} 
              className={`history-item ${session.id === activeSessionId ? 'active' : ''}`}
              onClick={() => setActiveSessionId(session.id)}
            >
              {session.title}
            </div>
          ))}
        </div>
      </aside>

      <main className="main-content">
        <header className="chat-header">
          RAG Chatbot Assistant
        </header>
        
        <div className="chat-messages">
          {messages.map((msg) => (
            <MessageBubble key={msg.id} role={msg.role} content={msg.content} sources={msg.sources} />
          ))}
          {isTyping && <MessageBubble role="bot" content="" isTyping={true} />}
          <div ref={messagesEndRef} />
        </div>

        <div className="input-container">
          <div className="input-wrapper">
            <textarea
              className="chat-input"
              placeholder="Kirim pesan ke asisten RAG..."
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyDown={handleKeyDown}
              rows={1}
            />
            <button 
              className="send-button" 
              onClick={handleSendMessage}
              disabled={!inputValue.trim() || isTyping}
            >
              <Send size={16} />
            </button>
          </div>
        </div>
      </main>
    </div>
  );
};
