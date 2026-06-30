import React, { useState, useRef, useEffect } from 'react';
import { Send } from 'lucide-react';
import { MessageBubble } from './MessageBubble';

interface Message {
  id: string;
  role: 'user' | 'bot';
  content: string;
}

export const ChatInterface: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      role: 'bot',
      content: 'Halo! Saya adalah asisten AI Anda. Ada yang bisa saya bantu terkait dokumen yang Anda miliki?'
    }
  ]);
  const [inputValue, setInputValue] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isTyping]);

  const handleSendMessage = async () => {
    if (!inputValue.trim()) return;

    const newUserMsg: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: inputValue
    };

    setMessages(prev => [...prev, newUserMsg]);
    setInputValue('');
    setIsTyping(true);

    // Persiapkan history chat untuk dikirim (ambil 5 chat terakhir agar konteks tidak terlalu berat)
    const historyToSend = messages.slice(-5).map(m => ({
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
          question: inputValue,
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
        content: data.answer || "Maaf, respon kosong dari AI."
      };
      setMessages(prev => [...prev, newBotMsg]);
    } catch (error) {
      console.error('Error fetching chat:', error);
      const errorMsg: Message = {
        id: (Date.now() + 1).toString(),
        role: 'bot',
        content: 'Maaf, terjadi kesalahan atau server RAG belum menyala. Pastikan backend (main.py) sudah dijalankan.'
      };
      setMessages(prev => [...prev, errorMsg]);
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
        <button className="sidebar-new-chat" onClick={() => setMessages([])}>
          <svg stroke="currentColor" fill="none" strokeWidth="2" viewBox="0 0 24 24" strokeLinecap="round" strokeLinejoin="round" height="16" width="16" xmlns="http://www.w3.org/2000/svg"><line x1="12" y1="5" x2="12" y2="19"></line><line x1="5" y1="12" x2="19" y2="12"></line></svg>
          Chat Baru
        </button>
        
        <div className="sidebar-history">
          {/* History Dummy */}
          <div className="history-item">Penjelasan dokumen RAG</div>
          <div className="history-item">Analisis performa model</div>
          <div className="history-item">Pertanyaan tentang BAB 1</div>
        </div>
      </aside>

      <main className="main-content">
        <header className="chat-header">
          RAG Chatbot Assistant
        </header>
        
        <div className="chat-messages">
          {messages.map((msg) => (
            <MessageBubble key={msg.id} role={msg.role} content={msg.content} />
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
