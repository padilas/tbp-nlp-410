import React, { useState } from 'react'
import { ChatInterface } from './components/ChatInterface'
import { AdminPanel } from './components/AdminPanel'
import './index.css'

function App() {
  const [view, setView] = useState<'chat' | 'admin'>('chat');

  return (
    <>
      {/* Tombol Rahasia di Pojok Kanan Atas */}
      <button 
        onClick={() => setView(view === 'chat' ? 'admin' : 'chat')}
        className="switch-view-btn"
      >
        {view === 'chat' ? 'Go to Admin' : 'Go to Chat'}
      </button>

      {view === 'chat' ? <ChatInterface /> : <AdminPanel />}
    </>
  )
}

export default App
