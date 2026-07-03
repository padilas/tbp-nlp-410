import React, { useState, useEffect } from 'react';
import { Database, Settings, Search, FileText } from 'lucide-react';

interface Config {
  llm_provider: string;
  openrouter_model: string;
  ollama_model: string;
  chunk_size: number;
  chunk_overlap: number;
  embedding_model: string;
  vector_weight: number;
  bm25_weight: number;
}

interface RetrievedChunk {
  rank: number;
  source: string;
  page: number;
  content: string;
}

export const AdminPanel: React.FC = () => {
  const [config, setConfig] = useState<Config | null>(null);
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<RetrievedChunk[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    fetch('http://localhost:8000/api/admin/config')
      .then(res => res.json())
      .then(data => setConfig(data))
      .catch(err => console.error("Failed to load config", err));
  }, []);

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;
    
    setIsLoading(true);
    setError('');
    try {
      const response = await fetch('http://localhost:8000/api/admin/retrieve', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query })
      });
      
      if (!response.ok) throw new Error("Gagal mengambil data dari server");
      
      const data = await response.json();
      setResults(data.results || []);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="admin-container">
      <header className="admin-header">
        <div className="admin-header-title">
          <Database size={24} />
          <h2>RAG Admin Dashboard</h2>
        </div>
      </header>
      
      <div className="admin-content">
        <div className="admin-sidebar-layout">
          <div className="admin-card">
            <h3><Settings size={18} /> Environment Config</h3>
            {config ? (
              <ul className="config-list">
                <li><strong>LLM Provider:</strong> {config.llm_provider}</li>
                <li><strong>Model (Ollama):</strong> {config.ollama_model}</li>
                <li><strong>Model (Open):</strong> {config.openrouter_model}</li>
                <li><strong>Embedding:</strong> {config.embedding_model}</li>
                <li><strong>Chunk Size:</strong> {config.chunk_size}</li>
                <li><strong>Overlap:</strong> {config.chunk_overlap}</li>
                <li><strong>Vector Wt:</strong> {config.vector_weight}</li>
                <li><strong>BM25 Wt:</strong> {config.bm25_weight}</li>
              </ul>
            ) : (
              <p>Memuat konfigurasi...</p>
            )}
          </div>
        </div>

        <div className="admin-main">
          <div className="admin-card">
            <h3><Search size={18} /> Test Document Retrieval</h3>
            <p className="admin-desc">Masukkan pertanyaan untuk melihat dokumen apa saja yang diambil oleh algoritma Hybrid Search tanpa memanggil AI.</p>
            
            <form onSubmit={handleSearch} className="admin-search-form">
              <input 
                type="text" 
                value={query} 
                onChange={e => setQuery(e.target.value)}
                placeholder="Masukkan query pengetesan..."
                className="admin-input"
              />
              <button type="submit" disabled={isLoading} className="admin-btn">
                {isLoading ? 'Mencari...' : 'Retrieve'}
              </button>
            </form>

            {error && <div className="admin-error">{error}</div>}

            <div className="admin-results">
              {results.length > 0 && <h4>Hasil Pencarian (Top {results.length})</h4>}
              {results.map((res) => (
                <div key={res.rank} className="result-card">
                  <div className="result-header">
                    <span className="result-rank">#{res.rank}</span>
                    <span className="result-source"><FileText size={14} /> {res.source} (Hal. {res.page})</span>
                  </div>
                  <div className="result-content">{res.content}</div>
                </div>
              ))}
              {results.length === 0 && !isLoading && !error && (
                <div className="result-empty">Belum ada pencarian.</div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
