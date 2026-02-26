import { useState } from 'react';
import { Search, Zap, Clock } from 'lucide-react';

interface RouteSearchProps {
  onSearch: (sentence: string, useRag: boolean, metric: 'time' | 'distance') => void;
  loading: boolean;
}

function RouteSearch({ onSearch, loading }: RouteSearchProps) {
  const [sentence, setSentence] = useState('');
  const [useRag, setUseRag] = useState(true);
  const [metric, setMetric] = useState<'time' | 'distance'>('time');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (sentence.trim()) {
      onSearch(sentence, useRag, metric);
    }
  };

  const examples = [
    'Je voudrais aller de Paris à Lyon demain',
    'Comment me rendre à Marseille depuis Toulouse ?',
    'Un billet pour Nice départ Bordeaux',
    'Je dois aller de Lille à Strasbourg',
  ];

  return (
    <div className="route-search">
      <form onSubmit={handleSubmit}>
        <div className="search-input-wrapper">
          <Search size={20} className="search-icon" />
          <input
            type="text"
            className="search-input"
            placeholder="Exemple : Je veux aller de Paris à Lyon..."
            value={sentence}
            onChange={(e) => setSentence(e.target.value)}
            disabled={loading}
          />
        </div>

        <div className="options-grid">
          <div className="option-group">
            <label className="option-label">
              <input
                type="checkbox"
                checked={useRag}
                onChange={(e) => setUseRag(e.target.checked)}
                disabled={loading}
              />
              <Zap size={16} />
              <span>Utiliser RAG (recommandé)</span>
            </label>
            <p className="option-help">
              ChromaDB RAG améliore la précision de 75% à 100%
            </p>
          </div>

          <div className="option-group">
            <label className="metric-label">Optimisation :</label>
            <div className="metric-buttons">
              <button
                type="button"
                className={`metric-btn ${metric === 'time' ? 'active' : ''}`}
                onClick={() => setMetric('time')}
                disabled={loading}
              >
                <Clock size={16} />
                Temps
              </button>
              <button
                type="button"
                className={`metric-btn ${metric === 'distance' ? 'active' : ''}`}
                onClick={() => setMetric('distance')}
                disabled={loading}
              >
                Distance
              </button>
            </div>
          </div>
        </div>

        <button
          type="submit"
          className="submit-button"
          disabled={loading || !sentence.trim()}
        >
          {loading ? 'Recherche en cours...' : 'Rechercher l\'itinéraire'}
        </button>
      </form>

      <div className="examples">
        <p className="examples-title">Exemples :</p>
        <div className="examples-grid">
          {examples.map((example, index) => (
            <button
              key={index}
              className="example-chip"
              onClick={() => setSentence(example)}
              disabled={loading}
            >
              {example}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

export default RouteSearch;
