import { useState } from 'react';
import { Search, Cpu, GitFork, Clock, Leaf } from 'lucide-react';
import type { EngineInfo } from '../types';

interface RouteSearchProps {
  onSearch: (
    sentence: string,
    nlpEngine: string,
    graphEngine: string,
    optimization: 'time' | 'pollution',
    trainTypes?: string[]
  ) => void;
  loading: boolean;
  nlpEngines: EngineInfo[];
  graphEngines: EngineInfo[];
}

const TRAIN_TYPES = [
  'TGV INOUI', 'OUIGO', 'TER', 'INTERCITES', 'INTERCITES de nuit',
  'Lyria', 'ICE', 'TramTrain', 'Car TER', 'Navette',
];

function RouteSearch({ onSearch, loading, nlpEngines, graphEngines }: RouteSearchProps) {
  const [sentence, setSentence] = useState('');
  const [nlpEngine, setNlpEngine] = useState('rule-based');
  const [graphEngine, setGraphEngine] = useState('dijkstra');
  const [optimization, setOptimization] = useState<'time' | 'pollution'>('time');
  const [selectedTrainTypes, setSelectedTrainTypes] = useState<string[]>([]);
  const [showTrainFilter, setShowTrainFilter] = useState(false);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (sentence.trim()) {
      onSearch(
        sentence,
        nlpEngine,
        graphEngine,
        optimization,
        selectedTrainTypes.length > 0 ? selectedTrainTypes : undefined
      );
    }
  };

  const toggleTrainType = (type: string) => {
    setSelectedTrainTypes(prev =>
      prev.includes(type) ? prev.filter(t => t !== type) : [...prev, type]
    );
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
              <Cpu size={16} />
              <span>Moteur NLP</span>
            </label>
            <select
              className="engine-select"
              value={nlpEngine}
              onChange={(e) => setNlpEngine(e.target.value)}
              disabled={loading}
            >
              {nlpEngines.map(e => (
                <option key={e.name} value={e.name}>{e.name} — {e.description}</option>
              ))}
            </select>
          </div>

          <div className="option-group">
            <label className="option-label">
              <GitFork size={16} />
              <span>Moteur Graph</span>
            </label>
            <select
              className="engine-select"
              value={graphEngine}
              onChange={(e) => setGraphEngine(e.target.value)}
              disabled={loading}
            >
              {graphEngines.map(e => (
                <option key={e.name} value={e.name}>{e.name} — {e.description}</option>
              ))}
            </select>
          </div>

          <div className="option-group">
            <label className="metric-label">Optimisation :</label>
            <div className="metric-buttons">
              <button
                type="button"
                className={`metric-btn ${optimization === 'time' ? 'active' : ''}`}
                onClick={() => setOptimization('time')}
                disabled={loading}
              >
                <Clock size={16} />
                Temps
              </button>
              <button
                type="button"
                className={`metric-btn ${optimization === 'pollution' ? 'active' : ''}`}
                onClick={() => setOptimization('pollution')}
                disabled={loading}
              >
                <Leaf size={16} />
                CO2
              </button>
            </div>
          </div>

          <div className="option-group">
            <button
              type="button"
              className="train-filter-toggle"
              onClick={() => setShowTrainFilter(!showTrainFilter)}
            >
              Filtrer par type de train {selectedTrainTypes.length > 0 && `(${selectedTrainTypes.length})`}
            </button>
            {showTrainFilter && (
              <div className="train-type-grid">
                {TRAIN_TYPES.map(type => (
                  <label key={type} className="train-type-chip">
                    <input
                      type="checkbox"
                      checked={selectedTrainTypes.includes(type)}
                      onChange={() => toggleTrainType(type)}
                    />
                    <span className={`train-badge train-${type.toLowerCase().replace(/\s+/g, '-')}`}>
                      {type}
                    </span>
                  </label>
                ))}
              </div>
            )}
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
