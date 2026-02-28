import { useState } from 'react';
import { Cpu, GitFork, Play } from 'lucide-react';
import { travelOrderAPI } from '../services/api';
import type { NLPBenchmarkResult, GraphBenchmarkResult } from '../types';

function BenchmarkPanel() {
  // NLP benchmark state
  const [nlpText, setNlpText] = useState('Je veux aller de Paris à Lyon');
  const [nlpResults, setNlpResults] = useState<NLPBenchmarkResult[]>([]);
  const [nlpLoading, setNlpLoading] = useState(false);

  // Graph benchmark state
  const [graphDep, setGraphDep] = useState('Paris Gare de Lyon');
  const [graphDest, setGraphDest] = useState('Lyon Part Dieu');
  const [graphResults, setGraphResults] = useState<GraphBenchmarkResult[]>([]);
  const [graphLoading, setGraphLoading] = useState(false);

  const runNLPBenchmark = async () => {
    if (!nlpText.trim()) return;
    setNlpLoading(true);
    try {
      const results = await travelOrderAPI.benchmarkNLP(nlpText);
      setNlpResults(results);
    } catch (err) {
      console.error('NLP benchmark error:', err);
    } finally {
      setNlpLoading(false);
    }
  };

  const runGraphBenchmark = async () => {
    if (!graphDep.trim() || !graphDest.trim()) return;
    setGraphLoading(true);
    try {
      const results = await travelOrderAPI.benchmarkGraph(graphDep, graphDest);
      setGraphResults(results);
    } catch (err) {
      console.error('Graph benchmark error:', err);
    } finally {
      setGraphLoading(false);
    }
  };

  return (
    <div className="benchmark-panel">
      {/* NLP Benchmark */}
      <section className="benchmark-section">
        <h3 className="benchmark-title">
          <Cpu size={20} />
          Benchmark NLP
        </h3>
        <p className="benchmark-desc">
          Compare les 4 moteurs NLP sur le meme texte (extraction de depart/destination).
        </p>
        <div className="benchmark-input-row">
          <input
            type="text"
            className="benchmark-input"
            value={nlpText}
            onChange={(e) => setNlpText(e.target.value)}
            placeholder="Texte en francais..."
            disabled={nlpLoading}
          />
          <button
            className="benchmark-run-btn"
            onClick={runNLPBenchmark}
            disabled={nlpLoading || !nlpText.trim()}
          >
            <Play size={16} />
            {nlpLoading ? 'En cours...' : 'Lancer'}
          </button>
        </div>

        {nlpResults.length > 0 && (
          <div className="benchmark-table-wrapper">
            <table className="benchmark-table">
              <thead>
                <tr>
                  <th>Engine</th>
                  <th>Valide</th>
                  <th>Depart</th>
                  <th>Destination</th>
                  <th className="num">Confiance</th>
                  <th className="num">Temps (ms)</th>
                  <th className="num">Memoire (KB)</th>
                </tr>
              </thead>
              <tbody>
                {nlpResults.map((r, i) => (
                  <tr key={i} className={r.is_valid ? 'row-valid' : 'row-invalid'}>
                    <td className="engine-name">{r.engine_name}</td>
                    <td>{r.is_valid ? '✓' : '✗'}</td>
                    <td>{r.departure || '—'}</td>
                    <td>{r.destination || '—'}</td>
                    <td className="num">{(r.confidence * 100).toFixed(0)}%</td>
                    <td className="num">{r.time_ms.toFixed(1)}</td>
                    <td className="num">{r.memory_kb.toFixed(0)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      {/* Graph Benchmark */}
      <section className="benchmark-section">
        <h3 className="benchmark-title">
          <GitFork size={20} />
          Benchmark Graph
        </h3>
        <p className="benchmark-desc">
          Compare les 3 moteurs graph sur le meme trajet (Dijkstra, A*, BFS).
        </p>
        <div className="benchmark-input-row">
          <input
            type="text"
            className="benchmark-input"
            value={graphDep}
            onChange={(e) => setGraphDep(e.target.value)}
            placeholder="Gare de depart..."
            disabled={graphLoading}
          />
          <input
            type="text"
            className="benchmark-input"
            value={graphDest}
            onChange={(e) => setGraphDest(e.target.value)}
            placeholder="Gare d'arrivee..."
            disabled={graphLoading}
          />
          <button
            className="benchmark-run-btn"
            onClick={runGraphBenchmark}
            disabled={graphLoading || !graphDep.trim() || !graphDest.trim()}
          >
            <Play size={16} />
            {graphLoading ? 'En cours...' : 'Lancer'}
          </button>
        </div>

        {graphResults.length > 0 && (
          <div className="benchmark-table-wrapper">
            <table className="benchmark-table">
              <thead>
                <tr>
                  <th>Engine</th>
                  <th>Trouve</th>
                  <th className="num">Poids total</th>
                  <th className="num">Noeuds explores</th>
                  <th className="num">Temps (ms)</th>
                  <th className="num">Memoire (KB)</th>
                  <th>Route</th>
                </tr>
              </thead>
              <tbody>
                {graphResults.map((r, i) => (
                  <tr key={i} className={r.found ? 'row-valid' : 'row-invalid'}>
                    <td className="engine-name">{r.engine_name}</td>
                    <td>{r.found ? '✓' : '✗'}</td>
                    <td className="num">{r.total_weight.toFixed(1)}</td>
                    <td className="num">{r.nodes_explored}</td>
                    <td className="num">{r.time_ms.toFixed(1)}</td>
                    <td className="num">{r.memory_kb.toFixed(0)}</td>
                    <td className="route-preview">
                      {r.found ? `${r.steps.length} gares` : '—'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  );
}

export default BenchmarkPanel;
