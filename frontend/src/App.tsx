import { useState, useEffect, useCallback } from 'react';
import { Train, Activity, Database, Cpu } from 'lucide-react';
import RouteSearch from './components/RouteSearch';
import AudioRecorder from './components/AudioRecorder';
import RouteDisplay from './components/RouteDisplay';
import BenchmarkPanel from './components/BenchmarkPanel';
import HealthStatus from './components/HealthStatus';
import ErrorDisplay from './components/ErrorDisplay';
import { travelOrderAPI } from './services/api';
import type {
  RouteResponse, ResolveResponse, HealthResponse, StatsResponse,
  ApiError, EngineInfo,
} from './types';
import './styles/App.css';

function App() {
  const [routeResult, setRouteResult] = useState<RouteResponse | null>(null);
  const [resolveResult, setResolveResult] = useState<ResolveResponse | null>(null);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [stats, setStats] = useState<StatsResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState<'text' | 'voice' | 'benchmark'>('text');
  const [error, setError] = useState<ApiError | null>(null);
  const [nlpEngines, setNlpEngines] = useState<EngineInfo[]>([]);
  const [graphEngines, setGraphEngines] = useState<EngineInfo[]>([]);

  useEffect(() => {
    loadHealthAndStats();
    loadEngines();
  }, []);

  const loadHealthAndStats = async () => {
    try {
      const [healthData, statsData] = await Promise.all([
        travelOrderAPI.getHealth(),
        travelOrderAPI.getStats(),
      ]);
      setHealth(healthData);
      setStats(statsData);
    } catch (error) {
      console.error('Error loading status:', error);
    }
  };

  const loadEngines = async () => {
    try {
      const engines = await travelOrderAPI.getEngines();
      setNlpEngines(engines.filter(e => e.type === 'nlp'));
      setGraphEngines(engines.filter(e => e.type === 'graph'));
    } catch (error) {
      console.error('Error loading engines:', error);
    }
  };

  const handleResolve = useCallback(async (
    text: string,
    nlpEngine: string,
    graphEngine: string,
    optimization: 'time' | 'pollution',
    trainTypes?: string[]
  ) => {
    setLoading(true);
    setRouteResult(null);
    setResolveResult(null);
    setError(null);

    try {
      const result = await travelOrderAPI.resolve({
        text,
        nlp_engine: nlpEngine,
        graph_engine: graphEngine,
        optimization,
        train_types: trainTypes || null,
      });
      setResolveResult(result);
    } catch (err: any) {
      console.error('Search error:', err);
      if (err.response?.data) {
        const apiError = err.response.data;
        if (apiError.error) {
          setError(apiError);
        } else {
          setError({
            error: true,
            error_type: 'api_error',
            error_message: apiError.detail || 'Error during search',
            suggestion: 'Check that the API is running on http://localhost:8000'
          });
        }
      } else if (err.code === 'ERR_NETWORK' || err.message === 'Network Error') {
        setError({
          error: true,
          error_type: 'connection_error',
          error_message: 'Cannot connect to server',
          error_details: 'The API server is not responding',
          suggestion: 'Start the API with: uv run uvicorn src.server.app:app --reload --port 8000'
        });
      } else {
        setError({
          error: true,
          error_type: 'unknown_error',
          error_message: err.message || 'Unexpected error',
          suggestion: 'Try again or check the server logs'
        });
      }
    } finally {
      setLoading(false);
    }
  }, []);

  const handleAudioSearch = useCallback(async (audioFile: File, metric: 'time' | 'distance') => {
    setLoading(true);
    setRouteResult(null);
    setResolveResult(null);
    setError(null);

    try {
      const result = await travelOrderAPI.speechToRoute(audioFile, metric);
      if (result.route_response?.error) {
        setError(result.route_response as unknown as ApiError);
        return;
      }
      setRouteResult(result.route_response);
    } catch (err: any) {
      console.error('Transcription error:', err);
      if (err.code === 'ERR_NETWORK') {
        setError({
          error: true,
          error_type: 'connection_error',
          error_message: 'Cannot connect to server',
          suggestion: 'Check that the API is running'
        });
      } else {
        setError({
          error: true,
          error_type: 'unknown_error',
          error_message: err.message || 'Transcription error',
          suggestion: 'Check audio format and try again'
        });
      }
    } finally {
      setLoading(false);
    }
  }, []);

  const handleDismissError = useCallback(() => {
    setError(null);
  }, []);

  return (
    <div className="app">
      <header className="header">
        <div className="header-content">
          <div className="logo">
            <Train size={32} />
            <h1>Travel Order Resolver</h1>
          </div>
          <p className="subtitle">
            Strategy Pattern: {nlpEngines.length} NLP engines + {graphEngines.length} Graph engines
          </p>
        </div>
      </header>

      <div className="container">
        <HealthStatus health={health} stats={stats} />

        <div className="main-card">
          <div className="tabs">
            <button
              className={`tab ${activeTab === 'text' ? 'active' : ''}`}
              onClick={() => setActiveTab('text')}
            >
              Recherche par texte
            </button>
            <button
              className={`tab ${activeTab === 'voice' ? 'active' : ''}`}
              onClick={() => setActiveTab('voice')}
            >
              Recherche vocale
            </button>
            <button
              className={`tab ${activeTab === 'benchmark' ? 'active' : ''}`}
              onClick={() => setActiveTab('benchmark')}
            >
              <Cpu size={16} style={{ marginRight: '6px', verticalAlign: 'middle' }} />
              Benchmark
            </button>
          </div>

          <div className="tab-content">
            {activeTab === 'text' ? (
              <RouteSearch
                onSearch={handleResolve}
                loading={loading}
                nlpEngines={nlpEngines}
                graphEngines={graphEngines}
              />
            ) : activeTab === 'voice' ? (
              <AudioRecorder onAudioUpload={handleAudioSearch} loading={loading} />
            ) : (
              <BenchmarkPanel />
            )}
          </div>

          {error && (
            <ErrorDisplay
              error={error}
              onDismiss={handleDismissError}
            />
          )}

          {(routeResult || resolveResult) && !error && (
            <div className="results-section">
              <RouteDisplay result={routeResult} resolveResult={resolveResult} />
            </div>
          )}

          {loading && (
            <div className="loading-overlay">
              <div className="spinner"></div>
              <p>Traitement en cours...</p>
            </div>
          )}
        </div>

        <footer className="footer">
          <div className="footer-content">
            <div className="footer-section">
              <Activity size={16} />
              <span>
                NLP: <strong>{nlpEngines.map(e => e.name).join(', ') || 'loading...'}</strong>
              </span>
            </div>
            <div className="footer-section">
              <Database size={16} />
              <span>
                SNCF Data: <strong>{(stats as any)?.total_stations || '...'}</strong> gares
              </span>
            </div>
            <div className="footer-section">
              <Train size={16} />
              <span>
                Graph: <strong>{graphEngines.map(e => e.name).join(', ') || 'loading...'}</strong>
              </span>
            </div>
          </div>
          <p className="copyright">
            T-AIA-911-PAR_14 - Epitech MSc Pro 2026 - Strategy Pattern (4 NLP + 3 Graph)
          </p>
        </footer>
      </div>
    </div>
  );
}

export default App;
