import { useState, useEffect, useCallback } from 'react';
import { Train, Activity, Database } from 'lucide-react';
import RouteSearch from './components/RouteSearch';
import AudioRecorder from './components/AudioRecorder';
import RouteDisplay from './components/RouteDisplay';
import HealthStatus from './components/HealthStatus';
import ErrorDisplay from './components/ErrorDisplay';
import { travelOrderAPI } from './services/api';
import type { RouteResponse, HealthResponse, StatsResponse, ApiError } from './types';
import './styles/App.css';

function App() {
  const [routeResult, setRouteResult] = useState<RouteResponse | null>(null);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [stats, setStats] = useState<StatsResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState<'text' | 'voice'>('text');
  const [error, setError] = useState<ApiError | null>(null);
  const [lastSearch, setLastSearch] = useState<{
    sentence: string;
    useRag: boolean;
    metric: 'time' | 'distance';
  } | null>(null);

  useEffect(() => {
    // Charger le statut au démarrage
    loadHealthAndStats();
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
      console.error('Erreur lors du chargement du statut:', error);
    }
  };

  const handleRouteSearch = useCallback(async (
    sentence: string,
    useRag: boolean,
    metric: 'time' | 'distance'
  ) => {
    setLoading(true);
    setRouteResult(null);
    setError(null);
    setLastSearch({ sentence, useRag, metric });

    try {
      const result = await travelOrderAPI.getRoute({
        sentence,
        use_rag: useRag,
        metric,
      });

      // Verifier si la reponse contient une erreur
      if (result.error) {
        setError(result as unknown as ApiError);
        return;
      }

      setRouteResult(result);
    } catch (err: any) {
      console.error('Erreur lors de la recherche:', err);

      // Gerer les erreurs de l'API
      if (err.response?.data) {
        const apiError = err.response.data;
        if (apiError.error) {
          setError(apiError);
        } else {
          setError({
            error: true,
            error_type: 'network_error',
            error_message: apiError.detail || 'Erreur lors de la recherche',
            suggestion: 'Verifiez que l\'API est lancee sur http://localhost:8000'
          });
        }
      } else if (err.code === 'ERR_NETWORK' || err.message === 'Network Error') {
        setError({
          error: true,
          error_type: 'connection_error',
          error_message: 'Impossible de se connecter au serveur',
          error_details: 'Le serveur API ne repond pas',
          suggestion: 'Verifiez que l\'API est lancee avec: uvicorn api_spacy:app --reload'
        });
      } else {
        setError({
          error: true,
          error_type: 'unknown_error',
          error_message: err.message || 'Une erreur inattendue s\'est produite',
          suggestion: 'Reessayez ou contactez l\'administrateur'
        });
      }
    } finally {
      setLoading(false);
    }
  }, []);

  const handleAudioSearch = useCallback(async (audioFile: File, metric: 'time' | 'distance') => {
    setLoading(true);
    setRouteResult(null);
    setError(null);

    try {
      const result = await travelOrderAPI.speechToRoute(audioFile, metric);

      if (result.route_response?.error) {
        setError(result.route_response as unknown as ApiError);
        return;
      }

      setRouteResult(result.route_response);
    } catch (err: any) {
      console.error('Erreur lors de la transcription:', err);

      if (err.response?.data) {
        const apiError = err.response.data;
        if (apiError.error) {
          setError(apiError);
        } else {
          setError({
            error: true,
            error_type: 'transcription_error',
            error_message: apiError.detail || 'Erreur lors de la transcription',
            suggestion: 'Assurez-vous que Whisper est installe sur le serveur'
          });
        }
      } else if (err.code === 'ERR_NETWORK') {
        setError({
          error: true,
          error_type: 'connection_error',
          error_message: 'Impossible de se connecter au serveur',
          suggestion: 'Verifiez que l\'API est lancee'
        });
      } else {
        setError({
          error: true,
          error_type: 'unknown_error',
          error_message: err.message || 'Erreur de transcription',
          suggestion: 'Verifiez le format audio et reessayez'
        });
      }
    } finally {
      setLoading(false);
    }
  }, []);

  const handleDismissError = useCallback(() => {
    setError(null);
  }, []);

  const handleRetry = useCallback(() => {
    if (lastSearch) {
      handleRouteSearch(lastSearch.sentence, lastSearch.useRag, lastSearch.metric);
    }
  }, [lastSearch, handleRouteSearch]);

  return (
    <div className="app">
      <header className="header">
        <div className="header-content">
          <div className="logo">
            <Train size={32} />
            <h1>Travel Order Resolver</h1>
          </div>
          <p className="subtitle">
            Extraction NER (spaCy) + Fuzzy Matching + Pathfinder A*
          </p>
        </div>
      </header>

      <div className="container">
        {/* Status bar */}
        <HealthStatus health={health} stats={stats} />

        {/* Main content */}
        <div className="main-card">
          {/* Tabs */}
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
          </div>

          {/* Content */}
          <div className="tab-content">
            {activeTab === 'text' ? (
              <RouteSearch onSearch={handleRouteSearch} loading={loading} />
            ) : (
              <AudioRecorder onAudioUpload={handleAudioSearch} loading={loading} />
            )}
          </div>

          {/* Error Display */}
          {error && (
            <ErrorDisplay
              error={error}
              onDismiss={handleDismissError}
              onRetry={lastSearch ? handleRetry : undefined}
            />
          )}

          {/* Results */}
          {routeResult && !error && (
            <div className="results-section">
              <RouteDisplay result={routeResult} />
            </div>
          )}

          {loading && (
            <div className="loading-overlay">
              <div className="spinner"></div>
              <p>Traitement en cours...</p>
            </div>
          )}
        </div>

        {/* Footer */}
        <footer className="footer">
          <div className="footer-content">
            <div className="footer-section">
              <Activity size={16} />
              <span>
                Precision NER: <strong>99.88%</strong> (F1-Score)
              </span>
            </div>
            <div className="footer-section">
              <Database size={16} />
              <span>
                SNCF Data: <strong>{stats?.total_stations || '2,785'}</strong> gares
              </span>
            </div>
            <div className="footer-section">
              <Train size={16} />
              <span>
                spaCy NER + Fuzzy Matching + Pathfinder A*
              </span>
            </div>
          </div>
          <p className="copyright">
            T-AIA-911-PAR_14 - Epitech MSc Pro 2026 - spaCy NER (F1: 99.88%)
          </p>
        </footer>
      </div>
    </div>
  );
}

export default App;
