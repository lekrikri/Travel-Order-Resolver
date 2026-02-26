import { AlertCircle, XCircle, AlertTriangle, RefreshCw } from 'lucide-react';
import type { ApiError } from '../types';

interface ErrorDisplayProps {
  error: ApiError | null;
  onDismiss?: () => void;
  onRetry?: () => void;
}

function ErrorDisplay({ error, onDismiss, onRetry }: ErrorDisplayProps) {
  if (!error) return null;

  const getErrorIcon = () => {
    switch (error.error_type) {
      case 'service_unavailable':
        return <AlertTriangle size={24} />;
      case 'invalid_input':
        return <AlertCircle size={24} />;
      default:
        return <XCircle size={24} />;
    }
  };

  const getErrorTitle = () => {
    switch (error.error_type) {
      case 'service_unavailable':
        return 'Service non disponible';
      case 'invalid_input':
        return 'Saisie invalide';
      case 'ner_error':
        return 'Erreur d\'extraction';
      case 'fuzzy_error':
        return 'Erreur de correspondance';
      case 'pipeline_error':
        return 'Erreur de traitement';
      case 'http_error':
        return 'Erreur de communication';
      default:
        return 'Erreur';
    }
  };

  return (
    <div className="error-display">
      <div className="error-header">
        <div className="error-icon">
          {getErrorIcon()}
        </div>
        <div className="error-title-section">
          <h3 className="error-title">{getErrorTitle()}</h3>
          <p className="error-message">{error.error_message}</p>
        </div>
        {onDismiss && (
          <button className="error-dismiss" onClick={onDismiss} title="Fermer">
            <XCircle size={20} />
          </button>
        )}
      </div>

      {error.error_details && (
        <div className="error-details">
          <strong>Details :</strong> {error.error_details}
        </div>
      )}

      {error.suggestion && (
        <div className="error-suggestion">
          <strong>Suggestion :</strong> {error.suggestion}
        </div>
      )}

      {onRetry && (
        <button className="error-retry-btn" onClick={onRetry}>
          <RefreshCw size={16} />
          Reessayer
        </button>
      )}
    </div>
  );
}

export default ErrorDisplay;
