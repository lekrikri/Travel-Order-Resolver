import { Activity, Cpu, Database, CheckCircle, XCircle } from 'lucide-react';
import type { HealthResponse, StatsResponse } from '../types';

interface HealthStatusProps {
  health: HealthResponse | null;
  stats: StatsResponse | null;
}

function HealthStatus({ health }: HealthStatusProps) {
  if (!health) {
    return (
      <div className="health-status loading">
        <Activity size={16} className="spin" />
        <span>Connexion à l'API...</span>
      </div>
    );
  }

  const isHealthy = health.status === 'ok' && health.nlp_engines > 0;

  return (
    <div className={`health-status ${isHealthy ? 'healthy' : 'unhealthy'}`}>
      <div className="health-item">
        {isHealthy ? (
          <CheckCircle size={16} className="status-icon success" />
        ) : (
          <XCircle size={16} className="status-icon error" />
        )}
        <span>
          API : <strong>{health.status}</strong>
        </span>
      </div>

      <div className="health-item">
        <Cpu size={16} />
        <span>
          {health.gpu_available ? (
            <>
              GPU : <strong>{health.gpu_name}</strong>
            </>
          ) : (
            <>CPU : <strong>Actif</strong></>
          )}
        </span>
      </div>

      <div className="health-item">
        <Database size={16} />
        <span>
          SNCF : <strong>{health.stations}</strong> gares,{' '}
          <strong>{health.connections}</strong> connexions
        </span>
      </div>

      <div className="health-item">
        <Activity size={16} />
        <span>
          Engines : <strong>{health.nlp_engines} NLP</strong> + <strong>{health.graph_engines} Graph</strong>
        </span>
      </div>
    </div>
  );
}

export default HealthStatus;
