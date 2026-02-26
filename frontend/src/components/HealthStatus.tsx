import { Activity, Cpu, Database, CheckCircle, XCircle } from 'lucide-react';
import type { HealthResponse, StatsResponse } from '../types';

interface HealthStatusProps {
  health: HealthResponse | null;
  stats: StatsResponse | null;
}

function HealthStatus({ health, stats }: HealthStatusProps) {
  if (!health) {
    return (
      <div className="health-status loading">
        <Activity size={16} className="spin" />
        <span>Connexion à l'API...</span>
      </div>
    );
  }

  const isHealthy = health.status === 'healthy' && health.model_loaded;

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

      {stats && (
        <div className="health-item">
          <Database size={16} />
          <span>
            SNCF : <strong>{stats.total_stations}</strong> gares,{' '}
            <strong>{stats.total_connections}</strong> connexions
          </span>
        </div>
      )}

      <div className="health-item">
        <Activity size={16} />
        <span>
          Modèle : <strong>{health.model_loaded ? 'Chargé' : 'Non chargé'}</strong>
        </span>
      </div>
    </div>
  );
}

export default HealthStatus;
