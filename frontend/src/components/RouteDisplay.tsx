import { MapPin, Navigation, Clock, Gauge, CheckCircle, XCircle, Map, AlertTriangle } from 'lucide-react';
import type { RouteResponse } from '../types';
import RouteMap from './RouteMap';

interface RouteDisplayProps {
  result: RouteResponse;
}

function RouteDisplay({ result }: RouteDisplayProps) {
  const formatTime = (minutes: number | null) => {
    if (!minutes) return 'N/A';
    const hours = Math.floor(minutes / 60);
    const mins = Math.round(minutes % 60);
    return hours > 0 ? `${hours}h ${mins}min` : `${mins}min`;
  };

  const formatDistance = (km: number | null) => {
    if (!km) return 'N/A';
    return `${km.toFixed(1)} km`;
  };

  return (
    <div className="route-display">
      <div className="result-header">
        {result.is_valid ? (
          <div className="status-badge success">
            <CheckCircle size={20} />
            <span>Itinéraire trouvé</span>
          </div>
        ) : (
          <div className="status-badge error">
            <XCircle size={20} />
            <span>Commande invalide</span>
          </div>
        )}

        <div className="confidence-badge">
          Confiance : <strong>{(result.confidence * 100).toFixed(1)}%</strong>
        </div>
      </div>

      <div className="sentence-display">
        <p className="sentence-label">Phrase analysee :</p>
        <p className="sentence-text">{result.sentence}</p>
      </div>

      {/* Affichage du warning si present */}
      {result.warning && (
        <div className="warning-display">
          <AlertTriangle size={20} className="warning-icon" />
          <p className="warning-text">{result.warning}</p>
        </div>
      )}

      {result.is_valid && result.departure && result.arrival ? (
        <>
          <div className="cities-summary">
            <div className="city-card departure">
              <MapPin size={24} className="city-icon" />
              <div className="city-info">
                <p className="city-label">Départ</p>
                <p className="city-name">{result.departure}</p>
              </div>
            </div>

            <Navigation size={32} className="arrow-icon" />

            <div className="city-card arrival">
              <MapPin size={24} className="city-icon" />
              <div className="city-info">
                <p className="city-label">Arrivée</p>
                <p className="city-name">{result.arrival}</p>
              </div>
            </div>
          </div>

          {result.route && result.route.length > 0 && (
            <>
              <div className="route-stats">
                <div className="stat-card">
                  <div className="stat-icon">
                    <MapPin size={20} />
                  </div>
                  <div className="stat-info">
                    <p className="stat-label">Gares</p>
                    <p className="stat-value">{result.route_count}</p>
                  </div>
                </div>

                <div className="stat-card">
                  <div className="stat-icon">
                    <Clock size={20} />
                  </div>
                  <div className="stat-info">
                    <p className="stat-label">Durée estimée</p>
                    <p className="stat-value">{formatTime(result.total_time)}</p>
                  </div>
                </div>

                <div className="stat-card">
                  <div className="stat-icon">
                    <Gauge size={20} />
                  </div>
                  <div className="stat-info">
                    <p className="stat-label">Distance</p>
                    <p className="stat-value">{formatDistance(result.total_distance)}</p>
                  </div>
                </div>
              </div>

              {/* Carte interactive si les coordonnées GPS sont disponibles */}
              {result.details && result.details.length > 0 &&
               result.details.some(d => d.latitude && d.longitude) && (
                <div className="route-map-section">
                  <h3 className="route-title">
                    <Map size={20} style={{ marginRight: '8px', verticalAlign: 'middle' }} />
                    Carte interactive
                  </h3>
                  <RouteMap stations={result.details} />
                </div>
              )}

              <div className="route-details">
                <h3 className="route-title">Itinéraire détaillé</h3>
                <div className="route-stations">
                  {result.route.map((station, index) => (
                    <div key={index} className="station-item">
                      <div className="station-marker">
                        {index === 0 ? (
                          <div className="marker departure-marker">D</div>
                        ) : index === result.route!.length - 1 ? (
                          <div className="marker arrival-marker">A</div>
                        ) : (
                          <div className="marker step-marker">{index}</div>
                        )}
                        {index < result.route!.length - 1 && (
                          <div className="connector"></div>
                        )}
                      </div>
                      <div className="station-name">
                        {station}
                        {index === 0 && <span className="station-badge">Départ</span>}
                        {index === result.route!.length - 1 && (
                          <span className="station-badge arrival-badge">Arrivée</span>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </>
          )}

          <div className="output-format">
            <p className="output-label">Format de sortie :</p>
            <code className="output-code">{result.formatted_output}</code>
          </div>
        </>
      ) : (
        <div className="invalid-message">
          <XCircle size={48} />
          <p>
            {result.warning
              ? result.warning
              : "Cette commande ne contient pas d'informations valides sur un trajet en train."}
          </p>
          {result.departure && !result.arrival && (
            <p className="invalid-help">
              Depart detecte : <strong>{result.departure}</strong> - Arrivee manquante
            </p>
          )}
          {!result.departure && result.arrival && (
            <p className="invalid-help">
              Arrivee detectee : <strong>{result.arrival}</strong> - Depart manquant
            </p>
          )}
          {!result.departure && !result.arrival && (
            <p className="invalid-help">
              Essayez une phrase comme : "Je veux aller de Paris a Lyon"
            </p>
          )}
        </div>
      )}

      <div className="method-badge">
        Méthode : <strong>{result.method}</strong>
      </div>
    </div>
  );
}

export default RouteDisplay;
