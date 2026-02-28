import { useState } from 'react';
import {
  MapPin, Navigation, Clock, CheckCircle, XCircle, Map,
  AlertTriangle, Leaf, Euro, Timer, ChevronDown, ChevronUp, Train,
} from 'lucide-react';
import type { RouteResponse, ResolveResponse, BilletInfo } from '../types';
import RouteMap from './RouteMap';

interface RouteDisplayProps {
  result: RouteResponse | null;
  resolveResult: ResolveResponse | null;
}

function BilletCard({ billet, index }: { billet: BilletInfo; index: number }) {
  const [expanded, setExpanded] = useState(false);

  const formatDuration = (min: number) => {
    const h = Math.floor(min / 60);
    const m = Math.round(min % 60);
    return h > 0 ? `${h}h${m.toString().padStart(2, '0')}` : `${m}min`;
  };

  return (
    <div className="billet-card">
      <div className="billet-header">
        <span className={`train-badge train-${billet.train_type.toLowerCase().replace(/\s+/g, '-')}`}>
          <Train size={14} />
          {billet.train_type}
        </span>
        <span className="billet-number">Billet {index + 1}</span>
      </div>
      <div className="billet-route">
        <span className="billet-station">{billet.from_station}</span>
        <Navigation size={16} className="billet-arrow" />
        <span className="billet-station">{billet.to_station}</span>
      </div>
      <div className="billet-stats">
        <span className="billet-stat"><Clock size={14} /> {formatDuration(billet.duration_minutes)}</span>
        <span className="billet-stat"><Leaf size={14} /> {billet.co2_grams.toFixed(0)}g CO2</span>
        <span className="billet-stat"><Euro size={14} /> {billet.price_euros.toFixed(2)}</span>
      </div>
      {billet.stops.length > 0 && (
        <button className="billet-stops-toggle" onClick={() => setExpanded(!expanded)}>
          {expanded ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
          {billet.stops.length} arret{billet.stops.length > 1 ? 's' : ''} intermediaire{billet.stops.length > 1 ? 's' : ''}
        </button>
      )}
      {expanded && billet.stops.length > 0 && (
        <div className="billet-stops-list">
          {billet.stops.map((stop, i) => (
            <span key={i} className="billet-stop">{stop}</span>
          ))}
        </div>
      )}
    </div>
  );
}

function RouteDisplay({ result, resolveResult }: RouteDisplayProps) {
  const formatTime = (minutes: number | null) => {
    if (!minutes) return 'N/A';
    const hours = Math.floor(minutes / 60);
    const mins = Math.round(minutes % 60);
    return hours > 0 ? `${hours}h ${mins}min` : `${mins}min`;
  };

  // New resolve response display
  if (resolveResult) {
    return (
      <div className="route-display">
        <div className="result-header">
          {resolveResult.is_valid ? (
            <div className="status-badge success">
              <CheckCircle size={20} />
              <span>Itineraire trouve</span>
            </div>
          ) : (
            <div className="status-badge error">
              <XCircle size={20} />
              <span>Commande invalide</span>
            </div>
          )}

          <div className="perf-pills">
            <span className="perf-pill"><Timer size={14} /> NLP: {resolveResult.nlp_time_ms.toFixed(1)}ms</span>
            <span className="perf-pill"><Timer size={14} /> Graph: {resolveResult.graph_time_ms.toFixed(1)}ms</span>
            <span className="perf-pill total"><Timer size={14} /> Total: {resolveResult.total_time_ms.toFixed(1)}ms</span>
          </div>
        </div>

        <div className="sentence-display">
          <p className="sentence-label">Phrase analysee :</p>
          <p className="sentence-text">{resolveResult.text}</p>
        </div>

        {resolveResult.is_valid && resolveResult.departure && resolveResult.destination ? (
          <>
            <div className="cities-summary">
              <div className="city-card departure">
                <MapPin size={24} className="city-icon" />
                <div className="city-info">
                  <p className="city-label">Depart</p>
                  <p className="city-name">{resolveResult.departure}</p>
                </div>
              </div>

              <Navigation size={32} className="arrow-icon" />

              <div className="city-card arrival">
                <MapPin size={24} className="city-icon" />
                <div className="city-info">
                  <p className="city-label">Arrivee</p>
                  <p className="city-name">{resolveResult.destination}</p>
                </div>
              </div>
            </div>

            {/* Totals bar */}
            <div className="totals-bar">
              <div className="total-item">
                <Clock size={18} />
                <span>{formatTime(resolveResult.travel_time_min)}</span>
              </div>
              <div className="total-item">
                <Leaf size={18} />
                <span>{resolveResult.total_co2_grams.toFixed(0)}g CO2</span>
              </div>
              <div className="total-item">
                <Euro size={18} />
                <span>{resolveResult.total_price_euros.toFixed(2)} EUR</span>
              </div>
              <div className="total-item">
                <MapPin size={18} />
                <span>{resolveResult.route.length} gares</span>
              </div>
            </div>

            {/* Billets */}
            {resolveResult.billets.length > 0 && (
              <div className="billets-section">
                <h3 className="route-title">Billets</h3>
                <div className="billets-grid">
                  {resolveResult.billets.map((billet, i) => (
                    <BilletCard key={i} billet={billet} index={i} />
                  ))}
                </div>
              </div>
            )}

            {/* Route details */}
            {resolveResult.route.length > 0 && (
              <div className="route-details">
                <h3 className="route-title">Itineraire detaille</h3>
                <div className="route-stations">
                  {resolveResult.route.map((station, index) => (
                    <div key={index} className="station-item">
                      <div className="station-marker">
                        {index === 0 ? (
                          <div className="marker departure-marker">D</div>
                        ) : index === resolveResult.route.length - 1 ? (
                          <div className="marker arrival-marker">A</div>
                        ) : (
                          <div className="marker step-marker">{index}</div>
                        )}
                        {index < resolveResult.route.length - 1 && (
                          <div className="connector"></div>
                        )}
                      </div>
                      <div className="station-name">
                        {station}
                        {index === 0 && <span className="station-badge">Depart</span>}
                        {index === resolveResult.route.length - 1 && (
                          <span className="station-badge arrival-badge">Arrivee</span>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </>
        ) : (
          <div className="invalid-message">
            <XCircle size={48} />
            <p>Cette commande ne contient pas d'informations valides sur un trajet en train.</p>
            {resolveResult.departure && !resolveResult.destination && (
              <p className="invalid-help">
                Depart detecte : <strong>{resolveResult.departure}</strong> - Destination manquante
              </p>
            )}
            {!resolveResult.departure && resolveResult.destination && (
              <p className="invalid-help">
                Destination detectee : <strong>{resolveResult.destination}</strong> - Depart manquant
              </p>
            )}
            {!resolveResult.departure && !resolveResult.destination && (
              <p className="invalid-help">
                Essayez une phrase comme : "Je veux aller de Paris a Lyon"
              </p>
            )}
          </div>
        )}
      </div>
    );
  }

  // Legacy route response display
  if (result) {
    return (
      <div className="route-display">
        <div className="result-header">
          {result.is_valid ? (
            <div className="status-badge success">
              <CheckCircle size={20} />
              <span>Itineraire trouve</span>
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
                  <p className="city-label">Depart</p>
                  <p className="city-name">{result.departure}</p>
                </div>
              </div>
              <Navigation size={32} className="arrow-icon" />
              <div className="city-card arrival">
                <MapPin size={24} className="city-icon" />
                <div className="city-info">
                  <p className="city-label">Arrivee</p>
                  <p className="city-name">{result.arrival}</p>
                </div>
              </div>
            </div>

            {result.route && result.route.length > 0 && (
              <>
                <div className="route-stats">
                  <div className="stat-card">
                    <div className="stat-icon"><MapPin size={20} /></div>
                    <div className="stat-info">
                      <p className="stat-label">Gares</p>
                      <p className="stat-value">{result.route_count}</p>
                    </div>
                  </div>
                  <div className="stat-card">
                    <div className="stat-icon"><Clock size={20} /></div>
                    <div className="stat-info">
                      <p className="stat-label">Duree estimee</p>
                      <p className="stat-value">{formatTime(result.total_time)}</p>
                    </div>
                  </div>
                </div>

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
                  <h3 className="route-title">Itineraire detaille</h3>
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
                          {index === 0 && <span className="station-badge">Depart</span>}
                          {index === result.route!.length - 1 && (
                            <span className="station-badge arrival-badge">Arrivee</span>
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
          </div>
        )}

        <div className="method-badge">
          Methode : <strong>{result.method}</strong>
        </div>
      </div>
    );
  }

  return null;
}

export default RouteDisplay;
