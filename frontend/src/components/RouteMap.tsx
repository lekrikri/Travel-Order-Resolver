import { useEffect } from 'react';
import { MapContainer, TileLayer, Marker, Polyline, Popup, useMap } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';

// Fix pour les icônes Leaflet avec Vite
import icon from 'leaflet/dist/images/marker-icon.png';
import iconShadow from 'leaflet/dist/images/marker-shadow.png';
import iconRetina from 'leaflet/dist/images/marker-icon-2x.png';

// Configuration des icônes par défaut
const DefaultIcon = L.icon({
  iconUrl: icon,
  iconRetinaUrl: iconRetina,
  shadowUrl: iconShadow,
  iconSize: [25, 41],
  iconAnchor: [12, 41],
  popupAnchor: [1, -34],
  shadowSize: [41, 41],
});

L.Marker.prototype.options.icon = DefaultIcon;

// Icônes personnalisées pour départ et arrivée
const startIcon = new L.Icon({
  iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-green.png',
  shadowUrl: iconShadow,
  iconSize: [25, 41],
  iconAnchor: [12, 41],
  popupAnchor: [1, -34],
  shadowSize: [41, 41],
});

const endIcon = new L.Icon({
  iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-red.png',
  shadowUrl: iconShadow,
  iconSize: [25, 41],
  iconAnchor: [12, 41],
  popupAnchor: [1, -34],
  shadowSize: [41, 41],
});

interface StationDetail {
  step_number: number;
  code_uic?: string;
  nom_gare: string;
  commune?: string | null;
  latitude: number | null;
  longitude: number | null;
  next_time_minutes?: number;
  next_distance_km?: number;
}

interface RouteMapProps {
  stations: StationDetail[];
}

// Composant pour ajuster automatiquement la vue de la carte
function MapBounds({ positions }: { positions: [number, number][] }) {
  const map = useMap();

  useEffect(() => {
    if (positions.length > 0) {
      const bounds = L.latLngBounds(positions);
      map.fitBounds(bounds, { padding: [50, 50] });
    }
  }, [positions, map]);

  return null;
}

export default function RouteMap({ stations }: RouteMapProps) {
  // Vérifier que stations existe et n'est pas vide
  if (!stations || stations.length === 0) {
    return (
      <div style={{ padding: '20px', textAlign: 'center', color: '#666' }}>
        <p>Aucune donnee d'itineraire disponible.</p>
      </div>
    );
  }

  // Filtrer les stations avec coordonnées valides
  const validStations = stations.filter(
    (s) => s.latitude !== null && s.latitude !== undefined &&
           s.longitude !== null && s.longitude !== undefined
  );

  if (validStations.length === 0) {
    return (
      <div style={{ padding: '20px', textAlign: 'center', color: '#666' }}>
        <p>Aucune coordonnee GPS disponible pour cet itineraire.</p>
      </div>
    );
  }

  // Créer les positions pour la polyline
  const positions: [number, number][] = validStations.map((s) => [
    s.latitude!,
    s.longitude!,
  ]);

  // Centre initial (première gare) - avec fallback sur Paris
  const center: [number, number] = validStations[0]?.latitude && validStations[0]?.longitude
    ? [validStations[0].latitude, validStations[0].longitude]
    : [48.8566, 2.3522]; // Paris par défaut

  return (
    <div style={{ height: '500px', width: '100%', borderRadius: '8px', overflow: 'hidden' }}>
      <MapContainer
        center={center}
        zoom={7}
        style={{ height: '100%', width: '100%' }}
        scrollWheelZoom={true}
      >
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />

        {/* Ajuster automatiquement la vue */}
        <MapBounds positions={positions} />

        {/* Tracer la ligne de l'itinéraire */}
        <Polyline
          positions={positions}
          color="#3498db"
          weight={4}
          opacity={0.7}
        />

        {/* Marqueurs pour chaque gare */}
        {validStations.map((station, index) => {
          const isStart = index === 0;
          const isEnd = index === validStations.length - 1;
          const icon = isStart ? startIcon : isEnd ? endIcon : DefaultIcon;

          return (
            <Marker
              key={station.code_uic}
              position={[station.latitude!, station.longitude!]}
              icon={icon}
            >
              <Popup>
                <div style={{ minWidth: '200px' }}>
                  <h3 style={{ margin: '0 0 10px 0', color: '#2c3e50' }}>
                    {isStart ? '🚩 ' : isEnd ? '🏁 ' : '🔸 '}
                    {station.nom_gare}
                  </h3>
                  <p style={{ margin: '5px 0', fontSize: '13px', color: '#666' }}>
                    <strong>Étape {station.step_number}</strong>
                    {station.commune && station.commune !== station.nom_gare && (
                      <span> • {station.commune}</span>
                    )}
                  </p>
                  {station.next_time_minutes && (
                    <p style={{ margin: '5px 0', fontSize: '13px', color: '#3498db' }}>
                      ⏱️ Prochain trajet : {Math.round(station.next_time_minutes)} min
                      {station.next_distance_km && (
                        <span> ({Math.round(station.next_distance_km)} km)</span>
                      )}
                    </p>
                  )}
                  {isStart && (
                    <p style={{ margin: '5px 0', fontSize: '12px', color: '#27ae60' }}>
                      <strong>Départ</strong>
                    </p>
                  )}
                  {isEnd && (
                    <p style={{ margin: '5px 0', fontSize: '12px', color: '#e74c3c' }}>
                      <strong>Arrivée</strong>
                    </p>
                  )}
                </div>
              </Popup>
            </Marker>
          );
        })}
      </MapContainer>
    </div>
  );
}
