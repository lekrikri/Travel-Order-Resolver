// Types pour l'API Travel Order Resolver

export interface RouteRequest {
  sentence: string;
  use_rag: boolean;
  metric: 'time' | 'distance';
}

export interface StationDetail {
  step_number: number;
  code_uic: string;
  nom_gare: string;
  commune: string | null;
  latitude: number | null;
  longitude: number | null;
  next_time_minutes?: number;
  next_distance_km?: number;
}

export interface RouteResponse {
  error?: boolean;
  sentence_id: number;
  sentence: string;
  departure: string | null;
  arrival: string | null;
  departure_matched?: string | null;
  arrival_matched?: string | null;
  confidence: number;
  is_valid: boolean;
  route: string[] | null;
  route_count: number | null;
  total_distance: number | null;
  total_time: number | null;
  formatted_output: string;
  method: string;
  details?: StationDetail[];
  warning?: string;
}

export interface ApiError {
  error: boolean;
  error_type: string;
  error_message: string;
  error_details?: string | null;
  suggestion?: string | null;
}

export interface HealthResponse {
  status: string;
  gpu_available: boolean;
  gpu_name: string | null;
  model_loaded: boolean;
  sncf_stations: number | null;
  sncf_connections: number | null;
}

export interface StatsResponse {
  total_stations: number;
  total_connections: number;
  gpu_available: boolean;
  gpu_name: string | null;
  chromadb_examples: number;
}

export interface SpeechToRouteResponse {
  transcribed_text: string;
  route_response: RouteResponse;
}
