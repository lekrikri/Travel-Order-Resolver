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

// --- Strategy pattern backend types ---

export interface EngineInfo {
  name: string;
  description: string;
  type: 'nlp' | 'graph';
}

export interface ResolveRequest {
  text: string;
  nlp_engine: string;
  graph_engine: string;
  train_types?: string[] | null;
  optimization: 'time' | 'pollution';
}

export interface SegmentInfo {
  from_station: string;
  to_station: string;
  train_type: string;
  weight_minutes: number;
}

export interface BilletInfo {
  from_station: string;
  to_station: string;
  train_type: string;
  duration_minutes: number;
  co2_grams: number;
  price_euros: number;
  stops: string[];
}

export interface ResolveResponse {
  text: string;
  departure: string | null;
  destination: string | null;
  route: string[];
  segments: SegmentInfo[];
  billets: BilletInfo[];
  travel_time_min: number;
  total_co2_grams: number;
  total_price_euros: number;
  is_valid: boolean;
  nlp_time_ms: number;
  graph_time_ms: number;
  total_time_ms: number;
}

export interface NLPBenchmarkResult {
  departure: string | null;
  destination: string | null;
  confidence: number;
  is_valid: boolean;
  time_ms: number;
  memory_kb: number;
  engine_name: string;
}

export interface GraphBenchmarkResult {
  steps: string[];
  segments: SegmentInfo[];
  found: boolean;
  total_weight: number;
  nodes_explored: number;
  time_ms: number;
  memory_kb: number;
  engine_name: string;
}
