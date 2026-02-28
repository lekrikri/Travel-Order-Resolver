import axios from 'axios';
import type {
  RouteRequest,
  RouteResponse,
  HealthResponse,
  StatsResponse,
  SpeechToRouteResponse,
  EngineInfo,
  ResolveRequest,
  ResolveResponse,
  NLPBenchmarkResult,
  GraphBenchmarkResult,
} from '../types';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const travelOrderAPI = {
  // Health check (new backend)
  async getHealth(): Promise<HealthResponse> {
    const response = await apiClient.get<HealthResponse>('/api/health');
    return response.data;
  },

  // Stats (new backend)
  async getStats(): Promise<StatsResponse> {
    const response = await apiClient.get<StatsResponse>('/api/stats');
    return response.data;
  },

  // Legacy: get route (NLP + Pathfinding) — old backend
  async getRoute(request: RouteRequest): Promise<RouteResponse> {
    const response = await apiClient.post<RouteResponse>('/api/route', request);
    return response.data;
  },

  // Legacy: extract cities only (NLP only)
  async extractCities(request: RouteRequest): Promise<RouteResponse> {
    const response = await apiClient.post<RouteResponse>('/api/extract', request);
    return response.data;
  },

  // Legacy: speech to route
  async speechToRoute(
    audioFile: File,
    metric: 'time' | 'distance' = 'time'
  ): Promise<SpeechToRouteResponse> {
    const formData = new FormData();
    formData.append('audio_file', audioFile);
    formData.append('metric', metric);

    const response = await apiClient.post<SpeechToRouteResponse>(
      '/api/speech-to-route',
      formData,
      {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      }
    );
    return response.data;
  },

  // --- Strategy pattern backend ---

  // List all engines
  async getEngines(): Promise<EngineInfo[]> {
    const response = await apiClient.get<EngineInfo[]>('/api/engines');
    return response.data;
  },

  // Resolve text to route (main endpoint)
  async resolve(request: ResolveRequest): Promise<ResolveResponse> {
    const response = await apiClient.post<ResolveResponse>('/api/resolve', request);
    return response.data;
  },

  // Benchmark all NLP engines on the same text
  async benchmarkNLP(text: string): Promise<NLPBenchmarkResult[]> {
    const response = await apiClient.post<NLPBenchmarkResult[]>('/api/benchmark/nlp', { text });
    return response.data;
  },

  // Benchmark all graph engines on the same departure/destination
  async benchmarkGraph(departure: string, destination: string): Promise<GraphBenchmarkResult[]> {
    const response = await apiClient.post<GraphBenchmarkResult[]>('/api/benchmark/graph', {
      departure,
      destination,
    });
    return response.data;
  },
};
