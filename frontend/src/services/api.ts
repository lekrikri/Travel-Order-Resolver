import axios from 'axios';
import type {
  RouteRequest,
  RouteResponse,
  HealthResponse,
  StatsResponse,
  SpeechToRouteResponse
} from '../types';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const travelOrderAPI = {
  // Health check
  async getHealth(): Promise<HealthResponse> {
    const response = await apiClient.get<HealthResponse>('/health');
    return response.data;
  },

  // Stats
  async getStats(): Promise<StatsResponse> {
    const response = await apiClient.get<StatsResponse>('/stats');
    return response.data;
  },

  // Get route (NLP + Pathfinding)
  async getRoute(request: RouteRequest): Promise<RouteResponse> {
    const response = await apiClient.post<RouteResponse>('/route', request);
    return response.data;
  },

  // Extract cities only (NLP only)
  async extractCities(request: RouteRequest): Promise<RouteResponse> {
    const response = await apiClient.post<RouteResponse>('/extract', request);
    return response.data;
  },

  // Speech to route
  async speechToRoute(
    audioFile: File,
    metric: 'time' | 'distance' = 'time'
  ): Promise<SpeechToRouteResponse> {
    const formData = new FormData();
    formData.append('audio_file', audioFile);
    formData.append('metric', metric);

    const response = await apiClient.post<SpeechToRouteResponse>(
      '/speech-to-route',
      formData,
      {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      }
    );
    return response.data;
  },
};
