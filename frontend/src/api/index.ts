import axios from "axios";

import type {
  ChatResponse,
  TokenResponse,
  Trip,
  TripCreate,
  TripPlanResponse,
  TripUpdate,
  User,
} from "@/types";

const client = axios.create({
  baseURL: import.meta.env.VITE_API_URL || "/api/v1",
  timeout: 60_000,
});

client.interceptors.request.use((config) => {
  const token = localStorage.getItem("access_token");
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

client.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem("access_token");
      window.dispatchEvent(new Event("auth-expired"));
    }
    return Promise.reject(error);
  },
);

export function getApiErrorMessage(error: unknown): string {
  if (axios.isAxiosError(error)) {
    const detail = error.response?.data?.detail;
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail)) {
      return detail.map((item) => item.msg).filter(Boolean).join("；");
    }
    if (error.code === "ECONNABORTED") return "请求超时，请稍后重试";
    return error.message;
  }
  return error instanceof Error ? error.message : "请求失败";
}

export const authApi = {
  async login(email: string, password: string): Promise<TokenResponse> {
    return (await client.post<TokenResponse>("/auth/login", { email, password })).data;
  },

  async register(email: string, password: string, displayName?: string): Promise<User> {
    return (
      await client.post<User>("/auth/register", {
        email,
        password,
        display_name: displayName || null,
      })
    ).data;
  },

  async me(): Promise<User> {
    return (await client.get<User>("/auth/me")).data;
  },
};

export const tripApi = {
  async list(): Promise<Trip[]> {
    return (await client.get<Trip[]>("/trips")).data;
  },

  async create(data: TripCreate): Promise<Trip> {
    return (await client.post<Trip>("/trips", data)).data;
  },

  async getDetail(id: string): Promise<Trip> {
    return (await client.get<Trip>(`/trips/${id}`)).data;
  },

  async update(id: string, data: TripUpdate): Promise<Trip> {
    return (await client.patch<Trip>(`/trips/${id}`, data)).data;
  },

  async remove(id: string): Promise<void> {
    await client.delete(`/trips/${id}`);
  },

  async generatePlan(
    id: string,
    additionalRequirements?: string,
  ): Promise<TripPlanResponse> {
    return (
      await client.post<TripPlanResponse>(`/trips/${id}/generate-plan`, {
        additional_requirements: additionalRequirements?.trim() || null,
      }, {
        timeout: 300_000,
      })
    ).data;
  },
};

export const chatApi = {
  async sendMessage(message: string, conversationId?: string | null): Promise<ChatResponse> {
    return (
      await client.post<ChatResponse>("/chat", {
        message,
        conversation_id: conversationId || null,
      })
    ).data;
  },
};

export { client };
