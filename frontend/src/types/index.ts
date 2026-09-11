export interface User {
  id: string;
  email: string;
  display_name: string | null;
  is_active: boolean;
  created_at: string;
  updated_at: string;
}

export interface TokenResponse {
  access_token: string;
  token_type: "bearer";
}

export type TripStatus = "draft" | "confirmed" | "completed";

export interface Trip {
  id: string;
  user_id: string;
  title: string;
  destination: string;
  country: string | null;
  start_date: string | null;
  end_date: string | null;
  budget: number | null;
  num_people: number;
  status: TripStatus;
  itinerary: ItineraryDay[] | null;
  preferences: Record<string, unknown> | null;
  created_at: string;
  updated_at: string;
}

export interface TripCreate {
  title: string;
  start_date?: string;
  end_date?: string;
  budget?: number;
  num_people: number;
  preferences: Record<string, unknown>;
}

export interface TripUpdate {
  title?: string;
  budget?: number;
  status?: TripStatus;
  itinerary?: Array<Record<string, unknown>>;
}

export interface TripPlanResponse {
  trip: Trip;
  summary: string;
  budget_analysis: string;
  total_cost: number;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  created_at: string;
}

export interface ChatResponse {
  reply: string;
  conversation_id: string;
}

export interface ItineraryItem {
  time: string;
  type: "activity" | "food" | "transport" | "hotel";
  description: string;
  cost: number;
}

export interface ItineraryDay {
  day_number: number;
  travel_date: string | null;
  title: string;
  items: ItineraryItem[];
}
