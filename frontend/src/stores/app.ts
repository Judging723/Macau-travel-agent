import { defineStore } from "pinia";
import { computed, ref } from "vue";

import type { ChatMessage, Trip, User } from "@/types";

export const useAuthStore = defineStore("auth", () => {
  const user = ref<User | null>(null);
  const token = ref<string | null>(localStorage.getItem("access_token"));
  const isAuthenticated = computed(() => Boolean(token.value));

  function login(currentUser: User, accessToken: string) {
    user.value = currentUser;
    token.value = accessToken;
    localStorage.setItem("access_token", accessToken);
  }

  function setUser(currentUser: User) {
    user.value = currentUser;
  }

  function logout() {
    user.value = null;
    token.value = null;
    localStorage.removeItem("access_token");
  }

  return { user, token, isAuthenticated, login, setUser, logout };
});

export const useTripStore = defineStore("trip", () => {
  const trips = ref<Trip[]>([]);
  const currentTrip = ref<Trip | null>(null);

  function setTrips(value: Trip[]) {
    trips.value = value;
  }

  function setCurrentTrip(value: Trip | null) {
    currentTrip.value = value;
  }

  return {
    trips,
    currentTrip,
    setTrips,
    setCurrentTrip,
  };
});

export const useChatStore = defineStore("chat", () => {
  const messages = ref<ChatMessage[]>([]);
  const isGenerating = ref(false);
  const conversationId = ref<string | null>(null);

  function addMessage(message: ChatMessage) {
    messages.value.push(message);
  }

  function clearMessages() {
    messages.value = [];
    conversationId.value = null;
  }

  return { messages, isGenerating, conversationId, addMessage, clearMessages };
});
