import { createRouter, createWebHistory } from "vue-router";

const routes = [
  {
    path: "/",
    name: "Dashboard",
    component: () => import("@/views/Dashboard.vue"),
  },
  {
    path: "/planner",
    name: "TripPlanner",
    component: () => import("@/views/TripPlanner.vue"),
  },
  {
    path: "/chat",
    name: "Chat",
    component: () => import("@/views/Chat.vue"),
  },
];

export default createRouter({
  history: createWebHistory(),
  routes,
});
