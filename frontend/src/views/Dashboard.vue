<template>
  <div class="page-stack">
    <header class="page-header dashboard-heading">
      <div>
        <p class="eyebrow">OVERVIEW</p>
        <h1>你好，{{ authStore.user?.display_name || "旅行者" }}</h1>
        <p>从一句澳门旅行想法开始。</p>
      </div>
      <RouterLink class="primary-link" to="/chat">
        <ChatDotRound />
        与澳门旅行助手对话
      </RouterLink>
    </header>

    <section class="stats-grid">
      <article v-for="stat in stats" :key="stat.label" class="stat-card panel">
        <div class="stat-icon">{{ stat.icon }}</div>
        <div>
          <strong>{{ stat.value }}</strong>
          <span>{{ stat.label }}</span>
        </div>
      </article>
    </section>

    <section>
      <div class="section-title">
        <div>
          <p class="eyebrow">QUICK START</p>
          <h2>选择一个入口</h2>
        </div>
      </div>
      <div class="action-grid">
        <RouterLink v-for="action in actions" :key="action.path" :to="action.path" class="action-card panel">
          <span class="action-icon">{{ action.icon }}</span>
          <div>
            <h3>{{ action.label }}</h3>
            <p>{{ action.description }}</p>
          </div>
          <ArrowRight class="action-arrow" />
        </RouterLink>
      </div>
    </section>

    <section class="panel recent-panel">
      <div class="section-title compact">
        <div>
          <p class="eyebrow">RECENT TRIPS</p>
          <h2>最近行程</h2>
        </div>
        <RouterLink to="/planner">查看全部</RouterLink>
      </div>

      <div v-if="loading" class="empty-state">正在加载行程...</div>
      <div v-else-if="recentTrips.length" class="recent-list">
        <article v-for="trip in recentTrips" :key="trip.id" class="recent-trip">
          <div class="trip-pin">📍</div>
          <div class="recent-trip-copy">
            <strong>{{ trip.title }}</strong>
            <span>{{ trip.destination }} · {{ trip.num_people }} 人</span>
          </div>
          <span class="status-pill" :class="trip.status">{{ statusText[trip.status] }}</span>
        </article>
      </div>
      <div v-else class="empty-state">
        还没有保存的行程，前往“行程规划”创建第一条记录。
      </div>
    </section>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from "vue";
import { RouterLink } from "vue-router";

import { tripApi } from "@/api";
import { useAuthStore, useTripStore } from "@/stores/app";
import type { TripStatus } from "@/types";

const authStore = useAuthStore();
const tripStore = useTripStore();
const loading = ref(true);

const statusText: Record<TripStatus, string> = {
  draft: "草稿",
  confirmed: "已确认",
  completed: "已完成",
};

const stats = computed(() => [
  { icon: "🧳", value: tripStore.trips.length, label: "我的行程" },
  {
    icon: "✅",
    value: tripStore.trips.filter((trip) => trip.status === "confirmed").length,
    label: "已确认行程",
  },
  { icon: "🤖", value: "在线", label: "澳门 AI 助手" },
]);

const actions = [
  { icon: "✨", label: "规划新行程", description: "保存旅行日期、预算和偏好", path: "/planner" },
  { icon: "💬", label: "咨询澳门助手", description: "通过连续对话完善澳门旅行方案", path: "/chat" },
];

const recentTrips = computed(() => tripStore.trips.slice(0, 4));

onMounted(async () => {
  try {
    tripStore.setTrips(await tripApi.list());
  } finally {
    loading.value = false;
  }
});
</script>
