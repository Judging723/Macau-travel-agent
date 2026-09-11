<template>
  <div v-if="bootstrapping" class="center-screen">
    <div class="loading-mark">✈️</div>
    <p>正在连接澳门旅行规划服务...</p>
  </div>

  <main v-else-if="!authStore.isAuthenticated || !authStore.user" class="auth-page">
    <section class="auth-intro">
      <div class="brand-mark">✈️</div>
      <p class="eyebrow">MACAO TRAVEL AGENT</p>
      <h1>把澳门旅行想法，交给智能体变成计划</h1>
      <p class="auth-description">
        基于大语言模型、LangGraph 与 RAG 的澳门个性化旅行规划应用。
      </p>
      <div class="feature-list">
        <span>🗺️ 行程规划</span>
        <span>🏛️ 澳门知识检索</span>
        <span>💬 连续旅行问答</span>
      </div>
    </section>

    <section class="auth-card panel">
      <div class="auth-tabs">
        <button :class="{ active: authMode === 'login' }" @click="authMode = 'login'">
          登录
        </button>
        <button :class="{ active: authMode === 'register' }" @click="authMode = 'register'">
          注册
        </button>
      </div>

      <el-form label-position="top" @submit.prevent="handleAuth">
        <el-form-item v-if="authMode === 'register'" label="昵称">
          <el-input v-model="authForm.displayName" placeholder="例如：Alice" size="large" />
        </el-form-item>
        <el-form-item label="邮箱">
          <el-input
            v-model="authForm.email"
            type="email"
            placeholder="alice@example.com"
            size="large"
          />
        </el-form-item>
        <el-form-item label="密码">
          <el-input
            v-model="authForm.password"
            type="password"
            placeholder="至少 8 个字符"
            show-password
            size="large"
          />
        </el-form-item>
        <el-button
          class="auth-submit"
          type="primary"
          native-type="submit"
          size="large"
          :loading="authLoading"
        >
          {{ authMode === "login" ? "进入工作台" : "创建账户" }}
        </el-button>
      </el-form>
    </section>
  </main>

  <div v-else class="app-shell">
    <aside class="sidebar">
      <div class="sidebar-brand">
        <span class="sidebar-logo">✈️</span>
        <div>
          <strong>澳门旅行</strong>
          <small>Agent Console</small>
        </div>
      </div>

      <nav class="nav-list">
        <RouterLink v-for="item in navigation" :key="item.path" :to="item.path">
          <component :is="item.icon" />
          <span>{{ item.label }}</span>
        </RouterLink>
      </nav>

      <div class="sidebar-user">
        <div class="avatar">{{ userInitial }}</div>
        <div class="user-copy">
          <strong>{{ authStore.user.display_name || "旅行者" }}</strong>
          <small>{{ authStore.user.email }}</small>
        </div>
        <button class="icon-button" title="退出登录" @click="logout">
          <SwitchButton />
        </button>
      </div>
    </aside>

    <main class="main-content">
      <RouterView />
    </main>
  </div>
</template>

<script setup lang="ts">
import { ElMessage } from "element-plus";
import { computed, onBeforeUnmount, onMounted, reactive, ref } from "vue";
import { RouterLink, RouterView, useRouter } from "vue-router";

import { authApi, getApiErrorMessage } from "@/api";
import { useAuthStore, useChatStore, useTripStore } from "@/stores/app";

const router = useRouter();
const authStore = useAuthStore();
const tripStore = useTripStore();
const chatStore = useChatStore();

const bootstrapping = ref(true);
const authLoading = ref(false);
const authMode = ref<"login" | "register">("login");
const authForm = reactive({
  displayName: "",
  email: "",
  password: "",
});

const navigation = [
  { path: "/", label: "工作台", icon: "House" },
  { path: "/planner", label: "行程规划", icon: "MagicStick" },
  { path: "/chat", label: "澳门助手", icon: "ChatDotRound" },
];

const userInitial = computed(() => {
  const name = authStore.user?.display_name || authStore.user?.email || "T";
  return name.slice(0, 1).toUpperCase();
});

async function handleAuth() {
  const email = authForm.email.trim();
  if (!email || authForm.password.length < 8) {
    ElMessage.warning("请输入有效邮箱，密码至少需要 8 个字符");
    return;
  }

  authLoading.value = true;
  try {
    if (authMode.value === "register") {
      await authApi.register(email, authForm.password, authForm.displayName.trim());
    }

    const token = await authApi.login(email, authForm.password);
    localStorage.setItem("access_token", token.access_token);
    const user = await authApi.me();
    authStore.login(user, token.access_token);
    authForm.password = "";
    ElMessage.success(authMode.value === "login" ? "登录成功" : "注册成功");
  } catch (error) {
    ElMessage.error(getApiErrorMessage(error));
  } finally {
    authLoading.value = false;
  }
}

async function restoreSession() {
  if (!authStore.token) {
    bootstrapping.value = false;
    return;
  }

  try {
    authStore.setUser(await authApi.me());
  } catch {
    authStore.logout();
  } finally {
    bootstrapping.value = false;
  }
}

function logout() {
  authStore.logout();
  tripStore.setTrips([]);
  tripStore.setCurrentTrip(null);
  chatStore.clearMessages();
  router.push("/");
}

function handleExpiredSession() {
  if (authStore.isAuthenticated) {
    logout();
    ElMessage.warning("登录已过期，请重新登录");
  }
}

onMounted(() => {
  window.addEventListener("auth-expired", handleExpiredSession);
  restoreSession();
});

onBeforeUnmount(() => {
  window.removeEventListener("auth-expired", handleExpiredSession);
});
</script>
