<template>
  <div class="page-stack">
    <header class="page-header">
      <div>
        <p class="eyebrow">TRIP WORKSPACE</p>
        <h1>澳门行程规划</h1>
        <p>目的地固定为澳门，只需填写日期、预算、人数和偏好。</p>
      </div>
    </header>

    <section class="planner-layout">
      <div class="panel planner-form-card">
        <div class="section-title compact">
          <div>
            <p class="eyebrow">NEW TRIP</p>
            <h2>创建行程</h2>
          </div>
        </div>

        <el-form label-position="top">
          <div class="form-grid two-columns">
            <el-form-item label="行程名称">
              <el-input v-model="form.title" placeholder="例如：澳门三日文化之旅" />
            </el-form-item>
            <el-form-item label="目的地">
              <el-input model-value="澳门特别行政区" disabled />
            </el-form-item>
          </div>

          <div class="form-grid two-columns">
            <el-form-item label="出行日期（可选）">
              <el-date-picker
                v-model="form.dates"
                type="daterange"
                value-format="YYYY-MM-DD"
                start-placeholder="开始日期"
                end-placeholder="结束日期"
                range-separator="至"
              />
            </el-form-item>
          </div>

          <div class="form-grid two-columns">
            <el-form-item label="总预算（元）">
              <el-input-number v-model="form.budget" :min="0" :step="500" controls-position="right" />
            </el-form-item>
            <el-form-item label="同行人数">
              <el-input-number v-model="form.numPeople" :min="1" :max="100" controls-position="right" />
            </el-form-item>
          </div>

          <el-form-item label="兴趣偏好（用逗号分隔）">
            <el-input v-model="form.interests" placeholder="博物馆，美食，城市漫步" />
          </el-form-item>

          <el-button type="primary" size="large" :loading="creating" @click="createTrip">
            保存行程
          </el-button>
        </el-form>
      </div>

      <div class="panel ai-planner-card">
        <div class="ai-card-title">
          <span>✨</span>
          <div>
            <h2>AI 规划建议</h2>
            <p>调用已完成的旅行 Agent，而不是浏览器端生成模板。</p>
          </div>
        </div>
        <el-input
          v-model="aiQuestion"
          type="textarea"
          :rows="5"
          placeholder="补充你的要求，例如：希望每天安排不要太满，优先博物馆和本地美食。"
        />
        <el-button
          class="ai-submit"
          type="primary"
          :loading="askingAi"
          :disabled="!tripStore.currentTrip"
          @click="askAgent"
        >
          让 Agent 生成建议
        </el-button>
        <p v-if="!tripStore.currentTrip" class="muted-text">
          请先创建或选择一条行程，再生成结构化计划。
        </p>
        <div v-if="aiAnswer" class="ai-answer">{{ aiAnswer }}</div>
      </div>
    </section>

    <section class="panel trip-list-panel">
      <div class="section-title compact">
        <div>
          <p class="eyebrow">SAVED TRIPS</p>
          <h2>我的行程</h2>
        </div>
        <el-button text :loading="loadingTrips" @click="loadTrips">刷新</el-button>
      </div>

      <div v-if="tripStore.trips.length" class="saved-trip-grid">
        <article
          v-for="trip in tripStore.trips"
          :key="trip.id"
          class="saved-trip"
          :class="{ selected: tripStore.currentTrip?.id === trip.id }"
          @click="tripStore.setCurrentTrip(trip)"
        >
          <div class="saved-trip-topline">
            <span>📍 {{ trip.destination }}</span>
            <el-select
              v-model="trip.status"
              size="small"
              @click.stop
              @change="updateStatus(trip)"
            >
              <el-option label="草稿" value="draft" />
              <el-option label="已确认" value="confirmed" />
              <el-option label="已完成" value="completed" />
            </el-select>
          </div>
          <h3>{{ trip.title }}</h3>
          <p>{{ formatTripMeta(trip) }}</p>
          <div class="saved-trip-footer">
            <span>{{ trip.budget == null ? "未设置预算" : `¥${trip.budget.toLocaleString()}` }}</span>
            <button class="danger-text" @click.stop="removeTrip(trip)">删除</button>
          </div>
        </article>
      </div>
      <div v-else-if="!loadingTrips" class="empty-state">还没有行程，请先创建一条。</div>
    </section>

    <section v-if="tripStore.currentTrip" class="panel trip-detail-panel">
      <div class="section-title compact">
        <div>
          <p class="eyebrow">CURRENT TRIP</p>
          <h2>{{ tripStore.currentTrip.title }}</h2>
        </div>
        <span class="budget-number">
          {{ tripStore.currentTrip.budget == null ? "预算未设置" : `预算 ¥${tripStore.currentTrip.budget.toLocaleString()}` }}
        </span>
      </div>
      <TripTimeline v-if="timelineDays.length" :days="timelineDays" />
      <div v-else class="empty-state">该行程还没有结构化 itinerary，可通过后端更新接口写入。</div>
    </section>
  </div>
</template>

<script setup lang="ts">
import { ElMessage, ElMessageBox } from "element-plus";
import { computed, onMounted, reactive, ref } from "vue";

import { getApiErrorMessage, tripApi } from "@/api";
import TripTimeline from "@/components/TripTimeline.vue";
import { useTripStore } from "@/stores/app";
import type { Trip, TripCreate, TripStatus } from "@/types";

const tripStore = useTripStore();
const creating = ref(false);
const loadingTrips = ref(false);
const askingAi = ref(false);
const aiQuestion = ref("");
const aiAnswer = ref("");

const form = reactive({
  title: "",
  dates: [] as string[],
  budget: undefined as number | undefined,
  numPeople: 2,
  interests: "",
});

const timelineDays = computed(() => tripStore.currentTrip?.itinerary ?? []);

async function loadTrips() {
  loadingTrips.value = true;
  try {
    const trips = await tripApi.list();
    tripStore.setTrips(trips);
    if (!tripStore.currentTrip && trips.length) {
      tripStore.setCurrentTrip(trips[0]);
    }
  } catch (error) {
    ElMessage.error(getApiErrorMessage(error));
  } finally {
    loadingTrips.value = false;
  }
}

async function createTrip() {
  if (!form.title.trim()) {
    ElMessage.warning("请填写行程名称");
    return;
  }

  const payload: TripCreate = {
    title: form.title.trim(),
    start_date: form.dates[0] || undefined,
    end_date: form.dates[1] || undefined,
    budget: form.budget,
    num_people: form.numPeople,
    preferences: {
      interests: form.interests
        .split(/[，,]/)
        .map((item) => item.trim())
        .filter(Boolean),
    },
  };

  creating.value = true;
  try {
    const trip = await tripApi.create(payload);
    tripStore.setTrips([trip, ...tripStore.trips]);
    tripStore.setCurrentTrip(trip);
    aiQuestion.value = `请为“${trip.title}”提供一份符合预算和兴趣偏好的旅行规划。`;
    form.title = "";
    form.dates = [];
    form.budget = undefined;
    form.interests = "";
    ElMessage.success("行程已保存");
  } catch (error) {
    ElMessage.error(getApiErrorMessage(error));
  } finally {
    creating.value = false;
  }
}

async function updateStatus(trip: Trip) {
  try {
    const updated = await tripApi.update(trip.id, { status: trip.status as TripStatus });
    Object.assign(trip, updated);
    if (tripStore.currentTrip?.id === updated.id) tripStore.setCurrentTrip(updated);
    ElMessage.success("行程状态已更新");
  } catch (error) {
    ElMessage.error(getApiErrorMessage(error));
    await loadTrips();
  }
}

async function removeTrip(trip: Trip) {
  try {
    await ElMessageBox.confirm(`确定删除“${trip.title}”吗？`, "删除行程", {
      confirmButtonText: "删除",
      cancelButtonText: "取消",
      type: "warning",
    });
    await tripApi.remove(trip.id);
    const remaining = tripStore.trips.filter((item) => item.id !== trip.id);
    tripStore.setTrips(remaining);
    if (tripStore.currentTrip?.id === trip.id) tripStore.setCurrentTrip(remaining[0] ?? null);
    ElMessage.success("行程已删除");
  } catch (error) {
    if (error === "cancel" || error === "close") return;
    ElMessage.error(getApiErrorMessage(error));
  }
}

async function askAgent() {
  const current = tripStore.currentTrip;
  if (!current) {
    ElMessage.warning("请先创建或选择一条行程");
    return;
  }

  askingAi.value = true;
  aiAnswer.value = "";
  try {
    const result = await tripApi.generatePlan(
      current.id,
      aiQuestion.value.trim() || undefined,
    );

    tripStore.setCurrentTrip(result.trip);
    tripStore.setTrips(
      tripStore.trips.map((trip) =>
        trip.id === result.trip.id ? result.trip : trip,
      ),
    );

    aiAnswer.value = [
      result.summary,
      `预计总花费：¥${result.total_cost.toLocaleString()}`,
      `预算分析：${result.budget_analysis}`,
    ].join("\n\n");

    ElMessage.success("AI 行程已生成并保存");
  } catch (error) {
    ElMessage.error(getApiErrorMessage(error));
  } finally {
    askingAi.value = false;
  }
}

function formatTripMeta(trip: Trip) {
  const dates = trip.start_date
    ? `${trip.start_date}${trip.end_date ? ` 至 ${trip.end_date}` : ""}`
    : "日期待定";
  return `${dates} · ${trip.num_people} 人`;
}

onMounted(loadTrips);
</script>
