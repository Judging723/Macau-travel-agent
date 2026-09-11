<template>
  <div class="timeline">
    <article v-for="(day, dayIndex) in days" :key="dayIndex" class="timeline-day">
      <div class="timeline-marker">
        <span>{{ dayIndex + 1 }}</span>
        <i v-if="dayIndex < days.length - 1"></i>
      </div>
      <div class="timeline-content">
        <h3>
          {{ day.title || `第 ${day.day_number} 天` }}
          <small v-if="day.travel_date">{{ day.travel_date }}</small>
        </h3>
        <div v-if="day.items.length" class="timeline-items">
          <div v-for="(item, itemIndex) in day.items" :key="itemIndex" class="timeline-item">
            <div class="timeline-item-topline">
              <span>{{ item.time || "时间待定" }}</span>
              <el-tag v-if="item.type" size="small" effect="plain">{{ item.type }}</el-tag>
            </div>
            <p>{{ item.description || "暂无活动说明" }}</p>
            <strong v-if="item.cost != null">¥{{ item.cost }}</strong>
          </div>
        </div>
        <p v-else class="muted-text">当天暂时没有活动明细。</p>
      </div>
    </article>
  </div>
</template>

<script setup lang="ts">
import type { ItineraryDay } from "@/types";

defineProps<{ days: ItineraryDay[] }>();
</script>
