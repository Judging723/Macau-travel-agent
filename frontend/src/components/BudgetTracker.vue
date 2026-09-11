<template>
  <section class="budget-tracker panel">
    <div class="section-title compact">
      <div>
        <p class="eyebrow">BUDGET</p>
        <h2>预算跟踪</h2>
      </div>
      <strong>¥{{ totalSpent.toLocaleString() }} / ¥{{ totalBudget.toLocaleString() }}</strong>
    </div>

    <div v-for="item in items" :key="item.category" class="budget-row">
      <div>
        <span>{{ item.category }}</span>
        <span>¥{{ item.spent.toLocaleString() }} / ¥{{ item.budget.toLocaleString() }}</span>
      </div>
      <el-progress
        :percentage="percentage(item.spent, item.budget)"
        :show-text="false"
        :stroke-width="8"
        :color="item.spent > item.budget ? '#ef6a76' : '#4f8cff'"
      />
    </div>
  </section>
</template>

<script setup lang="ts">
import { computed } from "vue";

interface BudgetItem {
  category: string;
  budget: number;
  spent: number;
}

const props = defineProps<{ items: BudgetItem[] }>();
const totalSpent = computed(() => props.items.reduce((sum, item) => sum + item.spent, 0));
const totalBudget = computed(() => props.items.reduce((sum, item) => sum + item.budget, 0));

function percentage(spent: number, budget: number) {
  if (budget <= 0) return spent > 0 ? 100 : 0;
  return Math.min(100, Math.round((spent / budget) * 100));
}
</script>
