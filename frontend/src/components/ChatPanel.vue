<template>
  <section class="chat-panel panel">
    <header class="chat-header">
      <div class="assistant-avatar">✈️</div>
      <div>
        <h1>澳门旅行助手</h1>
        <p>DeepSeek · LangGraph · RAG</p>
      </div>
      <el-button v-if="chatStore.messages.length" text @click="chatStore.clearMessages">
        新对话
      </el-button>
    </header>

    <div ref="messagesRef" class="message-list">
      <div v-if="!chatStore.messages.length" class="chat-welcome">
        <div class="welcome-orbit">🌍</div>
        <h2>今天想怎样游澳门？</h2>
        <p>我可以结合澳门官方资料和实时网页信息，帮你规划预算、路线与旅行重点。</p>
        <div class="quick-prompts">
          <button v-for="prompt in quickPrompts" :key="prompt" @click="sendQuickPrompt(prompt)">
            {{ prompt }}
          </button>
        </div>
      </div>

      <article
        v-for="message in chatStore.messages"
        :key="message.id"
        class="message-row"
        :class="message.role"
      >
        <div v-if="message.role === 'assistant'" class="message-avatar">✈️</div>
        <div class="message-bubble">{{ message.content }}</div>
      </article>

      <article v-if="chatStore.isGenerating" class="message-row assistant">
        <div class="message-avatar">✈️</div>
        <div class="message-bubble typing-indicator">
          <i></i><i></i><i></i>
        </div>
      </article>
    </div>

    <footer class="chat-composer">
      <el-input
        v-model="input"
        type="textarea"
        :autosize="{ minRows: 2, maxRows: 5 }"
        resize="none"
        placeholder="描述你的澳门旅行需求，Enter 发送，Shift + Enter 换行"
        :disabled="chatStore.isGenerating"
        @keydown.enter.exact.prevent="send"
      />
      <el-button
        class="send-button"
        type="primary"
        :disabled="!input.trim() || chatStore.isGenerating"
        :loading="chatStore.isGenerating"
        @click="send"
      >
        <Promotion />
      </el-button>
    </footer>
  </section>
</template>

<script setup lang="ts">
import { ElMessage } from "element-plus";
import { nextTick, ref, watch } from "vue";

import { chatApi, getApiErrorMessage } from "@/api";
import { useChatStore } from "@/stores/app";

const chatStore = useChatStore();
const input = ref("");
const messagesRef = ref<HTMLDivElement | null>(null);

const quickPrompts = [
  "结合已有资料推荐澳门历史城区路线",
  "查看我的澳门行程并评估预算",
  "帮我规划一次以世界遗产和澳门美食为主的旅行",
];

watch(
  () => [chatStore.messages.length, chatStore.isGenerating],
  async () => {
    await nextTick();
    messagesRef.value?.scrollTo({ top: messagesRef.value.scrollHeight, behavior: "smooth" });
  },
);

async function send() {
  const message = input.value.trim();
  if (!message || chatStore.isGenerating) return;

  chatStore.addMessage({
    id: crypto.randomUUID(),
    role: "user",
    content: message,
    created_at: new Date().toISOString(),
  });
  input.value = "";
  chatStore.isGenerating = true;

  try {
    const response = await chatApi.sendMessage(message, chatStore.conversationId);
    chatStore.conversationId = response.conversation_id;
    chatStore.addMessage({
      id: crypto.randomUUID(),
      role: "assistant",
      content: response.reply,
      created_at: new Date().toISOString(),
    });
  } catch (error) {
    ElMessage.error(getApiErrorMessage(error));
  } finally {
    chatStore.isGenerating = false;
  }
}

function sendQuickPrompt(prompt: string) {
  input.value = prompt;
  void send();
}
</script>
