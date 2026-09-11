import * as Icons from "@element-plus/icons-vue";
import ElementPlus from "element-plus";
import "element-plus/dist/index.css";
import "element-plus/theme-chalk/dark/css-vars.css";
import { createPinia } from "pinia";
import { createApp } from "vue";

import App from "./App.vue";
import router from "./router";
import "./style.css";

document.documentElement.classList.add("dark");

const app = createApp(App);

for (const [name, component] of Object.entries(Icons)) {
  app.component(name, component);
}

app.use(createPinia());
app.use(router);
app.use(ElementPlus);
app.mount("#app");
