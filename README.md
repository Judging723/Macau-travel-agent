# 澳门旅游智能体

一个聚焦澳门特别行政区的大模型应用开发项目。它不是通用聊天壳，而是把用户账户、行程数据、可追溯知识库、实时网页搜索、Agent 工具调用和 Self-Reflection 行程生成组合成一条完整链路。

## 核心能力

- 创建、查询、修改和删除澳门行程。
- 基于 LangGraph 工具调用的多轮旅行助手。
- 基于澳门特别行政区政府旅游局公开资料的 RAG 知识库。
- PDF 文本提取及 OCR 回退，支持 Markdown 和 TXT 资料。
- Dense + BM25 + RRF 混合检索，再使用 BGE CrossEncoder 重排。
- DeepSeek 生成行程，然后执行 Self-Reflection 和 Python 确定性质量检查。
- RAG、Agent 回答和行程质量三层评测。

## 系统架构

项目分为前端、API、Agent、RAG、数据和外部服务六层。

```mermaid
flowchart TB
    User[用户]

    subgraph Frontend[Frontend · Vue 3]
        UI[登录 / 行程管理 / AI 对话]
    end

    subgraph API[API Gateway · FastAPI]
        AuthAPI[Auth API]
        TripAPI[Trip API]
        ChatAPI[Chat API]
    end

    subgraph Agent[Agent Layer · LangGraph]
        OA[TravelOrchestrator<br/>ReAct 工具调用]
        TripsTool[list_my_trips]
        KnowledgeTool[search_travel_knowledge]
        WebTool[search_web]
        PlanTool[plan_trip]
        TA[TripAgent<br/>Self-Reflection]
    end

    subgraph RAG[RAG Layer]
        Dense[Dense 向量召回]
        Sparse[BM25 关键词召回]
        Fusion[RRF 融合]
        Rerank[BGE Reranker]
        Dense --> Fusion
        Sparse --> Fusion
        Fusion --> Rerank
    end

    subgraph Data[Data Layer]
        PG[(PostgreSQL<br/>用户 / 行程 / 对话状态)]
        Chroma[(ChromaDB<br/>知识片段 / 向量)]
    end

    subgraph External[External Services]
        DeepSeek[DeepSeek]
        Tavily[Tavily Search]
        Ollama[Ollama<br/>Qwen3 Embedding]
    end

    User --> UI
    UI --> AuthAPI
    UI --> TripAPI
    UI --> ChatAPI
    AuthAPI --> PG
    TripAPI --> PG
    ChatAPI --> OA
    OA <--> DeepSeek
    OA --> TripsTool
    OA --> KnowledgeTool
    OA --> WebTool
    OA --> PlanTool
    OA -.对话 checkpoint.-> PG
    TripsTool --> PG
    KnowledgeTool --> Dense
    KnowledgeTool --> Sparse
    Chroma --> Dense
    Chroma --> Sparse
    Ollama --> Dense
    Rerank --> OA
    WebTool --> Tavily
    PlanTool <--> TA
    TA <--> DeepSeek
    TripAPI <--> TA
```

### Agent 编排流程

```mermaid
flowchart TD
    Input[用户问题] --> OA[TravelOrchestrator]
    OA --> Decide{DeepSeek 选择工具}
    Decide -->|查询已保存行程| Trips[list_my_trips]
    Decide -->|静态澳门知识| Knowledge[search_travel_knowledge]
    Decide -->|实时信息| Web[search_web]
    Decide -->|完整行程规划| Plan[plan_trip]
    Trips --> OA
    Knowledge --> OA
    Web --> OA
    Plan --> Draft[TripAgent 生成草案]
    Draft --> Reflect[模型 Self-Reflection]
    Reflect --> Gate{Python 规则校验}
    Gate -->|不通过且未达上限| Revise[根据反馈修订]
    Revise --> Reflect
    Gate -->|通过或达到上限| Result[结构化行程]
    Result --> OA
    OA --> Answer[最终回答]
```

Agent 层包含两种不同范式：

- `TravelOrchestrator`：LangGraph 的工具调用循环，根据问题决定调用哪个工具。
- `TripAgent`：Self-Reflection，先生成草案，再反思，不合格时进行有上限的修订。Python 负责活动数量、日期和费用等可确定规则，避免仅依赖模型自评。

### RAG 检索流程

```mermaid
flowchart LR
    Query[用户查询]

    Query --> Embed[Ollama 查询向量化]
    Embed --> Dense[ChromaDB Dense 召回 Top N]

    Query --> Tokenize[Jieba 分词]
    Tokenize --> BM25[BM25 召回 Top N]

    Dense --> RRF[RRF 排名融合]
    BM25 --> RRF
    RRF --> BGE[BGE CrossEncoder 重排]
    BGE --> TopK[返回 Top K]
    TopK --> Tool[search_travel_knowledge]
    Tool --> AgentAnswer[Agent 生成有来源的回答]

    Chroma[(ChromaDB 已入库片段)] --> Dense
    Chroma -.构建语料索引.-> BM25
```

## 技术栈

- 后端：Python 3.11、FastAPI、Pydantic、SQLAlchemy Async、Alembic
- Agent：DeepSeek、LangChain、LangGraph、PostgreSQL Checkpointer
- RAG：Ollama `qwen3-embedding:4b`、ChromaDB、Jieba、BM25、RRF、`BAAI/bge-reranker-v2-m3`
- 文档解析：PyPDF、PyMuPDF、RapidOCR
- 实时搜索：Tavily
- 前端：Vue 3、TypeScript、Vite、Pinia、Element Plus

### 技术选型边界

| 当前选择 | 选择原因 | 什么时候考虑替换 |
| --- | --- | --- |
| FastAPI + Pydantic | 异步 API、类型校验和 OpenAPI 集成简单，适合模型服务 | 需要完整后台管理系统时考虑 Django；极致吞吐的网关层再考虑 Go |
| PostgreSQL + SQLAlchemy | 用户、行程和对话状态需要事务与关系查询，LangGraph 也有 PostgreSQL Checkpointer | 只读分析场景可增加数仓；Redis 只适合缓存，不替代主数据库 |
| ChromaDB | 当前仅 14 份资料、数百片段，易于本地部署和学习 | 数据达到十万级、需要多租户和分布式扩缩容时评估 Milvus 或 Qdrant |
| 进程内 BM25 + RRF | 小语料无需维护额外搜索集群，能补足向量检索的关键词召回 | 语料或实例增多、索引更新频繁时迁移 OpenSearch/Elasticsearch |
| Ollama + Qwen3 Embedding | 本地可复现、无按次费用，适合个人项目 | 生产环境追求并发和 SLA 时使用 GPU 推理服务或托管 Embedding API |
| BGE Reranker | 中文效果稳定，102 条评测显著提升首位命中 | 延迟成为硬约束时评测更小模型、量化或独立 GPU 推理服务 |
| LangGraph | 需要工具循环、PostgreSQL checkpoint 和有界状态流 | 如果只保留固定的单次调用流程，普通 Python 函数会更简单 |
| Tavily | 用于开放时间、票价等实时网页信息，避免把动态数据固化入库 | 正式产品应优先接入景区、天气和交通官方 API，搜索仅作补充 |

## 项目结构

```text
Travel Agent/
├── backend/
│   ├── app/
│   │   ├── agents/       # Orchestrator 与 Self-Reflection 行程 Agent
│   │   ├── api/          # 认证、行程、聊天 API
│   │   ├── db/           # 异步数据库会话
│   │   ├── models/       # User 和 Trip 数据模型
│   │   └── rag/          # 文档加载、分块、检索、重排
│   ├── data/knowledge/        # 可追溯的 PDF/Markdown 知识资料
│   ├── eval/                  # 三层评测集和评测脚本
│   ├── migrations/            # Alembic 数据库迁移
│   └── knowledge_manifest.json
├── frontend/                       # Vue 管理界面
└── compose.yaml                    # PostgreSQL 和 ChromaDB
```

## 本地启动

### 1. 准备环境

需要 Python 3.11、Node.js、Docker、Git 和 [Ollama](https://ollama.com/)。

```bash
docker compose up -d postgres chromadb
ollama pull qwen3-embedding:4b
```

### 2. 安装后端

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e ".[dev]"
cp .env.example .env
```

在 `backend/.env` 中至少填写：

```dotenv
TRAVEL_AGENT_JWT_SECRET_KEY=一个随机长字符串
TRAVEL_AGENT_DEEPSEEK_API_KEY=你的_DeepSeek_API_Key
TRAVEL_AGENT_TAVILY_API_KEY=你的_Tavily_API_Key
```

可以用下面的命令生成 JWT 密钥：

```bash
python3 -c "import secrets; print(secrets.token_urlsafe(48))"
```

### 3. 初始化数据库和知识库

```bash
alembic upgrade head
python3 -m app.rag.rag_service
```

第二条命令会扫描 `data/knowledge/`，按 `knowledge_manifest.json` 读取资料，执行解析、OCR、分块、去重、向量化和入库。首次启动 API 时会自动初始化 LangGraph checkpoint 表。

### 4. 启动前后端

后端：

```bash
uvicorn app.main:app --reload
```

前端：

```bash
cd ../frontend
npm ci
npm run dev
```

打开 `http://127.0.0.1:5173`。API 文档位于 `http://127.0.0.1:8000/docs`。

## API 概览

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| GET | `/api/v1/health` | 健康检查 |
| POST | `/api/v1/auth/register` | 注册 |
| POST | `/api/v1/auth/login` | 登录并获取 JWT |
| GET | `/api/v1/auth/me` | 获取当前用户 |
| GET/POST | `/api/v1/trips` | 查询或创建行程 |
| GET/PATCH/DELETE | `/api/v1/trips/{trip_id}` | 行程详情、更新和删除 |
| POST | `/api/v1/trips/{trip_id}/generate-plan` | 生成并保存结构化行程 |
| POST | `/api/v1/chat` | 多轮 Agent 问答 |

## 知识库与数据来源

当前知识库由 14 份澳门主题资料组成，涵盖综合指南、世界遗产、博物馆、社区步行、美食、交通、口岸、入境、住宿、无障碍和安全信息。资料标题、发布机构、原始 URL 和获取日期记录在 `backend/knowledge_manifest.json`。

静态知识由本地知识库回答；开放时间、票价、天气等可变信息由 Tavily 实时搜索，避免把易过期内容固化在向量库中。

## 测试与评测

单元测试与代码检查：

```bash
cd backend
pytest -q
ruff check app tests eval
```

RAG 评测：

```bash
python3 -m eval.run_rag_eval --mode dense
python3 -m eval.run_rag_eval --mode hybrid
python3 -m eval.run_rag_eval --mode rerank
```

Agent 和行程评测不会自动登录。先手动登录并设置临时令牌，再运行：

```bash
export TRAVEL_AGENT_EVAL_TOKEN="你登录后获得的_access_token"
python3 -m eval.run_agent_eval
python3 -m eval.run_plan_eval
```

当前本地基线：

| 评测层 | 用例 | Hit@1 | Hit@3 | Hit@5 | MRR@5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| RAG Dense | 102 | 71.57% | 94.12% | 96.08% | 0.8247 |
| RAG Hybrid（Dense + BM25 + RRF） | 102 | 74.51% | 96.08% | 97.06% | 0.8505 |
| RAG Rerank（Hybrid + BGE） | 102 | 91.18% | 100.00% | 100.00% | 0.9542 |

102 条查询由 33 个独立事实点及其人工改写组成，其中包含 12 条容易混淆的困难问法。重排使用 Hybrid Top 20 候选；Top 5 在旧 33 条集合上指标相同，但在扩展集合中会漏掉原始召回排名第 7 至 15 的有效片段，因此最终保留 Top 20。

Agent 回答评测包含知识问答、实时信息、范围拒答、困难问法和无证据拒答共 9 个用例，本轮 9/9 通过；行程质量评测包含 3 个用例，本轮完成的一日文化行程端到端冒烟用例 1/1 通过。由于模型输出具有随机性，发布结果时应使用当前模型配置重新运行并注明评测时间。

项目已配置 GitHub Actions，在 push 和 pull request 时自动执行后端测试、Ruff 检查和前端生产构建。RAG 检索日志会记录 Embedding、Dense、BM25、RRF、Rerank 各阶段延迟与候选数量。

## 已知边界

- 项目只服务澳门，不是全国或全球旅游助手。
- 票价、开放时间、酒店和餐厅信息可能变动，必须以实时网页资料为准。
- 本地 BGE 重排器的首次加载和推理较慢，生产环境可改用 GPU 或独立推理服务。
- Agent 回答具有随机性；项目使用结构化输出、确定性规则和有界重试降低失败概率。
