# Research-Ai-LLM-Everyday

## 🧠 Introduction

This repository is dedicated to the **daily research and practical application of AI technologies**, especially those revolving around **Large Language Models (LLMs)**, **Retrieval-Augmented Generation (RAG)**, and **Multi-Agent Systems**.

It aims to serve as a **personal lab for rapid experimentation and learning**, offering a collection of tools, agents, and patterns useful for both research and real-world applications.

### 🔍 Key Projects and Technologies

- **RAG Techniques and Multi-Agent Orchestration**:  
  Leverage advanced multi-agent, web search, environment, RAG systems for document search, knowledge retrieval, and reasoning. Explore powerful workflows with **Planning (ReAct flow)**, **Reflection**, **Tool Use**, and custom agents like:
  👉 [AI Multi-Agent Orchestrator RAG App with tools](https://github.com/buithanhdam/maowrag-unlimited-ai-agent)

- **Meeting Note Agent**:  
  Summarizes and organizes meeting discussions intelligently:  
  👉 [Meeting Note Tool](https://github.com/buithanhdam/meeting-note-tool)

- **Resume Builder (LLM-based)**:  
  Automate and enhance resume generation using AI:  
  👉 [Resume AI Builder](https://github.com/buithanhdam/resume_ai_builder)

- **Learning Agentic Patterns**:  
  Includes references to core building blocks of agent design like:
  - [Planning Agent](https://blog.langchain.dev/planning-agents/)
  - [Reflection](https://blog.langchain.dev/reflection-agents/)
  - [Parallel Agent](https://google.github.io/adk-docs/agents/workflow-agents/parallel-agents/)
  - [Router Agent](https://github.com/awslabs/agent-squad)
  **and more**

---

## 💡 Project Vision & Future Roadmap

This repository aims to stay at the cutting edge of:
- 🤖 **Autonomous Multi-Agent Systems**: self-guided, multi-step web and document understanding
- 🧩 **Agentic Design Patterns**: Planning, Reflection, Memory, Tool Use
- 🔍 **RAG Techniques**: ragging, retrieval, and retrieval-augmented generation (Hybrid search, Re-rerank, HyDE,...)
- 🛠️ **LLM-Powered Developer Tools** – such as code assistants and document builders
- **Up-to-date AI technologies, development practices**: CI/CD, DevOps, and AI technologies, development practices.

New experiments, integrations, and agent workflows will be continuously added to support the evolving landscape of **AI-first application development**.

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/buithanhdam/research-ai-llm-everyday.git
cd research-ai-llm-everyday
```

### 2. (Optional) Create a virtual environment

- **On Unix/macOS:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```
- **On Windows:**
  ```bash
  python -m venv venv
  .\venv\Scripts\activate
  ```

### 3. Install required dependencies

```bash
pip install -r requirements.txt
```

---

## 🔐 Environment Setup

Create a `.env` file by copying the example file:

```bash
cp .env.example .env
```

Add your API keys:

```env
GOOGLE_API_KEY=
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
TAVILY_API_KEY=

QDRANT_URL=http://localhost:6333
BACKEND_URL=http://localhost:8000
```

---

## ✅ Running Tests

### Prerequisites for Audio/Video Processing

To process audio/video files, FFmpeg is required:

#### For Ubuntu/Debian
```bash
sudo apt update
sudo apt install ffmpeg
```

#### For macOS (Homebrew)
```bash
brew install ffmpeg
```

#### For Windows
1. Download FFmpeg from [FFmpeg official website](https://ffmpeg.org/download.html).
2. Extract the files and add the `bin` folder to your system's PATH.
3. Restart your terminal and verify installation with:
   ```bash
   ffmpeg -version
   ```

Run unit tests using `pytest`:

```bash
pytest tests/
```

Or run individual test files directly:

```bash
python3 tests/llm_test.py
python3 tests/agent_test.py
python3 tests/rag_test.py
...
```

---

## 🚀 Running the Application

### 1. Start the Qdrant vector store with docker

```bash
docker compose -f docker-compose.qdrant.yaml up
```

- Qdrant Host: [http://localhost:6333/dashboard#](http://localhost:6333/dashboard#)

### 2. Start the FastAPI Backend

```bash
uvicorn app_fastapi:app --host 0.0.0.0 --port 8000 --reload
```

- API Documentation: [http://localhost:8000/docs](http://localhost:8000/docs)

### 3. Start the Streamlit Frontend

```bash
streamlit run app_streamlit.py --server.port=8501 --server.address=0.0.0.0
```

- Access the UI: [http://localhost:8501](http://localhost:8501)

---

## 🐳 Run all service with Docker

### 1. Build Docker Images

```bash
docker-compose build
```

### 2. Start Docker Containers

```bash
docker-compose up
```

- FastAPI backend: [http://localhost:8000](http://localhost:8000)  
- Streamlit UI: [http://localhost:8501](http://localhost:8501)
- Qdrant Host: [http://localhost:6333/dashboard#](http://localhost:6333/dashboard#)

### 3. Stop Containers

```bash
docker-compose down
```

---

## 🤝 Contributing

Have ideas or improvements?  
Feel free to fork the repo, create issues, or submit a pull request.

---

## 📄 License

This project is licensed under the MIT License.

---

## 🔗 References

- [Agentic Patterns (Neural Maze)](https://github.com/neural-maze/agentic_patterns/)
- [Multi-Agent Orchestrator (AWS Labs)](https://github.com/awslabs/agent-squad)
- [Google AgentDK Docs](https://google.github.io/adk-docs/)
- [Langchain](https://blog.langchain.dev/)
- [Maowrag unlimited ai agent](https://github.com/buithanhdam/maowrag-unlimited-ai-agent)
