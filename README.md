# RAG-Powered Multi-Agent Q&A Assistant

## Objective
A simple knowledge assistant that:
- **Retrieves** relevant information from a small document collection (RAG)
- **Generates** natural-language answers via an LLM
- **Orchestrates** tool/agent selection with a basic agentic workflow

## Features & Workflow
1. **Data Ingestion**: Loads and chunks 2-3 short documents from the `data/` folder.
2. **Vector Store & Retrieval**: Embeds chunks with Sentence Transformers and stores them in a FAISS index for fast similarity search.
3. **LLM Integration**: Uses OpenAI's GPT (or compatible) to generate answers based on retrieved context.
4. **Agentic Workflow**: Routes queries to a calculator, dictionary, or the RAG pipeline based on keywords. Logs every decision.
5. **Demo Interface**: Minimal CLI for Q&A. Shows which agent/tool was used, the retrieved context, and the final answer.

## File Structure
```
RAG_MultiAgent/
├── data/                  # Sample documents
├── rag_agent.py           # RAG pipeline logic
├── agent_orchestrator.py  # Agentic workflow logic
├── tools.py               # Calculator, dictionary, etc.
├── main.py                # CLI entry point
├── requirements.txt       # Dependencies
└── README.md              # This file
```

## Setup & Usage
1. **Clone the repo and navigate to the folder.**
2. **Create a virtual environment and install dependencies:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
3. **Set your OpenAI API key:**
   - Edit the `.env` file in the project root and add your API key:
     ```env
     OPENAI_API_KEY=sk-...
     ```
   - The project automatically loads environment variables from `.env`.
4. **Run the CLI:**
   ```bash
   python main.py
   ```

## Design Choices
- **Chunking:** Simple split by double newlines for demo purposes.
- **Vector DB:** FAISS for in-memory, fast similarity search.
- **LLM:** OpenAI GPT-3.5-turbo (can be swapped for other chat-completion APIs).
- **Routing:** Keyword-based agent selection for simplicity.

## Extending
- Add more tools/agents by extending `tools.py` and updating `agent_orchestrator.py`.
- Swap in a web UI (e.g., Streamlit) for the CLI in `main.py`.

---

**For questions, see code comments or contact the author.**
