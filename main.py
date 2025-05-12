import os
from rag_agent import RAGAgent
from agent_orchestrator import AgentOrchestrator
from dotenv import load_dotenv

# Load environment variables from .env file automatically
load_dotenv()

def main():
    print("Welcome to the RAG Multi-Agent Q&A Assistant!")
    openai_key = os.getenv("OPENAI_API_KEY")
    rag_agent = RAGAgent(data_dir="data", openai_api_key=openai_key)
    orchestrator = AgentOrchestrator(rag_agent)
    while True:
        query = input("\nAsk a question (or type 'exit'): ")
        if query.strip().lower() == 'exit':
            break
        answer = orchestrator.route(query)
        print(f"\nAnswer: {answer}")
        # Print the last log entry for transparency
        last_log = orchestrator.get_log()[-1]
        print(f"[LOG] Query: {last_log['query']}")
        print(f"[LOG] Decision: {last_log['decision']}")
        print(f"[LOG] Answer: {last_log['answer']}")
        # If the RAG pipeline was used, show the retrieved context
        if last_log['decision'] == 'rag':
            context = rag_agent.retrieve(query)
            print("[CONTEXT SNIPPETS]")
            for idx, (snippet, source) in enumerate(context, 1):
                print(f"  [{idx}] ({source}): {snippet}")
    print("Goodbye!")

if __name__ == "__main__":
    main()
