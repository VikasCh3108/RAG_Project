from rag_agent import RAGAgent
from tools import calculator, dictionary

class AgentOrchestrator:
    def __init__(self, rag_agent: RAGAgent):
        self.rag_agent = rag_agent
        self.log = []

    def route(self, query: str) -> str:
        decision = None
        answer = None
        if any(kw in query.lower() for kw in ["calculate", "+", "-", "*", "/"]):
            decision = "calculator"
            answer = calculator(query)
        elif "define" in query.lower() or "what is" in query.lower():
            decision = "dictionary"
            answer = dictionary(query)
        else:
            decision = "rag"
            answer = self.rag_agent.generate_answer(query)
        self.log.append({"query": query, "decision": decision, "answer": answer})
        return answer

    def get_log(self):
        return self.log
