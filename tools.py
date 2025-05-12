import math
import re
import requests

def calculator(query: str) -> str:
    # Extract and evaluate math expression
    expr = re.findall(r"[\d\s\+\-\*/\(\)\.]+", query)
    if expr:
        try:
            result = eval(expr[0], {"__builtins__": {}})
            return f"Result: {result}"
        except Exception as e:
            return f"Error: {e}"
    return "No valid expression found."

def dictionary(query: str) -> str:
    # Extract word after 'define' or 'what is'
    match = re.search(r"define ([a-zA-Z]+)|what is ([a-zA-Z]+)", query, re.IGNORECASE)
    word = match.group(1) or match.group(2) if match else None
    if not word:
        return "No word to define."
    # Simple static definitions (extend as needed)
    definitions = {
        'faiss': 'FAISS is a library for efficient similarity search and clustering of dense vectors.',
        'rag': 'RAG stands for Retrieval-Augmented Generation, a method combining retrieval and generation models.',
        'widget': 'A widget is a small gadget or mechanical device.'
    }
    return definitions.get(word.lower(), f"No definition found for '{word}'.")
