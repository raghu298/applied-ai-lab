"""
CLI interface for querying the RAG system.

Usage:
    python query.py
    python query.py "What is gradient descent?"
"""

import sys
import os

import rag


def print_result(result: dict):
    print("\n--- Answer ---")
    print(result["answer"])
    if result["sources"]:
        print("\n--- Sources ---")
        for i, src in enumerate(result["sources"], 1):
            print(f"[{i}] {src['file']} | Page {src['page']}")
            print(f"    \"{src['snippet']}...\"")
    print()


def main():
    if not os.path.exists("chroma_db"):
        print("Vector store not found. Run 'python ingest.py' first.")
        sys.exit(1)

    print("Loading RAG chain...")
    vector_store = rag.load_vector_store()
    chain = rag.build_rag_chain(vector_store)
    print("Ready. Type your question (or 'quit' to exit).\n")

    # Accept a one-shot question from command line
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        print(f"Q: {question}")
        result = rag.ask(chain, question)
        print_result(result)
        return

    # Interactive loop
    while True:
        try:
            question = input("Q: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break
        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye.")
            break
        result = rag.ask(chain, question)
        print_result(result)


if __name__ == "__main__":
    main()
