#!/usr/bin/env python3
"""
OpenAI-compatible API Usage Example

This script demonstrates how to use TeCoD with any OpenAI-compatible endpoint
as the LLM provider instead of a local HuggingFace model.

Supported providers: OpenAI, vLLM, Ollama, Together, Groq, LM Studio, etc.

Configuration:
  export OPENAI_API_KEY=your-api-key  # required for OpenAI-hosted endpoints
  export OPENAI_BASE_URL=https://api.openai.com/v1  # optional, defaults to OpenAI

  # For local providers (vLLM, Ollama, etc.):
  # export OPENAI_BASE_URL=http://localhost:11434/v1
  # OPENAI_API_KEY can be empty for unauthenticated local endpoints.

Note: API models only support SGC, ICL, and ZS methods.
      GCD and Base-GCD require direct model logit access and are not available.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.api import TeCoD

FINANCIAL_OPENAI_CONFIG = [
    "env@_global_=openai",
    "+data=financial",
    "tecod.model_id=gpt-4o",
    "tecod.temperature=0.0",
]


def main():
    """OpenAI-compatible API usage example."""
    print("TeCoD + OpenAI-compatible API Example")
    print("=" * 50)

    # --- Configuration ---
    # The OpenAI env profile changes only the generation backend.
    # Everything else (NLI, embeddings, vector store, sample data) stays local.
    print("Initializing TeCoD with OpenAI-compatible provider...")
    tecod = TeCoD(
        config_overrides=FINANCIAL_OPENAI_CONFIG,
        console_log_level="INFO",
    )

    if not tecod.is_ready:
        print("TeCoD initialization failed!")
        return

    print("TeCoD initialized with OpenAI-compatible backend")
    print()

    # --- Generate SQL ---
    queries = [
        "What is the count of accounts opting for post-transaction issuance that are located in north Moravia?",
        "How many accounts from north Bohemia are eligible to receive loans?",
        "Could you compute the number of loan-eligible accounts in west Bohemia?",
    ]

    print("Generating SQL for example queries:")
    print("-" * 50)

    for i, query in enumerate(queries, 1):
        print(f"\n{i}. Query: {query}")

        try:
            result = tecod.generate(query)

            print(f"   Method: {result.method.upper()}")
            print(f"   SQL: {result.pred_sql}")
            print(f"   Time: {result.total_time * 1000:.1f}ms")
            print(f"   Template ID: {result.template_id}")
            print(f"   NLI Label: {result.nli_label}")

        except Exception as e:
            print(f"   Error: {str(e)}")

    print("\n" + "=" * 50)
    print("Example completed!")

    tecod.cleanup()


if __name__ == "__main__":
    main()
