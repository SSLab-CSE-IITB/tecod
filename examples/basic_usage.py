#!/usr/bin/env python3
"""
Basic TeCoD Usage Example

This script demonstrates the simplest way to use TeCoD programmatically
for text-to-SQL generation.
"""

import os
import sys

# Add the parent directory to the path to import TeCoD
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.api import TeCoD

FINANCIAL_CONFIG = ["+data=financial"]


def main():
    """Basic usage example."""
    print("🔧 TeCoD Basic Usage Example")
    print("=" * 50)

    # Initialize TeCoD with default settings
    print("Initializing TeCoD...")
    tecod = TeCoD(
        config_overrides=FINANCIAL_CONFIG,
        device="auto",  # Auto-detect CUDA/CPU
        console_log_level="INFO",  # Show info messages
    )

    # Check if ready
    if not tecod.is_ready:
        print("❌ TeCoD initialization failed!")
        return

    print("✅ TeCoD initialized successfully!")
    print()

    queries = [
        "What is the count of accounts opting for post-transaction issuance that are located in north Moravia?",
        "How many accounts from north Bohemia are eligible to receive loans?",
        "Count the accounts in east Bohemia which are permitted to be offered loans.",
        "Could you compute the number of loan-eligible accounts in west Bohemia?",
        "How many accounts registered in Prague qualify for loan offers?",
    ]

    print("🔍 Generating SQL for example queries:")
    print("-" * 50)

    for i, query in enumerate(queries, 1):
        print(f"\n{i}. Query: {query}")

        try:
            # Generate SQL
            result = tecod.generate(query)

            # Display results
            print(f"   Method: {result.method.upper()}")
            print(f"   SQL: {result.pred_sql}")
            print(f"   Time: {result.total_time * 1000:.1f}ms")
            print(f"   Template ID: {result.template_id}")
            print(f"   NLI Score: {result.nli_score:.3f}")

        except Exception as e:
            print(f"   ❌ Error: {str(e)}")

    print("\n" + "=" * 50)
    print("✅ Basic example completed!")

    # Cleanup resources
    tecod.cleanup()


if __name__ == "__main__":
    main()
