#!/usr/bin/env python3
"""
Advanced TeCoD Usage Example

This script demonstrates advanced features like multiple-query processing,
context management, and detailed timing analysis.
"""

import os
import sys
import time

# Add the parent directory to the path to import TeCoD
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.api import TeCoD, create_tecod

FINANCIAL_CONFIG = ["+data=financial"]


def single_query_analysis():
    """Demonstrate detailed analysis of a single query."""
    print("\n🔍 Single Query Analysis")
    print("-" * 30)

    # Use context manager for automatic cleanup
    with TeCoD(
        config_overrides=FINANCIAL_CONFIG,
        console_log_level="ERROR",
    ) as tecod:
        query = (
            "What is the count of accounts opting for post-transaction issuance that are "
            "located in north Moravia?"
        )

        print(f"Query: {query}")
        result = tecod.generate(query)

        # Detailed timing analysis
        print("\n📊 Generation Results:")
        print(f"   Generated SQL: {result.pred_sql}")
        print(f"   Method Used: {result.method.upper()}")
        print(f"   Template ID: {result.template_id}")
        print(f"   NLI Label: {result.nli_label}")

        print("\n⏱️ Timing Breakdown:")
        print(f"   Total Time: {result.total_time * 1000:.1f}ms")

        if result.timing_data:
            for operation, duration in result.timing_data.items():
                if not operation.endswith("_count") and "count" not in operation:
                    print(f"   {operation}: {duration * 1000:.1f}ms")

        print("\n📈 Data Flow:")
        print(f"   Retrieved Examples: {result.retrieved_examples_count}")
        print(f"   NLI Processed: {result.nli_examples_count}")
        print(f"   ICL Examples Used: {len(result.icl_example_indices)}")


def multiple_query_demo():
    """Demonstrate processing multiple queries with one TeCoD instance."""
    print("\n📦 Multiple Query Demo")
    print("-" * 30)

    queries = [
        "How many accounts from north Bohemia are eligible to receive loans?",
        "Count the accounts in east Bohemia which are permitted to be offered loans.",
        "Find the number of accounts situated in central Bohemia that can get loans.",
        "Could you tell me the count of accounts in west Bohemia that are loan-eligible?",
        "Give me the total accounts from south Moravia that meet the eligibility for loans.",
        "What is the tally of accounts based in south Bohemia that can avail themselves of loans?",
        "How many accounts registered in Prague qualify for loan offers?",
        "Determine the number of accounts originating from north Bohemia which are eligible for financing.",
    ]

    tecod = create_tecod(
        config_overrides=FINANCIAL_CONFIG,
        console_log_level="WARNING",
    )

    print(f"Processing {len(queries)} queries with one TeCoD instance...")

    run_start = time.perf_counter()
    results = [tecod.generate(query, max_new_tokens=256) for query in queries]
    run_time = time.perf_counter() - run_start

    print(f"\n✅ Queries completed in {run_time * 1000:.1f}ms")
    print(f"Average per query: {(run_time / len(queries)) * 1000:.1f}ms")

    # Analyze results
    methods_used = {}
    total_generation_time = 0

    for i, result in enumerate(results, 1):
        if result.pred_sql:  # Success
            method = result.method
            methods_used[method] = methods_used.get(method, 0) + 1
            total_generation_time += result.total_time or 0

            print(f"\n{i}. {queries[i - 1]}")
            print(f"   SQL: {result.pred_sql}")
            print(f"   Method: {method.upper()}, Time: {(result.total_time or 0) * 1000:.1f}ms")
        else:
            print(f"\n{i}. {queries[i - 1]} - ❌ FAILED")

    print("\n📊 Query Statistics:")
    print(f"   Total Generation Time: {total_generation_time * 1000:.1f}ms")
    print(f"   Methods Used: {dict(methods_used)}")

    tecod.cleanup()


def performance_comparison():
    """Compare performance across different parameters."""
    print("\n🏃 Performance Comparison")
    print("-" * 30)

    query = "Could you compute the number of loan-eligible accounts in west Bohemia?"

    # Test different configurations
    configs = [
        {"top_k": 100, "num_beams": 1, "name": "Fast (top_k=100, greedy)"},
        {"top_k": 500, "num_beams": 1, "name": "Medium (top_k=500, greedy)"},
        {"top_k": 1000, "num_beams": 1, "name": "Thorough (top_k=1000, greedy)"},
        {"top_k": 1000, "num_beams": 4, "name": "Best Quality (top_k=1000, beam=4)"},
    ]

    with TeCoD(
        config_overrides=FINANCIAL_CONFIG,
        console_log_level="ERROR",
    ) as tecod:
        print(f"Query: {query}\n")

        for config in configs:
            name = config.pop("name")

            try:
                result = tecod.generate(query, **config)

                print(f"{name}:")
                print(f"   Time: {result.total_time * 1000:.1f}ms")
                print(f"   Method: {result.method.upper()}")
                print(f"   SQL: {result.pred_sql}")
                print(f"   Template Score: {result.nli_score:.3f}")
                print()

            except Exception as e:
                print(f"{name}: ❌ Error - {str(e)}\n")


def error_handling_demo():
    """Demonstrate error handling and edge cases."""
    print("\n⚠️ Error Handling Demo")
    print("-" * 30)

    with TeCoD(
        config_overrides=FINANCIAL_CONFIG,
        console_log_level="ERROR",
    ) as tecod:
        # Test various edge cases
        test_cases = [
            ("", "Empty query"),
            ("   ", "Whitespace only"),
            ("This is not a database query about cooking recipes", "Unrelated query"),
            ("SELECT * FROM users WHERE age > ?", "Query with parameters"),
            ("A" * 1000, "Very long query"),
        ]

        for query, description in test_cases:
            print(f"Testing: {description}")
            print(f"Query: '{query[:50]}{'...' if len(query) > 50 else ''}'")

            try:
                result = tecod.generate(query)
                print(f"✅ Success: {result.pred_sql}")
                print(f"   Method: {result.method}, Time: {result.total_time * 1000:.1f}ms")
            except Exception as e:
                print(f"❌ Expected error: {str(e)}")

            print()


def main():
    """Run all advanced examples."""
    print("🚀 TeCoD Advanced Usage Examples")
    print("=" * 50)

    try:
        single_query_analysis()
        multiple_query_demo()
        performance_comparison()
        error_handling_demo()

        print("\n" + "=" * 50)
        print("✅ All advanced examples completed!")

    except Exception as e:
        print(f"\n❌ Error running examples: {str(e)}")


if __name__ == "__main__":
    main()
