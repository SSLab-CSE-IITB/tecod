#!/usr/bin/env python3
"""
TeCoD Integration Example

This script shows how to integrate TeCoD into a larger application,
such as a web service or data analysis pipeline.
"""

import json
import os
import sys
from typing import Any

# Add the parent directory to the path to import TeCoD
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.api import TeCoD

FINANCIAL_CONFIG = ["+data=financial"]


class SQLGenerator:
    """
    A wrapper class that integrates TeCoD for SQL generation.

    This demonstrates how you might integrate TeCoD into a larger application
    that needs to generate SQL from natural language queries.
    """

    def __init__(self):
        """Initialize the SQL generator."""
        self.tecod = TeCoD(
            config_overrides=FINANCIAL_CONFIG,
            console_log_level="WARNING",  # Minimal logging
            file_log_level="DEBUG",
            log_file="sql_generator.log",
        )

    def generate_sql(self, query: str, include_metadata: bool = False) -> dict[str, Any]:
        """
        Generate SQL with optional metadata.

        Args:
            query: Natural language query
            include_metadata: Whether to include timing and method info

        Returns:
            Dictionary with SQL and optional metadata
        """
        try:
            result = self.tecod.generate(query)

            response = {"success": True, "sql": result.pred_sql, "query": query}

            if include_metadata:
                response.update(
                    {
                        "method": result.method,
                        "generation_time_ms": result.total_time * 1000
                        if result.total_time
                        else None,
                        "template_id": result.template_id,
                        "nli_score": result.nli_score,
                        "confidence": "high"
                        if result.nli_score > 0.8
                        else "medium"
                        if result.nli_score > 0.5
                        else "low",
                    }
                )

            return response

        except Exception as e:
            return {"success": False, "error": str(e), "query": query, "sql": None}

    def batch_process_file(self, input_file: str, output_file: str) -> None:
        """
        Process a file of queries and save results.

        Args:
            input_file: Path to file containing queries (one per line)
            output_file: Path to save JSON results
        """
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")

        # Read queries
        with open(input_file) as f:
            queries = [line.strip() for line in f if line.strip()]

        print(f"Processing {len(queries)} queries from {input_file}")

        # Generate SQL for each query
        results = []
        for i, query in enumerate(queries, 1):
            print(f"Processing {i}/{len(queries)}: {query[:50]}{'...' if len(query) > 50 else ''}")
            result = self.generate_sql(query, include_metadata=True)
            results.append(result)

        # Save results
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"Results saved to {output_file}")

        # Print summary
        successful = sum(1 for r in results if r["success"])
        print(f"\nSummary: {successful}/{len(queries)} successful generations")

    def close(self):
        """Clean up resources."""
        self.tecod.cleanup()


def demo_api_simulation():
    """Simulate using TeCoD in an API endpoint."""
    print("🌐 API Simulation Demo")
    print("-" * 30)

    generator = SQLGenerator()

    api_requests = [
        {
            "query": "What is the count of accounts opting for post-transaction issuance that are located in north Moravia?",
            "include_meta": True,
        },
        {
            "query": "How many accounts from north Bohemia are eligible to receive loans?",
            "include_meta": False,
        },
        {
            "query": "Could you compute the number of loan-eligible accounts in west Bohemia?",
            "include_meta": True,
        },
        {"query": "Invalid gibberish query xyz", "include_meta": True},
    ]

    for i, request in enumerate(api_requests, 1):
        print(f"\n📝 API Request {i}:")
        print(f"   Query: {request['query']}")

        response = generator.generate_sql(
            query=request["query"], include_metadata=request["include_meta"]
        )

        if response["success"]:
            print(f"   ✅ SQL: {response['sql']}")
            if "method" in response:
                print(f"   📊 Method: {response['method'].upper()}")
                print(f"   ⏱️ Time: {response['generation_time_ms']:.1f}ms")
                print(f"   🎯 Confidence: {response['confidence']}")
        else:
            print(f"   ❌ Error: {response['error']}")

    generator.close()


def demo_batch_file_processing():
    """Demonstrate batch file processing."""
    print("\n📁 Batch File Processing Demo")
    print("-" * 30)

    sample_queries = [
        "How many accounts from north Bohemia are eligible to receive loans?",
        "Count the accounts in east Bohemia which are permitted to be offered loans.",
        "Find the number of accounts situated in central Bohemia that can get loans.",
        "Could you tell me the count of accounts in west Bohemia that are loan-eligible?",
        "How many accounts registered in Prague qualify for loan offers?",
    ]

    input_file = "sample_queries.txt"
    output_file = "generated_sql.json"

    # Write sample queries
    with open(input_file, "w") as f:
        for query in sample_queries:
            f.write(query + "\n")

    print(f"Created {input_file} with {len(sample_queries)} sample queries")

    # Process the file
    generator = SQLGenerator()

    try:
        generator.batch_process_file(input_file, output_file)

        # Show results
        with open(output_file) as f:
            results = json.load(f)

        print("\n📋 Generated Results:")
        for result in results:
            status = "✅" if result["success"] else "❌"
            print(f"   {status} {result['query']}")
            if result["success"]:
                print(f"      SQL: {result['sql']}")
            else:
                print(f"      Error: {result['error']}")

    finally:
        generator.close()

        # Clean up
        if os.path.exists(input_file):
            os.remove(input_file)
        if os.path.exists(output_file):
            os.remove(output_file)


def main():
    """Run integration examples."""
    print("🔌 TeCoD Integration Examples")
    print("=" * 50)

    try:
        demo_api_simulation()
        demo_batch_file_processing()

        print("\n" + "=" * 50)
        print("✅ Integration examples completed!")

    except Exception as e:
        print(f"\n❌ Error running integration examples: {str(e)}")


if __name__ == "__main__":
    main()
