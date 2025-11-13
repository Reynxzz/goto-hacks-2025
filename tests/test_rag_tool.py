"""Test script for RAG Milvus Tool"""
import sys
from pathlib import Path
import json

# Add src directory to Python path
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

from src.tools import RAGMilvusTool
from src.config.settings import get_settings


def test_rag_tool():
    """Test RAG Milvus tool functionality."""
    print("=" * 60)
    print("RAG Milvus Tool Test - Combined Collection Search")
    print("=" * 60)

    try:
        settings = get_settings()

        # Initialize the tool
        print("\nInitializing RAG Milvus tool...")
        tool = RAGMilvusTool()

        if not tool.is_available():
            print("\n❌ RAG Milvus tool is not available")
            print("Make sure the database file exists at:", settings.rag.db_path)
            return 1

        print("✅ RAG Milvus tool initialized successfully")
        print(f"Database: {settings.rag.db_path}")
        print(f"Model: {settings.rag.embedding_model}")
        print(f"Top K results: {settings.rag.top_k}")
        print("\n🔍 Search Strategy: Single 'combined_item' collection with 'source' field")
        print("   Results include text and source information")

        # Test queries
        test_queries = [
            "What is user income data?",
            "Tell me about DGE",
            "How do push notifications work?",
            "Information about Genie",
            "ride model pipeline"
        ]

        for i, query in enumerate(test_queries, 1):
            print("\n" + "=" * 60)
            print(f"TEST {i}: {query}")
            print("=" * 60)

            result = tool._run(query)
            result_data = json.loads(result)

            # Check for errors
            if "error" in result_data:
                print(f"\n❌ Error: {result_data['error']}")
                continue

            print("\n" + "-" * 60)
            print("Search Summary:")
            print("-" * 60)
            print(f"Query: {result_data.get('query', 'N/A')}")
            print(f"Collection: {result_data.get('collection', 'N/A')}")
            print(f"Results Found: {result_data.get('results_count', 0)}")
            print(f"Sources Found: {len(result_data.get('sources_found', []))}")
            if result_data.get('sources_found'):
                print(f"Source List: {', '.join(result_data.get('sources_found', []))}")

            # Display top results with source information
            if result_data.get("results_count", 0) > 0:
                print("\n📌 Top Results:")
                print("-" * 60)

                # Group results by source for visualization
                results_by_source = {}
                for result in result_data.get("results", []):
                    source = result.get("source", "N/A")
                    if source not in results_by_source:
                        results_by_source[source] = []
                    results_by_source[source].append(result)

                # Display top 3 overall results
                for idx, result in enumerate(result_data.get("results", [])[:3], 1):
                    print(f"\n{idx}. Source: {result['source']}")
                    print(f"   Score: {result['score']:.4f}")
                    print(f"   Text: {result['text'][:150]}...")

                # Show source distribution
                print(f"\n📊 Results Distribution by Source:")
                for source, results in results_by_source.items():
                    print(f"   - {source}: {len(results)} result(s)")
            else:
                print("\n⚠️ No results found")

        print("\n" + "=" * 60)
        print("✅ All tests completed!")
        print("=" * 60)
        print("\n📋 Test Summary:")
        print(f"   - Total queries tested: {len(test_queries)}")
        print(f"   - Database schema: Single 'combined_item' collection")
        print(f"   - Model: GoToCompany/embeddinggemma-300m-gotoai-v1")
        print(f"   - Output fields: text + source")
        print("\n💡 Usage with documentation agent:")
        print("   python scripts/run_documentation_agent.py namespace/project --with-rag")
        print("\n🔧 Schema Features:")
        print("   ✓ Single combined collection (not separate collections)")
        print("   ✓ Source field indicates data origin")
        print("   ✓ Simpler and more efficient search")
        print("   ✓ No fuzzy logic needed")
        print("   ✓ Results include both text and source information")

        return 0

    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        return 130

    except Exception as e:
        print(f"\n❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(test_rag_tool())
