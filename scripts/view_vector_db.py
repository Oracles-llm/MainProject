"""
Script to view data points in the Qdrant vector database.
Displays collection info, point IDs, payloads, and vector information.
"""

import sys
from pathlib import Path
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.db.qdrant_client import QdrantDB
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def view_collection_info(qdrant: QdrantDB, collection_name: str):
    """Display collection information."""
    print("=" * 80)
    print("COLLECTION INFORMATION")
    print("=" * 80)
    
    info = qdrant.get_collection_info(collection_name)
    if info:
        print(f"Collection Name: {info['name']}")
        print(f"Vector Size: {info['vector_size']}")
        print(f"Distance Metric: {info['distance']}")
        print(f"Total Points: {info['points_count']}")
        print(f"Indexed Vectors: {info['indexed_vectors_count']}")
        print(f"Status: {info['status']}")
    else:
        print(f"Failed to retrieve collection info for '{collection_name}'")
    
    print()


def view_all_points(qdrant: QdrantDB, collection_name: str, limit: int = None):
    """Retrieve and display all points in the collection."""
    print("=" * 80)
    print("DATA POINTS IN COLLECTION")
    print("=" * 80)
    print()
    
    try:
        scroll_result = qdrant.client.scroll(
            collection_name=collection_name,
            limit=limit or 100,
            with_payload=True,
            with_vectors=True
        )
        
        points = scroll_result[0]
        next_page_offset = scroll_result[1]
        
        print(f"Retrieved {len(points)} point(s)")
        if next_page_offset:
            print(f"Note: More points available (next offset: {next_page_offset})")
        print()
        
        for i, point in enumerate(points, 1):
            print(f"{'─' * 80}")
            print(f"Point #{i}")
            print(f"{'─' * 80}")
            print(f"ID: {point.id}")
            print()
            
            print("Payload:")
            if point.payload:
                for key, value in point.payload.items():
                    if key == "text" and isinstance(value, str) and len(value) > 100:
                        print(f"  {key}: {value[:100]}... (truncated, {len(value)} chars total)")
                    else:
                        print(f"  {key}: {value}")
            else:
                print("  (empty)")
            print()
            
            print("Vectors:")
            if hasattr(point, 'vector'):
                if isinstance(point.vector, dict):
                    print("  Named vectors:")
                    for vec_name, vec_data in point.vector.items():
                        if vec_name == "":
                            vec_name = "(default dense)"
                        if isinstance(vec_data, list):
                            print(f"    {vec_name}: Dense vector [{len(vec_data)} dimensions]")
                            print(f"      First 5 values: {vec_data[:5]}")
                            print(f"      Last 5 values: {vec_data[-5:]}")
                        else:
                            print(f"    {vec_name}: {type(vec_data).__name__}")
                            if hasattr(vec_data, 'indices') and hasattr(vec_data, 'values'):
                                print(f"      Sparse vector with {len(vec_data.indices)} non-zero values")
                                print(f"      First 5 indices: {vec_data.indices[:5]}")
                                print(f"      First 5 values: {vec_data.values[:5]}")
                elif isinstance(point.vector, list):
                    print(f"  Dense vector: [{len(point.vector)} dimensions]")
                    print(f"    First 5 values: {point.vector[:5]}")
                    print(f"    Last 5 values: {point.vector[-5:]}")
                else:
                    print(f"  Vector type: {type(point.vector)}")
            else:
                print("  (no vector data)")
            print()
        
        return len(points)
        
    except Exception as e:
        logger.error(f"Failed to retrieve points: {e}")
        import traceback
        traceback.print_exc()
        return 0


def view_point_by_id(qdrant: QdrantDB, collection_name: str, point_id: str):
    """View a specific point by ID."""
    print("=" * 80)
    print(f"POINT DETAILS: {point_id}")
    print("=" * 80)
    print()
    
    try:
        result = qdrant.client.retrieve(
            collection_name=collection_name,
            ids=[point_id],
            with_payload=True,
            with_vectors=True
        )
        
        if result:
            point = result[0]
            print(f"ID: {point.id}")
            print()
            
            print("Payload:")
            print(json.dumps(point.payload, indent=2, ensure_ascii=False))
            print()
            
            print("Vectors:")
            if hasattr(point, 'vector'):
                if isinstance(point.vector, dict):
                    for vec_name, vec_data in point.vector.items():
                        if vec_name == "":
                            vec_name = "(default dense)"
                        print(f"\n{vec_name}:")
                        if isinstance(vec_data, list):
                            print(f"  Type: Dense vector")
                            print(f"  Dimensions: {len(vec_data)}")
                            print(f"  Sample values: {vec_data[:10]}")
                        else:
                            print(f"  Type: {type(vec_data).__name__}")
                            if hasattr(vec_data, 'indices') and hasattr(vec_data, 'values'):
                                print(f"  Non-zero values: {len(vec_data.indices)}")
                                print(f"  Sample indices: {vec_data.indices[:10]}")
                                print(f"  Sample values: {vec_data.values[:10]}")
                else:
                    print(f"  Dense vector: {len(point.vector)} dimensions")
                    print(f"  Sample: {point.vector[:10]}")
            else:
                print("  (no vector data)")
        else:
            print(f"Point with ID '{point_id}' not found")
            
    except Exception as e:
        logger.error(f"Failed to retrieve point: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main function to view vector database contents."""
    import argparse
    
    parser = argparse.ArgumentParser(description="View data points in Qdrant vector database")
    parser.add_argument(
        "--collection",
        type=str,
        default=settings.QDRANT_COLLECTION_NAME,
        help="Collection name (default: from settings)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of points to display (default: all)"
    )
    parser.add_argument(
        "--point-id",
        type=str,
        default=None,
        help="View specific point by ID"
    )
    parser.add_argument(
        "--local-path",
        type=str,
        default="./data/qdrant",
        help="Local Qdrant storage path (default: ./data/qdrant)"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("VECTOR DATABASE VIEWER")
    print("=" * 80)
    print()
    
    try:
        with QdrantDB(mode="local", local_path=args.local_path) as qdrant:
            collection_name = args.collection
            
            if not qdrant.collection_exists(collection_name):
                print(f"Error: Collection '{collection_name}' does not exist!")
                return
            
            view_collection_info(qdrant, collection_name)
            
            if args.point_id:
                view_point_by_id(qdrant, collection_name, args.point_id)
            else:
                view_all_points(qdrant, collection_name, limit=args.limit)
            
            print("=" * 80)
            print("VIEW COMPLETE")
            print("=" * 80)
            
    except Exception as e:
        logger.error(f"Failed to view vector database: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

