from qdrant_client import QdrantClient

def get_qdrant_sources():
    # Assuming you are running this locally where your KAE instance is hosted
    client = QdrantClient("http://localhost:6333")
    
    # Updated to match the collection name in your dashboard
    collection_name = "mb_sources" 
    
    # Grabbing the 'id' field which holds values like 'geft'
    payload_key = "id" 

    sources = set()
    offset = None

    print(f"Connecting to Qdrant and scanning '{collection_name}'... hold tight!")

    try:
        while True:
            # Paginating through the collection
            records, next_offset = client.scroll(
                collection_name=collection_name,
                scroll_filter=None,
                limit=1000,
                with_payload=[payload_key], 
                with_vectors=False, # We only need the payload metadata
                offset=offset
            )

            for record in records:
                if record.payload and payload_key in record.payload:
                    sources.add(record.payload[payload_key])

            if next_offset is None:
                break 
            
            offset = next_offset

        if not sources:
            print("Didn't find any sources. Make sure the database is populated!")
            return

        print(f"\nBrilliant. Found {len(sources)} distinct sources directly from Qdrant:\n")
        for source in sorted(sources):
            print(f"- {source}")

    except Exception as e:
        print(f"Something went a bit pear-shaped with the Qdrant connection: {e}")

if __name__ == "__main__":
    get_qdrant_sources()
