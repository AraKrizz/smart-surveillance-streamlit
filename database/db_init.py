import sqlite3
import chromadb
import os

DB_PATH = os.path.join("database", "events.db")
CHROMA_PATH = os.path.join("database", "chroma_store")

def initialize_databases():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS security_events (
            event_id TEXT PRIMARY KEY,
            camera_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            category TEXT NOT NULL,
            confidence REAL NOT NULL,
            summary TEXT NOT NULL,
            keyframe_path TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()
    print("[+] SQLite Database initialized.")

    os.makedirs(CHROMA_PATH, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    chroma_client.get_or_create_collection(name="incident_summaries")
    print("[+] ChromaDB Vector Store initialized.")

if __name__ == "__main__":
    initialize_databases()
