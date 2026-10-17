import sqlite3
import os
import chromadb
import re
from chromadb.utils import embedding_functions

class DatabaseManager:
    def __init__(self, db_path="database/events.db", chroma_path="database/chroma_store"):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self._init_sqlite()

        # ChromaDB for Semantic Search
        self.chroma_client = chromadb.PersistentClient(path=chroma_path)
        self.emb_fn = embedding_functions.DefaultEmbeddingFunction()
        self.collection = self.chroma_client.get_or_create_collection(
            name="incident_summaries",
            embedding_function=self.emb_fn
        )

    def _init_sqlite(self):
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS security_events (
                event_id TEXT PRIMARY KEY,
                camera_id TEXT,
                timestamp TEXT,
                category TEXT,
                confidence REAL,
                summary TEXT,
                keyframe_path TEXT
            )
        ''')
        self.conn.commit()

    def insert_event(self, event_data):
        # 1. Save hard facts to SQLite
        self.cursor.execute('''
            INSERT OR REPLACE INTO security_events 
            (event_id, camera_id, timestamp, category, confidence, summary, keyframe_path)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            event_data["event_id"], event_data["camera_id"], event_data["timestamp"],
            event_data["category"], event_data["confidence"], event_data["summary"],
            event_data["keyframe_path"]
        ))
        self.conn.commit()

        # 2. Save semantic meaning to ChromaDB
        self.collection.add(
            documents=[event_data["summary"]],
            metadatas=[{
                "event_id": event_data["event_id"],
                "camera_id": event_data["camera_id"],
                "timestamp": event_data["timestamp"],
                "category": event_data["category"],
                "keyframe_path": event_data["keyframe_path"]
            }],
            ids=[event_data["event_id"]]
        )

    def query_sql(self, query, params=()):
        self.cursor.execute(query, params)
        return self.cursor.fetchall()

    def search_semantic(self, query_text, n_results=3):
        # PHASE 5: The LangChain-style Agentic Router
        # Acts as a traffic cop to extract hard filters before running the vector search
        
        # 1. Extract Camera ID (e.g., "Camera 1", "CAM_01")
        camera_match = re.search(r'(?i)camera\s*(\d+)', query_text)
        where_filter = None
        
        if camera_match:
            cam_num = camera_match.group(1).zfill(2)
            target_camera = f"CAM_{cam_num}"
            where_filter = {"camera_id": target_camera}
            print(f"[Agent Router] Isolated hardware filter: {target_camera}")

        # 2. Extract Category Keywords
        if "fight" in query_text.lower() or "altercation" in query_text.lower():
            if where_filter:
                where_filter = {"$and": [where_filter, {"category": "Fighting / Altercation"}]}
            else:
                where_filter = {"category": "Fighting / Altercation"}
                
        elif "fall" in query_text.lower() or "collapsed" in query_text.lower() or "down" in query_text.lower():
            if where_filter:
                where_filter = {"$and": [where_filter, {"category": "Person Down"}]}
            else:
                where_filter = {"category": "Person Down"}

        # 3. Route to ChromaDB with the extracted SQL-style filters
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=where_filter
        )
        return results