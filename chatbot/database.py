# database.py
import config as c
import typing as t
import sqlite3 as q
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings

class Database:
    def __init__(self):
        self.db_path = c.SQLITE_DB_PATH
        self.vector_db = None
        self._initialize_sqlite()
        self._initialize_vector_db()

    def _initialize_sqlite(self):
        """Initialize SQLite database with required tables."""
        conn = q.connect(str(self.db_path))
        cursor = conn.cursor()

        # Demographics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS demographics (
                user_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                age INTEGER,
                gender TEXT,
                ethnicity TEXT,
                hometown TEXT,
                education TEXT,
                email TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Chats table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chats (
                chat_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                role TEXT NOT NULL,
                message TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES demographics(user_id)
            )
        """)

        # User context table (for LLM memory)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_context (
                user_id TEXT PRIMARY KEY,
                context_summary TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES demographics(user_id)
            )
        """)

        conn.commit()
        conn.close()

    def _initialize_vector_db(self):
        """Initialize vector database with sample university data."""
        try:
            embeddings = OllamaEmbeddings(model=c.EMBEDDING_MODEL, base_url=c.OLLAMA_BASE_URL)
            # Create documents from sample data
            documents = [item["content"] for item in c.SAMPLE_UNIVERSITY_DATA]
            metadatas = [{"title": item["title"], "page": item["page"]} for item in c.SAMPLE_UNIVERSITY_DATA]
            self.vector_db = Chroma.from_texts(texts=documents, embedding=embeddings, metadatas=metadatas, persist_directory=str(c.VECTOR_DB_PATH))
            print("Vector database initialized successfully.")
        except Exception as e:
            print(f"Warning: Could not initialize vector DB: {e}")
            print("Make sure Ollama is running and models are available.")
            self.vector_db = None

    def populate_sample_data(self):
        """Populate sample users into demographics table."""
        conn = q.connect(str(self.db_path))
        cursor = conn.cursor()

        for user_id, user_data in c.SAMPLE_USERS.items():
            cursor.execute("""
                INSERT OR IGNORE INTO demographics 
                (user_id, name, age, gender, ethnicity, hometown, education, email)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                user_id,
                user_data["name"],
                user_data["age"],
                user_data["gender"],
                user_data["ethnicity"],
                user_data["hometown"],
                user_data["education"],
                user_data["email"]
            ))

        conn.commit()
        conn.close()

    def get_user_demographics(self, user_id: str) -> t.Optional[t.Dict[str, t.Any]]:
        """Retrieve user demographic information."""
        conn = q.connect(str(self.db_path))
        conn.row_factory = q.Row
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM demographics WHERE user_id = ?", (user_id,))
        row = cursor.fetchone()
        conn.close()

        if row:
            return dict(row)
        return None

    def add_chat_message(self, user_id: str, role: str, message: str) -> bool:
        """Add a chat message to the database."""
        conn = q.connect(str(self.db_path))
        cursor = conn.cursor()

        try:
            cursor.execute("INSERT INTO chats (user_id, role, message) VALUES (?, ?, ?)", (user_id, role, message))
            conn.commit()
            return True
        except Exception as e:
            print(f"Error adding chat message: {e}")
            return False
        finally:
            conn.close()

    def get_chat_history(self, user_id: str, limit: int = 20) -> t.List[t.Dict[str, t.Any]]:
        """Retrieve chat history for a user."""
        conn = q.connect(str(self.db_path))
        conn.row_factory = q.Row
        cursor = conn.cursor()
        cursor.execute("""
            SELECT role, message, timestamp FROM chats 
            WHERE user_id = ? 
            ORDER BY timestamp DESC 
            LIMIT ?
        """, (user_id, limit))
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in reversed(rows)]

    def update_user_context(self, user_id: str, context_summary: str) -> bool:
        """Update or create user context summary."""
        conn = q.connect(str(self.db_path))
        cursor = conn.cursor()
        try:
            cursor.execute("""
                INSERT OR REPLACE INTO user_context (user_id, context_summary, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
            """, (user_id, context_summary))
            conn.commit()
            return True
        except Exception as e:
            print(f"Error updating user context: {e}")
            return False
        finally:
            conn.close()

    def get_user_context(self, user_id: str) -> t.Optional[str]:
        """Retrieve user context summary."""
        conn = q.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("""
            SELECT context_summary FROM user_context WHERE user_id = ?
        """, (user_id,))
        
        row = cursor.fetchone()
        conn.close()

        return row[0] if row else None

    def search_university_info(self, query: str, k: int = 3) -> t.List[t.Dict[str, t.Any]]:
        """Search vector database for university information."""
        if self.vector_db is None:
            print("Vector database not initialized.")
            return []

        try:
            results = self.vector_db.similarity_search(query, k=k)
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "content": result.page_content,
                    "metadata": result.metadata
                })
            return formatted_results
        except Exception as e:
            print(f"Error searching vector DB: {e}")
            return []
