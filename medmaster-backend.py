from flask import Flask, request, jsonify
from flask_cors import CORS
import sqlite3
import os
import base64
from datetime import datetime
import json

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Configuration
UPLOAD_FOLDER = 'uploads'
DATABASE = 'medmaster.db'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize database
def init_db():
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    
    # Table for uploaded files
    c.execute('''CREATE TABLE IF NOT EXISTS files (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT NOT NULL,
        filepath TEXT NOT NULL,
        filetype TEXT,
        filesize INTEGER,
        uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        file_data BLOB
    )''')
    
    # Table for chat history
    c.execute('''CREATE TABLE IF NOT EXISTS chats (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        chat_id TEXT UNIQUE NOT NULL,
        title TEXT,
        messages TEXT,
        summary TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')
    
    conn.commit()
    conn.close()

init_db()

# Helper function to get DB connection
def get_db():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

# Available models endpoint
@app.route('/api/models', methods=['GET'])
def get_models():
    models = [
        {"id": "medmaster-gpt4", "name": "MedMaster GPT-4"},
        {"id": "medmaster-claude", "name": "MedMaster Claude"},
        {"id": "medmaster-llama", "name": "MedMaster Llama Medical"}
    ]
    return jsonify({"models": models})

# File ingestion endpoint
@app.route('/api/ingest', methods=['POST'])
def ingest_files():
    if 'files' not in request.files:
        return jsonify({"error": "No files provided"}), 400
    
    files = request.files.getlist('files')
    saved = []
    
    conn = get_db()
    c = conn.cursor()
    
    for file in files:
        if file.filename:
            # Generate unique filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{timestamp}_{file.filename}"
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            
            # Save file to disk
            file.save(filepath)
            
            # Read file as binary for database storage
            with open(filepath, 'rb') as f:
                file_data = f.read()
            
            # Store in database
            c.execute('''INSERT INTO files (filename, filepath, filetype, filesize, file_data)
                        VALUES (?, ?, ?, ?, ?)''',
                     (file.filename, filepath, file.content_type, len(file_data), file_data))
            
            saved.append({
                "filename": file.filename,
                "filepath": filepath,
                "size": len(file_data)
            })
    
    conn.commit()
    conn.close()
    
    return jsonify({
        "saved": saved,
        "count": len(saved),
        "message": f"Successfully uploaded {len(saved)} file(s)"
    })

# Chat endpoint with agent-aware responses
@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    model = data.get('model', 'medmaster-gpt4')
    messages = data.get('messages', [])
    agents = data.get('agents', {
        'nlp': True,
        'retrieval': True,
        'img': True,
        'dx': False
    })
    
    if not messages:
        return jsonify({"error": "No messages provided"}), 400
    
    # Get the last user message
    last_message = messages[-1]['content'] if messages else ""
    
    # Generate response based on active agents
    response = generate_response(last_message, agents, model)
    
    return jsonify({
        "reply": response,
        "model": model,
        "agents_used": agents
    })

def generate_response(user_message, agents, model):
    """Generate dummy response based on active agents"""
    
    # Base response
    response_parts = []
    
    # Clinical NLP agent
    if agents.get('nlp', False):
        response_parts.append("🔍 [Clinical NLP] Analyzing medical terminology and entities...")
        if any(word in user_message.lower() for word in ['pain', 'symptom', 'fever', 'cough']):
            response_parts.append("   → Detected clinical symptoms in query")
    
    # Retrieval agent
    if agents.get('retrieval', False):
        response_parts.append("📚 [Retrieval] Searching medical knowledge base...")
        response_parts.append("   → Found 3 relevant medical references")
    
    # Image Segmentation agent
    if agents.get('img', False):
        # Check if there are uploaded images
        conn = get_db()
        c = conn.cursor()
        c.execute("SELECT COUNT(*) as count FROM files WHERE filetype LIKE 'image/%'")
        image_count = c.fetchone()['count']
        conn.close()
        
        if image_count > 0:
            response_parts.append("🖼️ [Image Segmentation] Processing medical images...")
            response_parts.append(f"   → Analyzed {image_count} image(s) from database")
            response_parts.append("   → Segmentation complete: Key regions identified")
    
    # Diagnosis agent (experimental)
    if agents.get('dx', False):
        response_parts.append("⚕️ [Diagnosis - Experimental] Running diagnostic analysis...")
        response_parts.append("   → Preliminary assessment generated")
        response_parts.append("   ⚠️ Note: This is an experimental feature")
    
    # Main response content
    response_parts.append("\n💬 Based on your query, here's what I found:")
    
    # Generate contextual response
    if 'diagnos' in user_message.lower():
        response_parts.append(
            "Medical diagnosis requires careful consideration of symptoms, medical history, "
            "and clinical findings. The active agents have processed your query and identified "
            "potential patterns in the data."
        )
    elif any(word in user_message.lower() for word in ['image', 'scan', 'xray', 'mri', 'ct']):
        response_parts.append(
            "Image analysis has been completed. The segmentation model has identified key "
            "anatomical structures and potential areas of interest for further review."
        )
    elif 'medication' in user_message.lower() or 'drug' in user_message.lower():
        response_parts.append(
            "Medication information retrieved from the knowledge base. Please consult with "
            "a healthcare provider before making any changes to medication regimens."
        )
    else:
        response_parts.append(
            f"I've processed your query: '{user_message[:50]}...' using the enabled agents. "
            f"Model: {model}. The system is ready to assist with medical information retrieval "
            "and analysis."
        )
    
    # Add agent status summary
    active_agents = [k for k, v in agents.items() if v]
    response_parts.append(f"\n✅ Active agents: {', '.join(active_agents) if active_agents else 'None'}")
    
    return "\n".join(response_parts)

# Get uploaded files
@app.route('/api/files', methods=['GET'])
def get_files():
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT id, filename, filetype, filesize, uploaded_at FROM files ORDER BY uploaded_at DESC")
    files = [dict(row) for row in c.fetchall()]
    conn.close()
    
    return jsonify({"files": files})

# Get specific file
@app.route('/api/files/<int:file_id>', methods=['GET'])
def get_file(file_id):
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT * FROM files WHERE id = ?", (file_id,))
    file = c.fetchone()
    conn.close()
    
    if not file:
        return jsonify({"error": "File not found"}), 404
    
    return jsonify({
        "id": file['id'],
        "filename": file['filename'],
        "filetype": file['filetype'],
        "filesize": file['filesize'],
        "uploaded_at": file['uploaded_at'],
        "data": base64.b64encode(file['file_data']).decode('utf-8') if file['file_data'] else None
    })

# Delete file
@app.route('/api/files/<int:file_id>', methods=['DELETE'])
def delete_file(file_id):
    conn = get_db()
    c = conn.cursor()
    
    # Get file path before deleting
    c.execute("SELECT filepath FROM files WHERE id = ?", (file_id,))
    file = c.fetchone()
    
    if file:
        # Delete from filesystem
        if os.path.exists(file['filepath']):
            os.remove(file['filepath'])
        
        # Delete from database
        c.execute("DELETE FROM files WHERE id = ?", (file_id,))
        conn.commit()
    
    conn.close()
    
    return jsonify({"message": "File deleted successfully"})

# Health check endpoint
@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "version": "1.0.0",
        "database": "connected",
        "timestamp": datetime.now().isoformat()
    })

if __name__ == '__main__':
    print("🏥 MedMaster Backend Starting...")
    print("📂 Upload folder:", UPLOAD_FOLDER)
    print("💾 Database:", DATABASE)
    print("🌐 Server running on http://localhost:8787")
    app.run(host='0.0.0.0', port=8787, debug=True)
