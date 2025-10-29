import os
import time
import json
import sqlite3
import re
import threading
import uuid
import base64
from flask import Flask, Response, stream_with_context, request, jsonify, g, send_file
import requests
from PIL import Image
import io
from queue import Queue

# ==============================================================================
# Database Setup
# ==============================================================================
DB = "chat_history.db"
db_lock = threading.Lock()

app = Flask(__name__)

def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect(DB, timeout=10, check_same_thread=False)
        g.db.row_factory = sqlite3.Row
    return g.db

@app.teardown_appcontext
def close_db(error):
    db = g.pop('db', None)
    if db is not None:
        db.close()

def init_db():
    with app.app_context():
        db = get_db()
        try:
            db.execute("SELECT ts FROM chats LIMIT 1")
        except sqlite3.OperationalError:
            db.execute("DROP TABLE IF EXISTS chats")
            db.execute("""
            CREATE TABLE chats(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                role TEXT,
                message TEXT,
                image_data TEXT,
                media_type TEXT,
                ts DATETIME DEFAULT CURRENT_TIMESTAMP
            )""")
            db.commit()

def save_msg(sid, role, msg, image_data=None, media_type=None):
    with db_lock:
        db = get_db()
        db.execute("INSERT INTO chats(session_id, role, message, image_data, media_type) VALUES (?,?,?,?,?)", (sid, role, msg, image_data, media_type))
        db.commit()

def update_last_bot_message(sid, new_content_chunk, is_code_block_open=False):
    with db_lock:
        db = get_db()
        cursor = db.execute("SELECT id, message FROM chats WHERE session_id=? AND role='bot' ORDER BY ts DESC LIMIT 1", (sid,))
        last_bot_msg = cursor.fetchone()
        updated_message = ""
        if last_bot_msg:
            existing_message = last_bot_msg['message']
            if is_code_block_open:
                updated_message = existing_message + new_content_chunk
            else:
                updated_message = existing_message + new_content_chunk
            db.execute("UPDATE chats SET message=? WHERE id=?", (updated_message, last_bot_msg['id']))
        else:
            updated_message = new_content_chunk
            save_msg(sid, "bot", updated_message)
        db.commit()
        return updated_message

def load_msgs(sid):
    db = get_db()
    cursor = db.execute("SELECT role, message, image_data, media_type FROM chats WHERE session_id=? ORDER BY ts ASC", (sid,))
    messages = []
    for row in cursor.fetchall():
        role = "assistant" if row['role'] == 'bot' else row['role']
        clean_message = re.sub(r'<think>[\s\S]*?</think>', '', row['message'], flags=re.IGNORECASE).strip()

        content = []
        if clean_message:
            content.append({"type": "text", "text": clean_message})

        if row['image_data'] and row['media_type']:
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": row['media_type'],
                    "data": row['image_data']
                }
            })

        if content:
            if role == 'user' and any(c.get("type") == "image" for c in content):
                 messages.append({'role': role, 'content': content})
            else:
                 messages.append({'role': role, 'content': clean_message})

    return messages

# ==============================================================================
# API Integration Section
# ==============================================================================

claude_session = requests.Session()
claude_headers = {
    'authority': 'ai-sdk-reasoning.vercel.app',
    'accept': '*/*',
    'accept-language': 'en-US,en;q=0.9',
    'cache-control': 'no-cache',
    'content-type': 'application/json',
    'origin': 'https://ai-sdk-reasoning.vercel.app',
    'pragma': 'no-cache',
    'referer': 'https://ai-sdk-reasoning.vercel.app/',
    'sec-ch-ua': '"Chromium";v="137", "Not/A)Brand";v="24"',
    'sec-ch-ua-mobile': '?1',
    'sec-ch-ua-platform': '"Android"',
    'sec-fetch-dest': 'empty',
    'sec-fetch-mode': 'cors',
    'sec-fetch-site': 'same-origin',
    'user-agent': 'Mozilla/5.0 (Linux; Android 10; K) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Mobile Safari/537.36',
}
claude_url = 'https://ai-sdk-reasoning.vercel.app/api/chat'

def stream_claude_sonnet(chat_history, is_reasoning_enabled, is_continuation=False, last_partial_line=None):
    api_messages = []
    for msg in chat_history:
        if isinstance(msg['content'], list): # Handle multipart messages (text + image)
            parts = msg['content']
        else: # Handle text-only messages
            parts = [{"type": "text", "text": msg['content']}]

        api_messages.append({
            "parts": parts,
            "id": str(uuid.uuid4())[:12],
            "role": msg['role']
        })

    payload = {
        'selectedModelId': 'sonnet-3.7',
        'isReasoningEnabled': is_reasoning_enabled,
        'id': str(uuid.uuid4())[:12],
        'messages': api_messages,
        'trigger': 'submit-user-message',
    }
    try:
        with claude_session.post(claude_url, headers=claude_headers, json=payload, stream=True, timeout=90) as r:
            r.raise_for_status()
            code_block_open = False
            code_fence_count = 0
            buffer = ""
            for line in r.iter_lines():
                if line:
                    decoded = line.decode('utf-8')
                    if decoded.startswith("data: "):
                        decoded = decoded[6:]
                    if decoded == "[DONE]":
                        continue
                    try:
                        data_json = json.loads(decoded)
                        if data_json.get("type") == "text-delta":
                            delta = data_json.get("delta", "")
                            if delta:
                                # Track code block state
                                code_fence_count += delta.count('```')
                                code_block_open = code_fence_count % 2 == 1
                                if is_continuation and last_partial_line and buffer == "":
                                    # Trim overlapping content and avoid extra whitespace
                                    delta = delta.lstrip()
                                    if delta.startswith(last_partial_line):
                                        delta = delta[len(last_partial_line):]
                                    elif last_partial_line.endswith(delta[0]):
                                        delta = delta[1:]  # Skip the first character if it matches the last one
                                buffer += delta
                                yield delta, code_block_open
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        continue
    except Exception as e:
        yield f"🚨 Claude API Error: {str(e)}", False

# ==============================================================================
# Background Task Management
# ==============================================================================
tasks = {}
tasks_lock = threading.Lock()

def generation_worker(task_id, sid, chat_history, is_reasoning_enabled, model, action):
    with tasks_lock:
        tasks[task_id] = {"status": "running", "result": "", "queue": Queue()}

    buffer = ""
    try:
        generator = stream_claude_sonnet(
            chat_history,
            is_reasoning_enabled,
            is_continuation=(action == "continue"),
            last_partial_line=chat_history[-1]['content'].split('\n')[-1].rstrip() if action == "continue" and chat_history and chat_history[-1]['role'] == 'assistant' else None
        )

        code_block_open = False
        for chunk, is_code_block_open in generator:
            buffer += chunk
            code_block_open = is_code_block_open
            with tasks_lock:
                tasks[task_id]["queue"].put(chunk)

        with app.app_context():
            if action == "continue":
                update_last_bot_message(sid, buffer, code_block_open)
            else:
                save_msg(sid, "bot", buffer)

        with tasks_lock:
            tasks[task_id]["status"] = "completed"
            tasks[task_id]["queue"].put(None) # Signal completion

    except Exception as e:
        with tasks_lock:
            tasks[task_id]["status"] = "error"
            tasks[task_id]["error"] = str(e)
            tasks[task_id]["queue"].put(None)

def cleanup_tasks():
    while True:
        time.sleep(600) # Run every 10 minutes
        with tasks_lock:
            stale_tasks = []
            for task_id, task in tasks.items():
                if task['status'] in ['completed', 'error']:
                    stale_tasks.append(task_id)
            for task_id in stale_tasks:
                del tasks[task_id]

# ==============================================================================
# Flask Routes
# ==============================================================================

@app.route("/")
def index():
    return send_file('index.html')

@app.route('/favicon.ico')
def favicon():
    return '', 204

@app.route("/chat", methods=["POST"])
def chat():
    try:
        image_data = None
        media_type = None
        if request.content_type.startswith('multipart/form-data'):
            data = request.form.to_dict()
            if 'file' in request.files:
                file = request.files.get('file')
                if file and file.filename:
                    image_bytes = file.read()
                    image_data = base64.b64encode(image_bytes).decode('utf-8')
                    media_type = file.mimetype
        else:
            data = request.json

        sid = data["session"]
        model = data.get("model", "claude-sonnet-3.7")
        action = data.get("action", "chat")
        raw_reasoning = data.get("isReasoningEnabled", True)
        is_reasoning_enabled = str(raw_reasoning).lower() == 'true'

        if action == "chat":
            text = data["text"]
            save_msg(sid, "user", text, image_data=image_data, media_type=media_type)
            chat_history = load_msgs(sid)
        elif action == "continue":
            chat_history = load_msgs(sid)
            if not (chat_history and chat_history[-1]['role'] == 'assistant'):
                 return Response("No previous bot message to continue.", status=400)

        task_id = str(uuid.uuid4())
        thread = threading.Thread(
            target=generation_worker,
            args=(task_id, sid, chat_history, is_reasoning_enabled, model, action)
        )
        thread.daemon = True
        thread.start()

        return jsonify({"task_id": task_id})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/chat/stream/<task_id>")
def stream(task_id):
    app.logger.info(f"Streaming request received for task_id: {task_id}")
    def generate():
        with tasks_lock:
            task = tasks.get(task_id)
            if not task:
                app.logger.error(f"Task not found for task_id: {task_id}")
                yield f"data: {{'error': 'Task not found'}}\n\n"
                return

        q = task["queue"]
        while True:
            chunk = q.get()
            if chunk is None:
                break
            yield f"data: {json.dumps({'chunk': chunk})}\n\n"

    return Response(stream_with_context(generate()), mimetype="text/event-stream")


if __name__ == "__main__":
    init_db()
    cleanup_thread = threading.Thread(target=cleanup_tasks, daemon=True)
    cleanup_thread.start()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
