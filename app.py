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
                ts DATETIME DEFAULT CURRENT_TIMESTAMP
            )""")
            db.commit()

def save_msg(sid, role, msg):
    with db_lock:
        db = get_db()
        db.execute("INSERT INTO chats(session_id, role, message) VALUES (?,?,?)", (sid, role, msg))
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
                # If code block is open, append directly without extra fences
                updated_message = existing_message + new_content_chunk
            else:
                # Normal append for non-code or closed code blocks
                updated_message = existing_message + new_content_chunk
            db.execute("UPDATE chats SET message=? WHERE id=?", (updated_message, last_bot_msg['id']))
        else:
            updated_message = new_content_chunk
            save_msg(sid, "bot", updated_message)
        db.commit()
        return updated_message

def load_msgs(sid):
    db = get_db()
    cursor = db.execute("SELECT role, message FROM chats WHERE session_id=? ORDER BY ts ASC", (sid,))
    messages = []
    for row in cursor.fetchall():
        role = "assistant" if row['role'] == 'bot' else row['role']
        clean_message = re.sub(r'<think>[\s\S]*?</think>', '', row['message'], flags=re.IGNORECASE).strip()
        if clean_message:
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
    api_messages = [
        {"parts": [{"type": "text", "text": msg['content']}], "id": str(uuid.uuid4())[:12], "role": msg['role']}
        for msg in chat_history
    ]
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
# Flask Routes
# ==============================================================================

@app.route("/")
def index():
    return send_file('index.html')

@app.route('/favicon.ico')
def favicon():
    return '', 204

@app.route("/upload_file", methods=["POST"])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    try:
        filename = file.filename.lower()
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_bytes = file.read()
            image = Image.open(io.BytesIO(image_bytes))
            width, height = image.size
            file.seek(0)
            encoded_string = base64.b64encode(image_bytes).decode('utf-8')
            mime_type = file.mimetype
            base64_uri = f"data:{mime_type};base64,{encoded_string}"
            return jsonify({
                "id": str(uuid.uuid4()),
                "name": file.filename,
                "size": len(image_bytes),
                "width": width,
                "height": height,
                "fileType": mime_type,
                "base64": base64_uri,
                "type": "image"
            })
        elif filename.endswith(('.py', '.js', '.txt')):
            content = file.read().decode('utf-8')
            return jsonify({
                "id": str(uuid.uuid4()),
                "name": file.filename,
                "size": len(content),
                "content": content,
                "type": "code"
            })
        else:
            return jsonify({"error": "Unsupported file type. Use images (.png, .jpg, .jpeg) or code files (.py, .js, .txt)"}), 400
    except Exception as e:
        return jsonify({"error": f"Failed to process file: {str(e)}"}), 500

@app.route("/execute_code", methods=["POST"])
def execute_code():
    try:
        data = request.json
        code = data.get("code")
        language = data.get("language", "python")
        if not code:
            return jsonify({"error": "No code provided"}), 400
        if language != "python":
            return jsonify({"error": "Only Python execution is supported currently"}), 400
        return jsonify({"output": "Code execution is not fully implemented server-side. Use client-side Pyodide for now."})
    except Exception as e:
        return jsonify({"error": f"Execution error: {str(e)}"}), 500

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.json
        sid = data["session"]
        model = data.get("model", "claude-sonnet-3.7")
        action = data.get("action", "chat")
        is_reasoning_enabled = data.get("isReasoningEnabled", True)

        if action == "chat":
            text = data["text"]
            file_info = data.get("fileInfo")
            user_message_to_save = f"[File: {file_info['name']}]\n{text}" if file_info else text
            save_msg(sid, "user", user_message_to_save)
            chat_history = load_msgs(sid)
        elif action == "continue":
            chat_history = load_msgs(sid)
            if chat_history and chat_history[-1]['role'] == 'assistant':
                last_content = chat_history[-1]['content']
                last_lines = '\n'.join(last_content.split('\n')[-3:])
                last_line = last_content.split('\n')[-1].rstrip()
                open_code_block = last_content.count('```') % 2 == 1
                if open_code_block:
                    continue_content = (
                        f"Please continue the code precisely from the last character of the incomplete line: '{last_line}'.\n"
                        f"Start your response with the exact characters needed to complete the line, without repeating any part of '{last_line}', "
                        f"and without adding any extra spaces, newlines, ``` fences, or introductory phrases. For example, if the last line was 'parent_i', "
                        f"start with 'd' to complete 'parent_id' seamlessly."
                    )
                else:
                    continue_content = (
                        f"Please continue the response precisely from where you left off. "
                        f"The last part was:\n{last_lines}\n"
                        f"Start with the next sentence or content without repetition, extra spaces, newlines, or introductory phrases."
                    )
                continue_prompt = {
                    'role': 'user',
                    'content': continue_content
                }
                chat_history.append(continue_prompt)
            else:
                return Response("No previous bot message to continue.", status=400)
            text = "continue"
            file_info = None
        else:
            return Response("Invalid action.", status=400)

        def gen():
            buffer = ""
            code_block_open = False
            try:
                if model == 'claude-sonnet-3.7':
                    last_line = chat_history[-1]['content'].split('\n')[-1].rstrip() if action == "continue" and chat_history else None
                    for chunk_text, is_code_block_open in stream_claude_sonnet(chat_history, is_reasoning_enabled, is_continuation=(action == "continue"), last_partial_line=last_line):
                        buffer += chunk_text
                        code_block_open = is_code_block_open
                        yield chunk_text
                else:
                    error_msg = f"🚫 The selected model '{model}' is not supported."
                    yield error_msg
                    buffer = error_msg
            except requests.exceptions.RequestException as e:
                error_msg = f"🤖 **Connection Error**\n\nI couldn't reach the AI service for model '{model}'. Details: {e}"
                yield error_msg
                buffer = error_msg
            except Exception as e:
                error_msg = f"🤖 **System Error**\n\nUnexpected error: {str(e)}"
                yield error_msg
                buffer = error_msg
            if buffer:
                with app.app_context():
                    if action == "continue":
                        update_last_bot_message(sid, buffer, code_block_open)
                    else:
                        save_msg(sid, "bot", buffer)

        return Response(stream_with_context(gen()), mimetype="text/plain; charset=utf-8")
    except Exception as e:
        return Response(f"Server error: {str(e)}", status=500)

if __name__ == "__main__":
    init_db()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
