import os
from flask import Flask, request, jsonify
from llama_stack_client import LlamaStackClient
from llama_stack_client.types import UserMessage, SystemMessage

app = Flask(__name__)

# --- Configuration ---
STACK_HOST = os.getenv("LLAMA_STACK_HOST", "172.21.137.132")
STACK_PORT = os.getenv("LLAMA_STACK_PORT", "8321")
VECTOR_DB_ID = os.getenv("VECTOR_DB_ID", "pension_vector_db_002") 
LLM_MODEL_ID = "llama-31-8b-instruct-quantizedw4a16-150" 

# --- Client Setup ---
ls_client = None

def get_client():
    global ls_client
    if ls_client is None:
        url = f"http://{STACK_HOST}:{STACK_PORT}"
        try:
            ls_client = LlamaStackClient(base_url=url)
            print(f"✅ Connected to Llama Stack at {url}")
        except Exception as e:
            print(f"❌ Connection Failed: {e}")
            raise e
    return ls_client

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"}), 200

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({"error": "Missing 'query' field"}), 400

    user_text = data['query']
    
    try:
        client = get_client()

        # --- STEP 1: RETRIEVAL (Vector IO) ---
        print(f"🔍 Searching Vector DB: {VECTOR_DB_ID}...")
        
        # FIX 1: Pass raw text to 'query'. 
        # The Llama Stack handles the embedding automatically based on your DB config.
        # FIX 2: Use 'top_k' instead of 'max_chunks'.
        search_results = client.vector_io.query(
            vector_db_id=VECTOR_DB_ID,
            query=user_text,  
            params={
                "top_k": 3,
                "score_threshold": 0.5 
            }
        )
        
        # Extract text content from the results
        chunks = [chunk.content for chunk in search_results.chunks]
        
        if not chunks:
            context_block = "No relevant documents found."
        else:
            context_block = "\n---\n".join(chunks)

        # --- STEP 2: GENERATION (Inference) ---
        print(f"🤖 Generating Answer...")
        
        # FIX 3: Create Message Objects (Do not pass strings directly)
        system_prompt = f"""
        You are a helpful assistant. Answer the user's question using ONLY the context below.
        
        CONTEXT:
        {context_block}
        """
        
        messages_list = [
            SystemMessage(content=system_prompt, role="system"),
            UserMessage(content=user_text, role="user")
        ]
        
        # FIX 4: Use 'model_id' (not 'model')
        response = client.inference.chat_completion(
            model_id=LLM_MODEL_ID, 
            messages=messages_list,
            stream=False
        )
        
        bot_answer = response.completion_message.content

        return jsonify({
            "answer": bot_answer,
            "context_used": chunks
        })

    except Exception as e:
        app.logger.error(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # In OpenShift, Gunicorn will actually run this, but this helps local debug
    app.run(host='0.0.0.0', port=8080)
