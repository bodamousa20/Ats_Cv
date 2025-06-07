from flask import Flask, jsonify, request
from flask_cors import CORS
from Ats_Bert import EnhancedATSSystem  # import your class
import multiprocessing
import tempfile
import os
import json
import time
import logging
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# Initialize Flask app
app = Flask(__name__)
CORS(app)


ats_system = EnhancedATSSystem()  # instantiate once for reuse

@app.route('/evaluate', methods=['POST'])
def evaluate_resume():
    data = request.json
    resume_text = data.get('resume_text', '')
    jd_text = data.get('job_description', '')
    
    if not resume_text or not jd_text:
        return jsonify({'error': 'resume_text and job_description are required'}), 400
    
    result = ats_system.calculate_similarity(resume_text, jd_text)
    return jsonify(result)

#######
# Load data and model once at startup
df = pd.read_csv('Final_Diversified_Interview_Dataset.csv')
model = SentenceTransformer('all-MiniLM-L6-v2')

corpus = df['Question'].astype(str).tolist()
corpus_embeddings = model.encode(corpus, convert_to_tensor=True)

def get_top_k_unique_questions(user_query, top_k=10):
    query_embedding = model.encode(user_query, convert_to_tensor=True)
    hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=len(df))[0]
    
    seen = set()
    results = []
    for hit in hits:
        idx = hit['corpus_id']
        question = df.iloc[idx]['Question']
        if question not in seen:
            seen.add(question)
            results.append({
                "question": question,
                "score": round(hit['score'], 4)
            })
        if len(results) == top_k:
            break
    return results

@app.route('/get-questions', methods=['GET'])
def get_questions():
    user_query = request.args.get('query')
    if not user_query:
        return jsonify({"error": "Missing 'query' parameter"}), 400

    results = get_top_k_unique_questions(user_query)
    return jsonify({
        "query": user_query,
        "results": results
    })
########

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
