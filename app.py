from flask import Flask, jsonify, request
from flask_cors import CORS
from Ats_Bert import EnhancedATSSystem
import multiprocessing
import tempfile
import os
import json
import time
import logging
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import gc

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Lazy loading of models and data
_ats_system = None
_df = None
_model = None
_corpus_embeddings = None

def get_ats_system():
    global _ats_system
    if _ats_system is None:
        _ats_system = EnhancedATSSystem()
    return _ats_system

def get_model_and_data():
    global _df, _model, _corpus_embeddings
    if _model is None:
        logger.info("Loading model and data...")
        _df = pd.read_csv('Final_Diversified_Interview_Dataset.csv')
        _model = SentenceTransformer('all-MiniLM-L6-v2')
        _corpus_embeddings = _model.encode(_df['Question'].astype(str).tolist(), convert_to_tensor=True)
        gc.collect()  # Force garbage collection
    return _df, _model, _corpus_embeddings

@app.route('/evaluate', methods=['POST'])
def evaluate_resume():
    try:
        data = request.json
        resume_text = data.get('resume_text', '')
        jd_text = data.get('job_description', '')
        
        if not resume_text or not jd_text:
            return jsonify({'error': 'resume_text and job_description are required'}), 400
        
        ats_system = get_ats_system()
        result = ats_system.calculate_similarity(resume_text, jd_text)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error in evaluate_resume: {str(e)}")
        return jsonify({'error': str(e)}), 500

def get_top_k_unique_questions(user_query, top_k=10):
    try:
        df, model, corpus_embeddings = get_model_and_data()
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
    except Exception as e:
        logger.error(f"Error in get_top_k_unique_questions: {str(e)}")
        raise

@app.route('/get-questions', methods=['GET'])
def get_questions():
    try:
        user_query = request.args.get('query')
        if not user_query:
            return jsonify({"error": "Missing 'query' parameter"}), 400

        results = get_top_k_unique_questions(user_query)
        return jsonify({
            "query": user_query,
            "results": results
        })
    except Exception as e:
        logger.error(f"Error in get_questions: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
