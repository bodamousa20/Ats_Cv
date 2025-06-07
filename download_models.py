import os
import logging
from transformers import BertTokenizer, BertModel
from sentence_transformers import SentenceTransformer
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_model(model_name, model_type):
    try:
        logger.info(f"Downloading {model_type}: {model_name}")
        if model_type == "bert":
            tokenizer = BertTokenizer.from_pretrained(model_name)
            model = BertModel.from_pretrained(model_name)
            logger.info(f"Successfully downloaded {model_name}")
        elif model_type == "sentence_transformer":
            model = SentenceTransformer(model_name)
            logger.info(f"Successfully downloaded {model_name}")
        return True
    except Exception as e:
        logger.error(f"Error downloading {model_name}: {str(e)}")
        return False

def main():
    # Create cache directory
    cache_dir = os.path.join(os.getcwd(), "model_cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    # Set environment variables
    os.environ["TRANSFORMERS_CACHE"] = cache_dir
    os.environ["HF_HOME"] = cache_dir
    
    # Models to download
    models = [
        ("bert-base-uncased", "bert"),
        ("bert-base-nli-mean-tokens", "sentence_transformer"),
        ("all-MiniLM-L6-v2", "sentence_transformer")
    ]
    
    # Download each model
    success = True
    for model_name, model_type in models:
        if not download_model(model_name, model_type):
            success = False
    
    if not success:
        raise Exception("Failed to download one or more models")

if __name__ == "__main__":
    main() 