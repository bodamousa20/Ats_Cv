import spacy
import sys
import logging

def initialize_spacy():
    try:
        # Try to load the model
        nlp = spacy.load('en_core_web_sm')
        logging.info("Successfully loaded spaCy model")
        return nlp
    except OSError:
        logging.error("Error loading spaCy model. Attempting to download...")
        try:
            # Try to download the model
            spacy.cli.download('en_core_web_sm')
            nlp = spacy.load('en_core_web_sm')
            logging.info("Successfully downloaded and loaded spaCy model")
            return nlp
        except Exception as e:
            logging.error(f"Failed to download spaCy model: {str(e)}")
            sys.exit(1)

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    # Initialize spaCy
    nlp = initialize_spacy() 