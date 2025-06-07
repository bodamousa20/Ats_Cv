import spacy
from transformers import BertTokenizer, BertModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch
from collections import defaultdict
import re
from init_spacy import initialize_spacy

# Load models (initialize once)
nlp = initialize_spacy()
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')
sentence_bert = SentenceTransformer('bert-base-nli-mean-tokens')  # Alternative for faster embeddings

class EnhancedATSSystem:
    def __init__(self):
        # Initialize models
        self.nlp = nlp
        self.tokenizer = tokenizer
        self.bert_model = bert_model
        self.sentence_bert = sentence_bert
        
        # Configure GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bert_model.to(self.device)
        
        # Enhanced skill normalization dictionary
        self.skill_normalization = {
            'python': 'Python',
            'java': 'Java',
            'spring boot': 'Spring Boot',
            'sql': 'SQL',
            'mysql': 'MySQL',
            'mongodb': 'MongoDB',
            'nodejs': 'NodeJS',
            'node.js': 'NodeJS',
            'react': 'React',
            'reactjs': 'ReactJS',
            'php': 'PHP',
            'llm': 'LLM',
            'ai': 'AI',
            'docker': 'Docker',
            'git': 'Git',
            'c++': 'C++',
            'html': 'HTML',
            'restful': 'RESTful',
            'jwt': 'JWT'
        }
        
        # Education degree patterns
        self.education_degrees = {
            'bachelor': 'Bachelor',
            'bs': 'Bachelor',
            'b.sc': 'Bachelor',
            'computer science': 'Computer Science',
            'information technology': 'Information Technology',
            'cs': 'Computer Science',
            'it': 'Information Technology'
        }

    def extract_entities(self, text):
        """Enhanced entity extraction with better education and skill matching"""
        doc = self.nlp(text.lower())
        entities = defaultdict(list)
        
        # Extract skills from normalization dictionary
        for token in doc:
            if token.text in self.skill_normalization:
                entities['skills'].append(self.skill_normalization[token.text])
        
        # Extract multi-word skills
        for chunk in doc.noun_chunks:
            chunk_text = chunk.text.lower()
            if chunk_text in self.skill_normalization:
                entities['skills'].append(self.skill_normalization[chunk_text])
        
        # Extract experience durations
        for ent in doc.ents:
            if ent.label_ == 'DATE' and ('year' in ent.text or 'month' in ent.text):
                entities['experience'].append(ent.text)
        
        # Enhanced education extraction
        education_keywords = ['degree', 'bachelor', 'master', 'phd', 'study', 'major']
        for sent in doc.sents:
            if any(keyword in sent.text for keyword in education_keywords):
                for degree in self.education_degrees:
                    if degree in sent.text:
                        entities['education'].append(self.education_degrees[degree])
        
        # Extract from technical skills section if exists
        if "technical skills" in text.lower():
            tech_skills_section = text.lower().split("technical skills")[1].split("\n\n")[0]
            for skill in self.skill_normalization:
                if skill in tech_skills_section:
                    entities['skills'].append(self.skill_normalization[skill])
        
        # Deduplicate and clean
        for key in entities:
            entities[key] = list(set(entities[key]))
            if key == 'skills':
                entities[key] = [s for s in entities[key] if len(s) > 1]  # Remove single letters
        
        return dict(entities)

    def get_embeddings(self, text):
        """Get BERT embeddings for text"""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
        
        # Use mean pooling of all token embeddings
        embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        return embeddings[0]

    def calculate_similarity(self, resume_text, jd_text):
        """Calculate comprehensive matching score with enhanced education matching"""
        # Get embeddings
        resume_embedding = self.sentence_bert.encode(resume_text)
        jd_embedding = self.sentence_bert.encode(jd_text)
        
        # Calculate cosine similarity
        semantic_similarity = float(cosine_similarity([resume_embedding], [jd_embedding])[0][0])
        
        # Extract entities
        resume_entities = self.extract_entities(resume_text)
        jd_entities = self.extract_entities(jd_text)
        
        # Calculate skill match
        resume_skills = set(resume_entities.get('skills', []))
        jd_skills = set(jd_entities.get('skills', []))
        skill_match = float(len(resume_skills & jd_skills) / max(len(jd_skills), 1))
        
        # Calculate education match
        resume_edu = set(resume_entities.get('education', []))
        jd_edu = set(jd_entities.get('education', []))
        edu_match = 1 if (not jd_edu) or any(re_edu in jd_edu or jd_e in re_edu 
                              for re_edu in resume_edu for jd_e in jd_edu) else 0
        
        # Calculate experience match (simple version)
        exp_match = 1 if ('experience' in resume_entities and 
                         any('2' in exp or '3' in exp for exp in resume_entities['experience'])) else 0
        
        # Calculate final score (weighted average)
        weights = {
            'semantic_similarity': 0.5,
            'skill_match': 0.3,
            'education_match': 0.15,
            'experience_match': 0.05
        }
        
        final_score = float(
            weights['semantic_similarity'] * semantic_similarity +
            weights['skill_match'] * skill_match +
            weights['education_match'] * edu_match +
            weights['experience_match'] * exp_match
        )
        
        # Prepare results
        missing_skills = list(jd_skills - resume_skills)
        missing_edu = list(jd_edu - resume_edu) if not edu_match else []
        
        feedback = self._generate_feedback(
            final_score, 
            missing_skills, 
            missing_edu,
            resume_entities.get('experience', [])
        )
        
        return {
            'ats_score': round(float(final_score * 100), 1),
            'semantic_similarity': round(float(semantic_similarity * 100), 1),
            'skill_match': round(float(skill_match * 100), 1),
            'education_match': float(edu_match * 100),
            'experience_match': float(exp_match * 100),
            'missing_skills': missing_skills,
            'missing_education': missing_edu,
            'feedback': feedback,
            'resume_entities': resume_entities,
            'jd_entities': jd_entities
        }

    def _generate_feedback(self, score, missing_skills, missing_edu, experience):
        """Generate actionable feedback with specific suggestions"""
        feedback = []
        suggestions = []
        
        # Score-based feedback
        if score >= 0.8:
            feedback.append("Excellent match! Your resume strongly aligns with the job requirements.")
        elif score >= 0.6:
            feedback.append("Good match with the job description.")
        else:
            feedback.append("The resume needs improvement to better match the job.")
        
        # Missing skills feedback
        if missing_skills:
            suggestions.append(f"Consider adding these skills: {', '.join(missing_skills[:5])}")
        
        # Missing education feedback
        if missing_edu:
            suggestions.append(f"The job prefers education in: {', '.join(missing_edu)}")
        
        # Experience feedback
        if not any('2' in exp or '3' in exp for exp in experience):
            suggestions.append("Highlight your 2-3 years of relevant experience more prominently")
        
        # Compile feedback
        if suggestions:
            feedback.append("Suggestions for improvement:")
            feedback.extend(suggestions)
        
        return "\n".join(feedback)