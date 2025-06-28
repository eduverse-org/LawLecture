from flask import Flask, request, jsonify
import os
import logging
import re
from sentence_transformers import SentenceTransformer
import psycopg2
from psycopg2.extras import RealDictCursor
import numpy as np
from groq import Groq
from dotenv import load_dotenv
from flask_cors import CORS

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

class LegalTeacherPrompt:
    """Generates prompts for a legal teacher persona to create a five-lesson curriculum."""

    def __init__(self):
        """Initialize the legal teacher persona."""
        self.persona = """
You are a highly knowledgeable and patient legal teacher specializing in Indian law. Your goal is to educate a common person or student with no prior legal knowledge, using clear, simple, and engaging language. When asked to generate a curriculum or lessons, create a structured script for exactly five lessons on the specified legal topic, each designed to be spoken for at least 5 minutes (approximately 500-600 words per lesson). Use only plain text, no HTML tags like <span>. Structure each lesson like a classroom lecture with an introduction, main content (broken into clear, numbered steps or sections), relevant examples or analogies, historical or practical context, and a conclusion summarizing key points. Ensure the content is accurate, comprehensive, and suitable for beginners.
"""

    def generate_prompt(self, query, context):
        """Generate a lecture-style prompt for a five-lesson curriculum on the specified topic."""
        if any(keyword in query.lower() for keyword in ["curriculum", "lessons", "teach", "step-wise"]):
            topic_match = re.search(r"(?:curriculum|lessons|teach).*?(?:on|about)\s+(.+?)(?:\.|$)", query, re.IGNORECASE)
            topic = topic_match.group(1).strip() if topic_match else "the specified legal topic"
            prompt = f"""
{self.persona}

Context: {context}

Task: Create a detailed script for a five-lesson curriculum on {topic}. Each lesson must be plain text, approximately 500-600 words (suitable for 5 minutes of speech), and structured as follows:
1. **Introduction**: Briefly introduce the lesson's focus and its importance in the context of Indian law or the specified topic.
2. **Main Content**: Break down the lesson into 3-5 numbered steps or sections, explaining key concepts in simple language.
3. **Examples/Analogies**: Include at least one example or analogy per lesson to make concepts relatable.
4. **Context**: Provide historical or practical context (e.g., relevant cases, amendments, or real-world applications).
5. **Conclusion**: Summarize key takeaways and their relevance to everyday life or legal understanding.

Instructions:
- Ensure exactly five lessons, each clearly labeled (e.g., "Lesson 1: [Title]").
- Use the provided context to ground the content in accurate information, prioritizing Indian law where relevant.
- Avoid HTML tags like <span> or any formatting beyond plain text.
- Make the content engaging, educational, and beginner-friendly.
- If the topic is vague, infer a specific legal topic from the query or context (e.g., Indian Constitution, dowry laws).
"""
        else:
            prompt = f"""
{self.persona}

Context: {context}

Question: {query}

Instructions:
1. Begin with a brief introduction to the topic, explaining its importance in Indian law.
2. Break down the answer into clear, numbered steps or sections, using simple language.
3. Include examples or analogies to make complex legal concepts relatable.
4. Provide historical or practical context to enhance understanding.
5. Conclude with a summary of key takeaways and their relevance to everyday life.
6. Ensure the response is detailed, educational, and structured like a lecture script in plain text.
"""
        return prompt

class LegalLearningPipeline:
    """Handles query processing, document retrieval, and response generation."""
    
    LEGAL_KEYWORDS = [
        "law", r"legal", r"penal", "contract", "justice", "dowry", "constitution", "evidence",
        "act", "section", "court", "judge", "lawyer", "legal", "case", "right", "silence",
        "prohibition", "amendment", "criminal", "civil", "statute", "jurisprudence"
    ]
    NON_LEGAL_KEYWORDS = [
        r"code\b", "coding", "programming", "python", "javascript", "html", "css",
        "algorithm", "software", "database", "debug", "program", "script", "computer"
    ]
    COMPLEX_QUERY_KEYWORDS = ["curriculum", "teach", "step", "lesson", "explain in detail"]

    def __init__(self):
        """Initialize with embedding model, database connection, and Groq client."""
        self.logger = self._setup_logging()
        
        # Initialize embedding model (gte-small)
        self.model = SentenceTransformer('thenlper/gte-small')
        
        self.db_url = os.getenv('DATABASE_URL')
        if not self.db_url:
            raise ValueError("DATABASE_URL not found in environment variables.")
        
        # Initialize Groq client
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables.")
        self.groq = Groq(api_key=groq_api_key)
        
        # Initialize prompt generator
        self.teacher_prompt = LegalTeacherPrompt()

    def _setup_logging(self):
        """Configure logging for the application."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler("legal_learning_app.log"),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)

    def get_db_connection(self):
        """Create and return a database connection using connection string."""
        try:
            conn = psycopg2.connect(self.db_url)
            return conn
        except Exception as e:
            self.logger.error(f"Database connection failed: {str(e)}")
            return None

    def is_legal_query(self, query):
        """Classify query as legal or non-legal based on keywords."""
        query_lower = query.lower()
        
        # Check for non-legal keywords first
        for keyword in self.NON_LEGAL_KEYWORDS:
            if re.search(keyword, query_lower):
                self.logger.info(f"Non-legal query detected: {query} (matched {keyword})")
                return False
        
        # Check for legal keywords
        for keyword in self.LEGAL_KEYWORDS:
            if re.search(keyword, query_lower):
                self.logger.info(f"Legal query detected: {query} (matched {keyword})")
                return True
        
        self.logger.info(f"Non-legal query detected: {query} (no legal keywords)")
        return False

    def estimate_query_complexity(self, query):
        """Estimate query complexity to set max_tokens."""
        query_lower = query.lower()
        for keyword in self.COMPLEX_QUERY_KEYWORDS:
            if keyword in query_lower:
                return 4000
        return 1000

    def cosine_similarity(self, a, b):
        """Calculate cosine similarity between two vectors."""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def retrieve_documents(self, query, n_results=10):
        """Retrieve similar documents from PostgreSQL database."""
        try:
            self.logger.info(f"Processing query: {query}")
            
            # Generate query embedding
            query_embedding = self.model.encode([query])[0]
            
            # Connect to database
            conn = self.get_db_connection()
            if not conn:
                return []
            
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Retrieve all embeddings and texts from database
            cursor.execute("SELECT embedding, content FROM legal_embeddings")
            rows = cursor.fetchall()
            
            if not rows:
                self.logger.warning("No documents found in database")
                return []
            
            # Calculate similarities and rank documents
            similarities = []
            for row in rows:
                # Convert stored embedding back to numpy array
                stored_embedding = np.array(row['embeddings'])
                similarity = self.cosine_similarity(query_embedding, stored_embedding)
                similarities.append({
                    'text': row['text'],
                    'similarity': similarity
                })
            
            # Sort by similarity and return top n_results
            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            top_documents = similarities[:n_results]
            
            cursor.close()
            conn.close()
            
            self.logger.info(f"Retrieved {len(top_documents)} documents")
            return [doc['text'] for doc in top_documents]
            
        except Exception as e:
            self.logger.error(f"Document retrieval failed: {str(e)}")
            return []

    def generate_response(self, query, retrieved_docs):
        """Generate LLM response using legal teacher prompt."""
        try:
            # Prepare context from retrieved documents
            context = "\n".join(retrieved_docs) if retrieved_docs else "No relevant documents found. Use general knowledge about Indian law."
            
            # Generate prompt using the legal teacher persona
            prompt = self.teacher_prompt.generate_prompt(query, context)
            
            # Estimate complexity for token limit
            max_tokens = self.estimate_query_complexity(query)
            
            self.logger.info(f"Generating LLM response with max_tokens={max_tokens}...")
            
            # Generate response using Groq
            response = self.groq.chat.completions.create(
                model="llama3-70b-8192",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens
            )
            
            answer = response.choices[0].message.content.strip()
            self.logger.info("Response generated successfully")
            return answer
            
        except Exception as e:
            self.logger.error(f"LLM response generation failed: {str(e)}")
            return "Error generating response. Please try again."

    def process_query(self, query, n_results=10):
        """Main pipeline: classify query, retrieve documents, generate response."""
        self.logger.info(f"Starting pipeline for query: {query}")
        
        # Check if query is legal-related
        if not self.is_legal_query(query):
            response = "Sorry, I can only assist with legal and law-based questions."
            self.logger.info(f"Rejected non-legal query: {query}")
            return {
                "query": query,
                "response": response,
                "status": "rejected"
            }
        
        # Retrieve relevant documents
        retrieved_docs = self.retrieve_documents(query, n_results)
        
        # Generate response
        response = self.generate_response(query, retrieved_docs)
        
        return {
            "query": query,
            "response": response,
            "documents_found": len(retrieved_docs),
            "status": "success"
        }

# Initialize the pipeline
pipeline = LegalLearningPipeline()

@app.route('/learn', methods=['POST'])
def learn_topic():
    """Main endpoint to process learning queries."""
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data or 'query' not in data:
            return jsonify({
                "error": "Missing 'query' field in request body"
            }), 400
        
        query = data['query'].strip()
        if not query:
            return jsonify({
                "error": "Query cannot be empty"
            }), 400
        
        # Optional parameters
        n_results = data.get('n_results', 10)
        
        # Process the query
        result = pipeline.process_query(query, n_results)
        
        return jsonify(result)
        
    except Exception as e:
        app.logger.error(f"Error processing request: {str(e)}")
        return jsonify({
            "error": "Internal server error",
            "message": str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True)