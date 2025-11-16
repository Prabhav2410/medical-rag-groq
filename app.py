"""
Flask Web Application for Medical RAG Chatbot
File: app.py
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
from medical_rag import MedicalRAG

app = Flask(__name__)
CORS(app)

# Initialize RAG system
print("Initializing Medical RAG System...")
try:
    rag = MedicalRAG(
        medquad_csv="medquad.csv",
        disease_csv="disease_symptoms.csv"
    )
    print("✅ RAG System initialized successfully!")
except Exception as e:
    print(f"❌ Failed to initialize RAG: {e}")
    rag = None

@app.route('/')
def home():
    """Render main page"""
    return render_template('index.html')

@app.route('/api/diagnose', methods=['POST'])
def diagnose():
    """API endpoint for diagnosis"""
    try:
        if not rag:
            return jsonify({
                'success': False,
                'error': 'RAG system not initialized. Check your CSV files and API keys.'
            }), 500
        
        data = request.get_json()
        query = data.get('query', '').strip()
        
        if not query:
            return jsonify({
                'success': False,
                'error': 'Please provide a query.'
            }), 400
        
        # Get diagnosis
        result = rag.diagnose(query, show_sources=True)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'rag_initialized': rag is not None
    })

if __name__ == '__main__':
    # Create templates folder if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    print("\n" + "=" * 70)
    print("🏥 Medical RAG Chatbot Server")
    print("=" * 70)
    print("Server starting at: http://localhost:5000")
    print("=" * 70 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)