🧠 Medical RAG System (Groq + FAISS + Sentence Transformers)

A lightweight Retrieval-Augmented Generation (RAG) system for medical question answering and diagnostic support using:

✔ Groq LLaMA 3.1 70B
✔ SentenceTransformer embeddings
✔ FAISS vector search
✔ Medical datasets (MedQuAD + Symptoms + Diseases)
✔ Python backend (Flask-ready)

This project enables AI-assisted medical information retrieval based only on verified medical datasets, not hallucinations.

🚀 Features

🔍 FAISS-based vector search over medical datasets

🩺 Supports symptom → disease queries

📖 Integration with MedQuAD medical Q&A dataset

🧠 Medical reasoning using Groq (LLaMA 3.1 70B)

⚠ Safety-focused responses (no dosage, no prescriptions)

🔧 Ready as a backend API or command-line tool

🌱 Lightweight, simple, and fast

📂 Project Structure
CUSTOMER SUPPORT AGENT /
│
├── app.py                 # Optional Flask server
├── medical_rag.py         # Main RAG backend
│
├── disease_symptoms.csv
├── medquad.csv
├── symptom_Description.csv
├── symptom_precaution.csv
├── Symptom-severity.csv
│
├── prepared/              # Preprocessed JSON/CSV (ignored)
├── templates/             # Optional UI (ignored)
├── requirements.txt
└── .gitignore


⚠ Dataset + .env files are ignored and not included for safety

📦 Installation
1️⃣ Clone the repo
git clone https://github.com/YOUR_USERNAME/medical-rag-groq.git
cd medical-rag-groq

2️⃣ Create & activate environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux

3️⃣ Install dependencies
pip install -r requirements.txt

🔑 Environment Variables

Create a .env file:

GROQ_API_KEY=your_groq_key_here

▶️ Run (Command-line mode)
python medical_rag.py


Example:

💬 Ask a medical question (or 'quit'): What are symptoms of malaria?

▶️ Optional: Run Flask server
python app.py

🧬 Tech Stack
Component	Tool
Embeddings	SentenceTransformer (all-MiniLM-L6-v2)
Vector DB	FAISS
LLM	Groq LLaMA 3.1 70B
Data	MedQuAD, Disease CSVs
Backend	Python
Env	dotenv
📊 Datasets Used (Not Included in Repo)
File	Description
medquad.csv	Medical Q&A dataset
disease_symptoms.csv	Disease–symptom relationships
symptom_Description.csv	Disease descriptions
symptom_precaution.csv	Medical precautions
Symptom-severity.csv	Severity scores

Place the CSVs in project root before running.

🧠 Example Output
Query: What are symptoms of dengue?

Answer:
Based on the knowledge base, Dengue symptoms include high fever,
rash, muscle pain, and joint pain. Consult a healthcare provider
for confirmation and treatment.

⚠ Safety Disclaimer

This project is for educational and research purposes only.
It does NOT replace professional medical diagnosis or treatment.

💡 Future Plans

UI dashboard

API endpoints (FastAPI / Flask)

Cloud deployment

Additional datasets

Medication knowledge base

Fine-tuned MedLM

👨‍💻 Author

Prabhav Verma
🔥 Open Source + ML Enthusiast
