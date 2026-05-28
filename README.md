greenprompt/
│
├── data/
│   ├── logs/
│   └── documents/
│
├── vector_db/
│
├── backend/
│   ├── ingest.py    
│   ├── rag.py         
│   └── query.py      
│
└── .env

STEP 1 — Create .env
Paste ur openapikey here: 
greenprompt/.env
OPENAI_API_KEY=your_api_key_here

STEP 2 — Install Packages
cd greenprompt/backend
pip install langchain
pip install langchain-community
pip install langchain-openai
pip install chromadb
pip install sentence-transformers
pip install openai
pip install python-dotenv
pip install tiktoken
pip install pandas

STEP 3 — Create rag.py
STEP 4 — Create query.py

STEP 5 — Run It
cd greenprompt/backend
python query.py

TEST CASES
Allowed Query
1. How can enterprises reduce AI carbon emissions?
Expected:
sustainability answer
recommendations
2. Explain GPT carbon footprint
Expected:
retrieved context answer
3.Rejected Query
Who won IPL?
This chatbot is restricted to:
AI Carbon Footprint Estimation...
