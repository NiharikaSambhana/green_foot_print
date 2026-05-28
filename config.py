# backend/config.py

BASE_URL = "https://genailab.tcs.in"

EMBEDDING_MODEL = (
    "azure/genailab-maas-text-embedding-3-large"
)

CHAT_MODEL = (
    "azure_ai/genailab-maas-DeepSeek-V3-0324"
)

DATA_FOLDER = "../data"

VECTOR_DB = "../vector_db/chroma_db"

CHUNK_SIZE = 1000

CHUNK_OVERLAP = 200
