import httpx

from langchain_community.vectorstores import (
    Chroma
)

from langchain_openai import (
    OpenAIEmbeddings
)

from config import (
    BASE_URL,
    EMBEDDING_MODEL,
    VECTOR_DB,
)


# -----------------------------------
# HTTP CLIENT
# -----------------------------------

client = httpx.Client(
    verify=False
)


# -----------------------------------
# DOMAIN GUARDRAILS
# -----------------------------------

ALLOWED_TOPICS = [

    "carbon",
    "carbon footprint",
    "co2",
    "co2e",
    "sustainability",
    "green ai",
    "llm",
    "energy",
    "energy usage",
    "power consumption",
    "emissions",
    "production log",
    "net zero",
    "token usage",
    "model efficiency",
    "environmental impact",
    "green prompt",
]


def is_allowed_query(query):

    query = query.lower()

    for topic in ALLOWED_TOPICS:

        if topic in query:
            return True

    return False


def rejection_message():

    return """
This chatbot only answers:

• AI Carbon Footprint
• Sustainability Analytics
• CO2e Emissions
• Green AI
• LLM Energy Usage
• Production Log Analysis

Please ask a sustainability-related query.
"""


# -----------------------------------
# SYSTEM PROMPT
# -----------------------------------

SYSTEM_PROMPT = """
You are GreenPrompt AI.

You are a domain expert in:

1. Carbon footprint estimation
2. CO2e emissions
3. Sustainable AI
4. LLM energy optimization
5. Production log sustainability
6. Enterprise AI efficiency

STRICT RULES:

- Reject unrelated questions.
- ONLY answer sustainability topics.
- Use retrieved context.
- Never hallucinate.
- Give enterprise-grade answers.
- Suggest optimization strategies.

If context is unavailable say:

'Insufficient sustainability data available.'
"""


# -----------------------------------
# EMBEDDING MODEL
# SAME MODEL AS INGEST.PY
# -----------------------------------

print("Loading embedding model...")

embedding_model = OpenAIEmbeddings(
    base_url=BASE_URL,
    model=EMBEDDING_MODEL,
    api_key="YOUR_API_KEY",
    http_client=client
)


# -----------------------------------
# LOAD CHROMADB
# -----------------------------------

print("Connecting to ChromaDB...")

vectordb = Chroma(
    persist_directory=VECTOR_DB,
    embedding_function=embedding_model
)


retriever = vectordb.as_retriever(
    search_kwargs={"k": 4}
)


# -----------------------------------
# RETRIEVAL FUNCTION
# -----------------------------------

def retrieve_context(query):

    try:

        docs = retriever.invoke(query)

        if not docs:
            return (
                "No relevant context found."
            )

        context = "\n\n".join(
            [
                doc.page_content
                for doc in docs
            ]
        )

        return context

    except Exception as e:

        return (
            f"Retrieval Error: {str(e)}"
        )
