from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import (
    HuggingFaceEmbeddings
)

# -----------------------------
# DOMAIN RESTRICTION
# -----------------------------

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
    "emission",
    "net zero",
    "climate",
    "production logs",
    "token usage",
    "model efficiency",
    "ai environmental impact"
]


def is_allowed_query(query):

    query = query.lower()

    for topic in ALLOWED_TOPICS:
        if topic in query:
            return True

    return False


def rejection_message():

    return """
This chatbot is restricted to:

• AI Carbon Footprint Estimation
• Sustainability Analytics
• LLM Energy Consumption
• CO2e Emission Analysis
• Production Log Sustainability

Please ask a sustainability-related question.
"""


# -----------------------------
# SYSTEM PROMPT
# -----------------------------

SYSTEM_PROMPT = """
You are GreenPrompt AI.

You are an enterprise sustainability
and carbon footprint expert.

You ONLY answer:

1. Carbon footprint estimation
2. CO2e emissions
3. Sustainable AI usage
4. LLM energy consumption
5. Production log analysis
6. Sustainability reporting
7. Green AI optimization

STRICT RULES:

- Reject unrelated queries.
- Never hallucinate.
- Use retrieved context.
- Be enterprise professional.
- Give optimization suggestions.

If context is unavailable say:

'Insufficient sustainability data available.'
"""


# -----------------------------
# VECTOR DATABASE CONNECTION
# -----------------------------

print("Loading embedding model...")

embedding_model = (
    HuggingFaceEmbeddings(
        model_name=
        "sentence-transformers/all-MiniLM-L6-v2"
    )
)

print("Connecting to ChromaDB...")

vector_db = Chroma(
    persist_directory="../vector_db",
    embedding_function=embedding_model
)

retriever = vector_db.as_retriever(
    search_kwargs={"k": 4}
)


# -----------------------------
# RETRIEVAL FUNCTION
# -----------------------------

def retrieve_context(query):

    try:

        docs = (
            retriever
            .get_relevant_documents(query)
        )

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


# -----------------------------
# INTELLIGENT SUMMARIZER
# -----------------------------

def summarize_results(results):

    total_emission = results.get(
        "total_emission",
        0
    )

    highest_team = results.get(
        "highest_team",
        "Unknown"
    )

    severe = results.get(
        "high_severity_count",
        0
    )

    summary = f"""

========== Sustainability Summary ==========

Total CO2e Emission:
{total_emission} kg

Highest Emission Team:
{highest_team}

High Severity Cases:
{severe}

Recommendations:

• Reduce token usage

• Use smaller models

• Enable response caching

• Shift workloads
to low-carbon regions

• Optimize prompts
"""

    return summary
