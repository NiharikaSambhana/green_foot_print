import httpx

from langchain_openai import (
    ChatOpenAI
)

from rag import (
    retrieve_context,
    is_allowed_query,
    rejection_message,
    SYSTEM_PROMPT
)

from config import (
    BASE_URL,
    CHAT_MODEL
)


# -----------------------------------
# HTTP CLIENT
# -----------------------------------

client = httpx.Client(
    verify=False
)


# -----------------------------------
# LLM CONNECTION
# -----------------------------------

llm = ChatOpenAI(

    base_url=BASE_URL,

    model=CHAT_MODEL,

    api_key="sk-k4NQXpytjb1jTyqHPnjcFQ",

    temperature=0.3,

    http_client=client
)


# -----------------------------------
# CHATBOT FUNCTION
# -----------------------------------

def ask_greenprompt(query):

    # DOMAIN GUARDRAIL

    if not is_allowed_query(query):

        return rejection_message()

    # RETRIEVE CONTEXT

    context = retrieve_context(
        query
    )

    # FINAL PROMPT

    final_prompt = f"""
Retrieved Context:

{context}

User Question:

{query}

Generate a professional
enterprise sustainability response.
"""

    try:

        response = llm.invoke(

            [

                (
                    "system",
                    SYSTEM_PROMPT
                ),

                (
                    "human",
                    final_prompt
                )

            ]
        )

        return response.content

    except Exception as e:

        return (
            f"LLM Error: {str(e)}"
        )


# -----------------------------------
# TERMINAL TEST
# -----------------------------------

def main():

    print("\n")
    print("=" * 50)
    print(" GreenPrompt AI ")
    print("=" * 50)

    print(
        "\nType 'exit' to stop\n"
    )

    while True:

        query = input(
            "Ask Question: "
        )

        if query.lower() == "exit":

            print(
                "\nGoodbye!"
            )

            break

        response = (
            ask_greenprompt(
                query
            )
        )

        print(
            "\nBot Response:\n"
        )

        print(response)

        print(
            "\n" + "=" * 50
        )


if __name__ == "__main__":
    main()
