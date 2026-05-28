from dotenv import load_dotenv
import os

from openai import OpenAI

from rag import (
    retrieve_context,
    is_allowed_query,
    rejection_message,
    SYSTEM_PROMPT
)

load_dotenv()

api_key = os.getenv(
    "OPENAI_API_KEY"
)

client = OpenAI(
    api_key=api_key
)


def ask_greenprompt(query):

    # -------------------------
    # Guardrail Check
    # -------------------------

    if not is_allowed_query(query):

        return rejection_message()

    # -------------------------
    # Retrieve Context
    # -------------------------

    context = retrieve_context(query)

    # -------------------------
    # Prompt Builder
    # -------------------------

    final_prompt = f"""
Retrieved Context:

{context}

User Question:

{query}

Generate an enterprise-grade answer.
"""

    try:

        response = (
            client.chat.completions.create(
                model="gpt-4o-mini",

                messages=[

                    {
                        "role": "system",
                        "content":
                        SYSTEM_PROMPT
                    },

                    {
                        "role": "user",
                        "content":
                        final_prompt
                    }

                ],

                temperature=0.3
            )
        )

        answer = (
            response
            .choices[0]
            .message.content
        )

        return answer

    except Exception as e:

        return (
            f"LLM Error: {str(e)}"
        )


# --------------------------------
# TERMINAL CHATBOT TEST
# --------------------------------

def main():

    print("\n")
    print("=" * 50)
    print(" GreenPrompt AI Chatbot ")
    print("=" * 50)

    print(
        "\nType 'exit' to quit.\n"
    )

    while True:

        query = input(
            "\nAsk Question: "
        )

        if query.lower() == "exit":

            print("\nGoodbye!")
            break

        response = (
            ask_greenprompt(query)
        )

        print("\nBot Response:\n")
        print(response)


if __name__ == "__main__":
    main()
