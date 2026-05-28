import httpx
from langchain_openai import ChatOpenAI

client = httpx.Client(
    verify=False
)

llm = ChatOpenAI(

    base_url="https://genailab.tcs.in",

    model="azure_ai/genailab-maas-DeepSeek-V3-0324",

    api_key="YOUR_KEY",

    http_client=client
)

response = llm.invoke(
    "Hello"
)

print(response.content)
