import json
import os
import asyncio
from azure.identity.aio import DefaultAzureCredential
from azure.search.documents.aio import SearchClient
from azure.search.documents.indexes.models import SearchIndex, AzureOpenAIVectorizer, AzureOpenAIVectorizerParameters
from azure.search.documents.indexes.aio import SearchIndexClient
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv

async def main():
    load_dotenv()

    credential = AzureKeyCredential(os.getenv("AZURE_SEARCH_KEY")) if os.getenv("AZURE_SEARCH_KEY") else DefaultAzureCredential()
    index_client = SearchIndexClient(endpoint=os.environ["AZURE_SEARCH_ENDPOINT"], credential=credential)
    with open("sample-index/schema.json") as f:
        index = json.load(f)
    search_index = SearchIndex.deserialize(index)
    search_index.name = os.environ["AZURE_SEARCH_INDEX"]
    search_index.vector_search.vectorizers.append(
        AzureOpenAIVectorizer(
            vectorizer_name="vectorizer",
            parameters=AzureOpenAIVectorizerParameters(
                resource_url=os.environ["AZURE_OPENAI_ENDPOINT"],
                deployment_name=os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"],
                model_name="text-embedding-3-large",
                api_key=os.environ["AZURE_OPENAI_API_KEY"]
            )
        )
    )
    search_index.vector_search.profiles[0].vectorizer_name = "vectorizer"
    await index_client.create_or_update_index(search_index)
    print("Set up index")

    search_client = SearchClient(endpoint=os.environ["AZURE_SEARCH_ENDPOINT"], index_name=os.environ["AZURE_SEARCH_INDEX"], credential=credential)
    with open("sample-index/data.json") as f:
        documents = json.load(f)
    await search_client.upload_documents(documents)
    print("Set up data")

    await search_client.close()
    await index_client.close()
    await credential.close()

if __name__ == "__main__":
    asyncio.run(main())
