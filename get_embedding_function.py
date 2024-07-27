# from langchain_community.embeddings.ollama import OllamaEmbeddings
# from langchain_community.embeddings.bedrock import BedrockEmbeddings
from transformers import AutoTokenizer, AutoModel
import torch
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

def get_embedding_function():
    # embeddings = BedrockEmbeddings(
    #     credentials_profile_name="default", region_name="us-east-1"
    # )
    # embeddings = OllamaEmbeddings(model="nomic-embed-text")
    embeddings = HuggingFaceBgeEmbeddings(
                                model_name='sentence-transformers/all-MiniLM-L6-v2',
                                model_kwargs={'device':'cpu'}, 
                                encode_kwargs={'normalize_embeddings': True})
    return embeddings
