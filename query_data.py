import argparse
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM

from get_embedding_function import get_embedding_function

import os
from huggingface_hub import login

# TODO: change your api key (or add more), if you using the model requires credential (hugging face)
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "YOUR_API_KEY"
login(token=os.environ["HUGGINGFACEHUB_API_TOKEN"])

CHROMA_PATH = "chroma"
CACHE_DIR = "./model_cache"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)

def query_rag(query_text: str):
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    # context_text = "\n\n\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    # Format the chat template into a string.
    prompt = prompt_template.format(context=context_text, question=query_text)
    # print(prompt)

    # model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"  # TODO: change model
    model_name = "openai-community/gpt2"  # TODO: change model
    tokenizer = AutoTokenizer.from_pretrained(model_name, add_eos_token=True, cache_dir = CACHE_DIR)
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir = CACHE_DIR)
    # tokenizer.pad_token = tokenizer.eos_token
    # print(tokenizer.max_model_input_sizes)
    max_length = 1024 # depends on model card
    print(f"Using max length: {max_length}")
    
    # model_name = "facebook/bart-large-cnn"
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Using different model type depends on your model task
    # model = AutoModelForSeq2SeqLM.from_pretrained(model_name) 
    

    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    outputs = model.generate(**inputs, max_length=max_length, num_beams=5, early_stopping=True)
    print(len(outputs))
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    return response_text

if __name__ == "__main__":
    main()
