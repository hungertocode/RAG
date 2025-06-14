from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import google.generativeai as genai
from elasticsearch import Elasticsearch
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables
load_dotenv()

# Google Gemini setup
api_key = os.getenv("GOOGLE_GEMINI_API_KEY")
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-1.5-flash")

# Elasticsearch client
es_client = Elasticsearch("http://localhost:9200")
index_name = "faq-document"  

# FastAPI app instance
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request and Response Models
class QueryInput(BaseModel):
    query: str

class QueryOutput(BaseModel):
    response: str

# ElasticSearch search function
def elastic_search(query):
    search_query = {
        "size": 3,
        "query": {
            "bool": {
                "must": {
                    "multi_match": {
                        "query": query,
                        "fields": ["question^4", "text"],
                        "type": "best_fields"
                    }
                },
                # "filter": {
                #     "term": {
                #         "course": "data-engineering-zoomcamp"
                #     }
                # }
            }
        }
    }

    response = es_client.search(index=index_name, body=search_query)
    result_docs = [hit["_source"] for hit in response["hits"]["hits"]]
    return result_docs

# Prompt builder
def build_prompt(query, search_results):
    prompt_template = """
    You are an assistant for a course. Answer the following QUESTION using only the information provided in the CONTEXT from the FAQ database.
    Stay factual, concise, and clear. Do not include information that is not present in the CONTEXT.
    Write your response in natural, human-like language that is easy to understand.
    QUESTION: {question}

    CONTEXT:
    {context}
    """.strip()

    context = ""
    for info in search_results:
        context += f"section: {info['section']}\nquestion: {info['question']}\nanswer: {info['text']}\n\n"

    final_prompt = prompt_template.format(question=query, context=context.strip())
    return final_prompt

# Gemini LLM call
def llm(prompt):
    response = model.generate_content(prompt)
    return response.text

# FastAPI endpoint
@app.post("/rag", response_model=QueryOutput)
def rag_api(request: QueryInput):
    try:
        search_results = elastic_search(request.query)
        prompt = build_prompt(request.query, search_results)
        answer = llm(prompt)
        return {"response": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
