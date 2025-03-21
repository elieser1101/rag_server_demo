import base64
import tempfile
import os

from urllib.request import urlopen
from fastapi import File, UploadFile, FastAPI, Body, Request
from pydantic import BaseModel
from app import RAG
import json


app = FastAPI()
rag = RAG()
rag.set_vector_db()

@app.post("/completion", status_code=201)
async def post_url(request: Request):
    print ("\nCOMPLETION REST API\n\n")
    body = await request.body()
    body_json = json.loads(body)
    print (body)
    print (body_json["query"])

    print (body_json)
    results = rag.app_main(body_json["query"])
    return {"rag_response": results}
