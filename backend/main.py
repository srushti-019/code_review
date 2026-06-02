from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel    

from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi
from groq import Groq
from dotenv import load_dotenv


from pdf_utils import extract_pdf_text
from embeddings import get_embedding
from db import conn, cursor


import numpy as np
import shutil
import os
import re

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv()
client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

MODELS = {
    "llama8b": "llama-3.1-8b-instant",
    "llama70b": "llama-3.3-70b-versatile",
    "gptoss20b": "openai/gpt-oss-20b",
    "gptoss120b": "openai/gpt-oss-120b"
}

reranker = CrossEncoder(
    "BAAI/bge-reranker-large"
)


bm25 = None
bm25_chunks = []


if not os.path.exists("uploads"):
    os.makedirs("uploads")

class QuestionRequest(BaseModel):
    question: str
    model_name: str = "llama70b"


def clean_text(text):
    if not text:
        return ""
    text = text.encode(
        "utf-8",
        errors="ignore"   # remove broken characters
    ).decode("utf-8")
    text = re.sub(
        r"\s+",
        " ",
        text
    )
    return text.strip()


def normalize_text(text):
    return clean_text(text).lower()

def embedding_to_pgvector(embedding):
    return "[" + ",".join(map(str, embedding)) + "]"

def build_bm25(document_id):
    global bm25
    global bm25_chunks
    cursor.execute(
        """
        SELECT
            id,
            chunk_text,
            page_start,
            page_end,
            chunk_index
        FROM document_chunks
        WHERE document_id = %s
        ORDER BY chunk_index
        """,
        (document_id,)
    )
    rows = cursor.fetchall()
    bm25_chunks = []
    tokenized = []
    for row in rows:
        chunk = {
            "id": row[0],
            "chunk_text": row[1],
            "page_start": row[2],
            "page_end": row[3],
            "chunk_index": row[4]
        }
        bm25_chunks.append(chunk)
        tokenized.append(normalize_text(row[1]).split())
    if tokenized:
        bm25 = BM25Okapi(tokenized)

def clean_citation_text(text):
    text = clean_text(text)
    if len(text) <= 450:
        return text
    text = text[:450]
    last_period = text.rfind(".")  # look for last full stop from right side
    last_space = text.rfind(" ")   # look for last space to avoid cutting words
    if last_period > 250:         # if full stop after 250 char cut text at sentence end
        text = text[:last_period + 1]
    elif last_space > 250:       # if no sentence ending(.) exists then cut at last space
        text = text[:last_space]
    return text.strip()


def cosine_similarity(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)

    denominator = (
        np.linalg.norm(v1)
        * np.linalg.norm(v2)
    )

    if denominator == 0:
        return 0.0

    return float(
        np.dot(v1, v2)
        / denominator
    )


@app.get("/")
def home():
    return {"message": "RAG Running"}


@app.get("/status")
def status():
    try:
        cursor.execute(
            """
            SELECT
                id,
                filename,
                upload_time
            FROM documents
            ORDER BY upload_time DESC
            LIMIT 1
            """
        )
        doc = cursor.fetchone()
        if not doc:
            return {
                "uploaded": False
            }
        return {
            "uploaded": True,
            "document_id": doc[0],
            "filename": doc[1],
            "upload_time": str(doc[2])
        }
    except Exception as e:
        conn.rollback()
        return {
            "error": str(e)
        }
    

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):   # UploadFile = uploaded pdf    File(...) = file is req
    try:
        file_path = f"uploads/{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(     # copy uploaded PDF into local storage
                file.file,
                buffer
            )
        cursor.execute(
            """
            INSERT INTO documents
            (
                filename
            )
            VALUES (%s)
            RETURNING id
            """,
            (file.filename,)
        )

        document_id = cursor.fetchone()[0]
        pages = extract_pdf_text(file_path)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=200
        ) 
        current_index = 0
        for page in pages:  
            page_text = clean_text(
                page["text"]
            )
            chunks = splitter.split_text(
                page_text
            )
            for chunk in chunks:
                chunk = clean_text(chunk)
                if len(chunk) < 50:      # tiny chunks are not stored
                    continue
                embedding = get_embedding(    # convert text into vector
                    chunk,
                    is_query=False
                )


                vector_str = embedding_to_pgvector(
                    embedding
                )

                cursor.execute(
                    """
                    INSERT INTO document_chunks
                    (
                        document_id,
                        chunk_text,
                        embedding,
                        page_start,
                        page_end,
                        chunk_index
                    )
                    VALUES
                    (
                        %s,
                        %s,
                        %s::vector,
                        %s,
                        %s,
                        %s
                    )
                    """,
                    (
                        document_id,
                        chunk,
                        vector_str,
                        page["page"],
                        page["page"],
                        current_index
                    )
                )


                current_index += 1
        conn.commit()
        build_bm25(document_id)
        return {
            "message": "PDF uploaded successfully",
            "filename": file.filename
        }
    except Exception as e:
        conn.rollback()
        return {
            "error": str(e)
        }


def vector_search(question, document_id, top_k=25):
    query_embedding = get_embedding(question, is_query=True)
    query_vector = embedding_to_pgvector(query_embedding)
    cursor.execute(
        """
        SELECT
            id,
            chunk_text,
            page_start,
            page_end,
            chunk_index,
            1 - (embedding <=> %s::vector) AS similarity
        FROM document_chunks
        WHERE document_id = %s
        ORDER BY embedding <=> %s::vector
        LIMIT %s
        """,    
        (
            query_vector,
            document_id,
            query_vector,
            top_k
        )
    )

    rows = cursor.fetchall()
    results = []
    for row in rows:
        results.append({
            "id": row[0],
            "chunk_text": row[1],
            "page_start": row[2],
            "page_end": row[3],
            "chunk_index": row[4],
            "vector_similarity": float(row[5])
        })
    return results


def bm25_search(question, top_k=25):
    global bm25
    if bm25 is None:
        return []
    tokens = normalize_text(question).split()
    scores = bm25.get_scores(tokens)        # compare query tokens against every chunk
    ranked_indices = np.argsort(scores)[::-1]  # sort chunk index by score (desc)
    results = []

    for idx in ranked_indices[:top_k]:
        chunk = bm25_chunks[idx]  # bm25_chunks = stores chunk data
        results.append({
            "id": chunk["id"],
            "chunk_text": chunk["chunk_text"],
            "page_start": chunk["page_start"],
            "page_end": chunk["page_end"],
            "chunk_index": chunk["chunk_index"],
            "bm25_similarity": float(scores[idx])
        })
    return results


def merge_results(vector_results, bm25_results):
    merged = {}
    for item in vector_results:     
        merged[item["id"]] = item
    for item in bm25_results:
        if item["id"] not in merged:    # chunk not in vector results
            merged[item["id"]] = item
        else:                          # chunk already in vector results
            merged[item["id"]]["bm25_similarity"] = item["bm25_similarity"]

    for item in merged.values():  
        item.setdefault(
            "vector_similarity",
            0.0
        )
        item.setdefault(
            "bm25_similarity",
            0.0
        )
    return list(merged.values())


def rerank_results(question, results):

    if not results:
        return []

    pairs = [
        [question, r["chunk_text"]]
        for r in results
    ]

    scores = reranker.predict(pairs)
    max_bm25 = max(
        [r.get("bm25_similarity", 0.0) for r in results],
        default=1.0
    )

    # avoid divide-by-zero
    if max_bm25 == 0:
        max_bm25 = 1.0

    for i in range(len(results)):
        rerank_score = float(scores[i])
        bm25_score = results[i].get(
            "bm25_similarity",
            0.0
        )
        normalized_bm25 = (
            bm25_score / max_bm25
        )

        results[i]["rerank_score"] = rerank_score
        results[i]["normalized_bm25"] = normalized_bm25
        results[i]["final_score"] = (
            0.80 * rerank_score
            +
            0.15 * results[i]["vector_similarity"]
            +
            0.05 * normalized_bm25
        )

    results = sorted(
        results,
        key=lambda x: x["final_score"],
        reverse=True
    )
    return results


def generate_answer(
    question,
    chunks,
    model_name,
    previous_answers=""
):

    context = "\n\n".join([
        c["chunk_text"]
        for c in chunks
    ])
    is_summary = any(
        word in question.lower()
        for word in [
            "summary",
            "summarize",
            "overview",
            "brief"
        ]
    )

    if is_summary:
        instruction = """
Summarize the ENTIRE document in a structured way.
Do not focus on individual sections.
"""
    else:
        instruction = """
Answer strictly from the document.
"""
    prompt = f"""
You are a strict RAG system.

{instruction}

QUESTION:
{question}

DOCUMENT:
{context}

Rules:
1. Use ONLY document context
2. No hallucination
3. Give only the answer itself
4. If answer not found then say:
Information not found in document.
"""

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=500,
        temperature=0.5
    )
    answer = response.choices[0].message.content
    return clean_text(answer)


def compute_answer_score(chunks):  # avg relevance score
    return round(
        float(
            sum([
                c["final_score"]
                for c in chunks
            ]) / len(chunks)
        ),
        4
    )


def get_answer_citations(
    answer,
    chunks,
    top_n=3
):

    if not chunks:
        return []

    pairs = [
        [answer, chunk["chunk_text"]]
        for chunk in chunks
    ]

    scores = reranker.predict(
        pairs
    )

    scored_chunks = []

    for chunk, score in zip(
        chunks,
        scores
    ):

        scored_chunks.append(
            (
                float(score),
                chunk
            )
        )

    scored_chunks.sort(
        key=lambda x: x[0],
        reverse=True
    )

    citations = []

    for rank, (score, chunk) in enumerate(
        scored_chunks[:top_n],
        start=1
    ):

        citations.append({

            "citation_rank":
            rank,

            "page_range":
            str(chunk["page_start"]),

            "matched_text":
            clean_citation_text(
                chunk["chunk_text"]
            ),

            "score":
            round(score, 4)
        })

    return citations



@app.post("/query")
async def query(data: QuestionRequest):

    try:

        cursor.execute(
            """
            SELECT id
            FROM documents
            ORDER BY upload_time DESC
            LIMIT 1
            """
        )

        latest_doc = cursor.fetchone()

        if not latest_doc:
            return {
                "error": "No document uploaded"
            }

        document_id = latest_doc[0]

        question = clean_text(
            data.question
        )

        selected_model = MODELS.get(
            data.model_name,
            "llama-3.3-70b-versatile"
        )

        vector_results = vector_search(
            question,
            document_id
        )

        bm25_results = bm25_search(
            question
        )

        merged = merge_results(
            vector_results,
            bm25_results
        )

        reranked = rerank_results(
            question,
            merged
        )

        if not reranked:

            return {
                "question": question,
                "model_used": selected_model,
                "top_k_answers": []
            }

        windows = []

        if len(reranked) > 0:
            windows.append(
                reranked[:5]
            )

        if len(reranked) > 5:
            windows.append(
                reranked[5:10]
            )

        if len(reranked) > 10:
            windows.append(
                reranked[10:15]
            )

        candidate_answers = []

        used_answers = set()

        previous_answers = ""

        for rank, window in enumerate(
            windows,
            start=1
        ):

            answer = generate_answer(
                question,
                window,
                selected_model,
                previous_answers
            )

            answer_clean = normalize_text(
                answer
            )

            if answer_clean in used_answers:
                continue

            used_answers.add(
                answer_clean
            )

            previous_answers += (
                "\n" + answer
            )

            retrieval_score = round(
                sum(
                    c["final_score"]
                    for c in window
                ) / len(window),
                4
            )

            if (
                "information not found"
                in answer_clean
            ):

                candidate_answers.append({

                    "answer_rank":
                    rank,

                    "retrieval_score":
                    retrieval_score,

                    "answer":
                    answer,

                    "citation_ranges":
                    [],

                    "citations":
                    []
                })

                continue

            citations = get_answer_citations(
                answer,
                window,
                top_n=3
            )

            candidate_answers.append({

                "answer_rank":
                rank,

                "retrieval_score":
                retrieval_score,

                "answer":
                answer,

                "citation_ranges":
                sorted(
                    list(
                        set(
                            c["page_range"]
                            for c in citations
                        )
                    ),
                    key=int
                ),

                "citations":
                citations
            })

        filtered = []

        for ans in candidate_answers:

            text = normalize_text(
                ans["answer"]
            )

            if (
                len(text) < 15
                and
                "information not found"
                not in text
            ):
                continue

            filtered.append(
                ans
            )

        filtered = sorted(

            filtered,

            key=lambda x:
            x["retrieval_score"],

            reverse=True
        )

        final_answers = []

        for i, ans in enumerate(
            filtered[:3],
            start=1
        ):

            final_answers.append({

                "answer_rank":
                i,

                "retrieval_score":
                ans["retrieval_score"],

                "answer":
                ans["answer"],

                "citation_ranges":
                ans["citation_ranges"],

                "citations":
                ans["citations"]
            })

        return {

            "question":
            question,

            "model_used":
            selected_model,

            "top_k_answers":
            final_answers
        }

    except Exception as e:

        conn.rollback()

        return {
            "error": str(e)
        }