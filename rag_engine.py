import os
import pickle
from pathlib import Path
from typing import List, Dict

import faiss
import numpy as np
from dotenv import load_dotenv
from google import genai
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from docx import Document

load_dotenv()

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
TOP_K = int(os.getenv("TOP_K", "3"))
MIN_SIMILARITY = float(os.getenv("MIN_SIMILARITY", "0.25"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "700"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))

SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".docx"}

class RAGEngine:
    def __init__(self, index_dir: Path):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.index_dir / "documents.faiss"
        self.metadata_path = self.index_dir / "metadata.pkl"
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.index = None
        self.chunks = []
        self._load_index()

    def _extract_text(self, path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix == ".txt":
            return path.read_text(encoding="utf-8", errors="ignore")
        if suffix == ".pdf":
            reader = PdfReader(str(path))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        if suffix == ".docx":
            doc = Document(str(path))
            return "\n".join(p.text for p in doc.paragraphs)
        return ""

    def _split_text(self, text: str) -> List[str]:
        words = text.split()
        if not words:
            return []
        chunks = []
        start = 0
        while start < len(words):
            end = min(start + CHUNK_SIZE, len(words))
            chunks.append(" ".join(words[start:end]))
            if end == len(words):
                break
            start = max(end - CHUNK_OVERLAP, start + 1)
        return chunks

    def build_index(self, directories: List[Path]) -> int:
        new_chunks = []
        for directory in directories:
            directory = Path(directory)
            if not directory.exists():
                continue
            for path in sorted(directory.rglob("*")):
                if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
                    text = self._extract_text(path).strip()
                    for chunk_number, chunk in enumerate(self._split_text(text), start=1):
                        new_chunks.append({
                            "text": chunk,
                            "source": path.name,
                            "chunk_id": chunk_number
                        })

        if not new_chunks:
            raise ValueError("No readable text found. Add at least one PDF, TXT, or DOCX file.")

        texts = [item["text"] for item in new_chunks]
        vectors = self.model.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True,
            show_progress_bar=True
        ).astype("float32")

        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        faiss.write_index(index, str(self.index_path))
        with self.metadata_path.open("wb") as file:
            pickle.dump(new_chunks, file)

        self.index = index
        self.chunks = new_chunks
        return len(new_chunks)

    def _load_index(self):
        if self.index_path.exists() and self.metadata_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            with self.metadata_path.open("rb") as file:
                self.chunks = pickle.load(file)

    def retrieve(self, question: str) -> List[Dict]:
        if self.index is None or not self.chunks:
            raise FileNotFoundError(
                "The document index is empty. Add documents and click 'Build / Refresh Index'."
            )
        query_vector = self.model.encode(
            [question], convert_to_numpy=True, normalize_embeddings=True
        ).astype("float32")
        scores, indices = self.index.search(query_vector, min(TOP_K, len(self.chunks)))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                item = dict(self.chunks[int(idx)])
                item["score"] = float(score)
                results.append(item)
        return results

    def answer(self, question: str) -> Dict:
        retrieved = self.retrieve(question)
        relevant = [item for item in retrieved if item["score"] >= MIN_SIMILARITY]
        if not relevant:
            return {
                "answer": "I could not find this information in the provided documents. Please ask a question related to the available documents.",
                "sources": []
            }

        context = "\n\n".join(
            f"[Source: {item['source']}, chunk {item['chunk_id']}]\n{item['text']}"
            for item in relevant
        )
        prompt = f"""You are a document-grounded question-answering assistant.
Answer the question using ONLY the context below.
Do not use outside knowledge or invent facts.
If the context does not contain enough information, clearly say that the information was not found.
Be concise and identify relevant source filenames in your answer when useful.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""

        if not GEMINI_API_KEY:
            raise ValueError(
                "Gemini API key is missing. Add GEMINI_API_KEY to your .env file."
            )

        client = genai.Client(api_key=GEMINI_API_KEY)
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )
        answer_text = (response.text or "").strip()
        if not answer_text:
            answer_text = "The model did not return a text answer. Please try again."
        return {"answer": answer_text, "sources": relevant}
