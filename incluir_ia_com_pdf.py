# ========================================
# RAG LOCAL (PDFs + FAISS + Llama3 via Ollama)
# ========================================

import os
import fitz  # PyMuPDF -> pip install pymupdf
import faiss  # pip install faiss-cpu
import numpy as np
import subprocess
from sentence_transformers import SentenceTransformer
from pathlib import Path


# 1) Extrair texto dos PDFs
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    texts = []
    for i, page in enumerate(doc):
        text = page.get_text("text")
        if text.strip():
            texts.append((i+1, text))  
    return texts


# 2) Dividir em chunks
def chunk_text(text, page, max_chars=1000, overlap=200):
    chunks = []
    i = 0
    while i < len(text):
        end = i + max_chars
        chunk = text[i:end]
        chunks.append({"page": page, "text": chunk})
        i = end - overlap
    return chunks


# 3) Classe RAG
class PDFRAG:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        print("Carregando modelo de embeddings...")
        self.embedder = SentenceTransformer(model_name)
        self.index = None
        self.chunks = []

    def build_index(self, pdf_folder="pdfs"):
        all_chunks = []
        for pdf_file in Path(pdf_folder).glob("*.pdf"):
            print(f"Extraindo texto de: {pdf_file}")
            texts = extract_text_from_pdf(str(pdf_file))
            for page, text in texts:
                chunks = chunk_text(text, page)
                all_chunks.extend(chunks)

        # embeddings
        texts_only = [c["text"] for c in all_chunks]
        vectors = self.embedder.encode(texts_only, show_progress_bar=True)
        vectors = np.array(vectors).astype("float32")

        # FAISS
        d = vectors.shape[1]
        self.index = faiss.IndexFlatL2(d)
        self.index.add(vectors)

        # salvar chunks
        self.chunks = all_chunks
        print(f"Index criado com {len(self.chunks)} chunks.")

    def search(self, query, top_k=5):
        q_vec = self.embedder.encode([query]).astype("float32")
        distances, indices = self.index.search(q_vec, top_k)

        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.chunks):
                chunk = self.chunks[idx]
                results.append({"page": chunk["page"], "text": chunk["text"], "score": float(dist)})
        return results


# 4) Conectar no Ollama
def ask_ollama(prompt, model="llama3"):
    cmd = ["ollama", "run", model]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    out, _ = proc.communicate(prompt)
    return out


# 5) Construir prompt com contexto
def build_prompt(query, results):
    context = ""
    for r in results:
        context += f"[Página {r['page']}]\n{r['text']}\n\n"

    prompt = f"""
Você é um assistente técnico que responde baseado apenas no conteúdo abaixo dos catálogos:

{context}

Pergunta: {query}

Responda de forma clara, cite a página do catálogo quando possível.
Se a resposta não estiver nos catálogos, diga "Não encontrei essa informação no catálogo".
"""
    return prompt


# 6) Execu��o principal
if __name__ == "__main__":
    rag = PDFRAG()
    rag.build_index(pdf_folder="pdfs")  # coloque seus PDFs na pasta "pdfs"

    while True:
        q = input("\nPergunta: ")
        if q.lower() in ["sair", "exit", "quit"]:
            break

        results = rag.search(q, top_k=3)
        prompt = build_prompt(q, results)
        resposta = ask_ollama(prompt, model="llama3")

        print("\n--- Resposta ---")
        print(resposta)
