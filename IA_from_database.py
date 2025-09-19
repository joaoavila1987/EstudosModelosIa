import json
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
import uuid

# === 1. Carregar JSON ===
with open("catalogo.json", "r", encoding="utf-8") as f:
    item = json.load(f)

# === 2. Criar embeddings ===
model = SentenceTransformer("all-MiniLM-L6-v2")  # modelo de embeddings leve

qdrant = QdrantClient(":memory:")  # em memória (pode ser servidor local)
qdrant.recreate_collection(
    collection_name="catalogo",
    vectors_config=VectorParams(size=384, distance="Cosine")
)

# Explodir aplicações em documentos separados
for aplicacao in item["aplicacoes"]:
    texto = f"{item['descricao']} | {aplicacao} | {item['fabricante']} | {item['codigo_fabricante']}"
    vector = model.encode(texto).tolist()

    qdrant.upsert(
        collection_name="catalogo",
        points=[
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "codigo": item["codigo"],
                    "descricao": item["descricao"],
                    "aplicacao": aplicacao,
                    "fabricante": item["fabricante"],
                    "codigo_fabricante": item["codigo_fabricante"],
                    "equivalentes": item["equivalentes"],
                    "especificacoes": item["especificacoes"]
                }
            )
        ]
    )

print("✅ Catalogo carregado no banco vetorial.")
