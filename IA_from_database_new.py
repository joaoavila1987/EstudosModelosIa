import json, sys
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
import uuid
from pathlib import Path

def load_json_safe(path):
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Arquivo não encontrado: {path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print("Erro ao decodificar JSON:")
        print(f"  Mensagem: {e.msg}")
        print(f"  Linha: {e.lineno} Coluna: {e.colno} (pos {e.pos})")
        # mostra contexto proximo ao ponto do erro
        with open(path, "r", encoding="utf-8-sig") as f:
            text = f.read()
            start = max(0, e.pos - 80)
            end = min(len(text), e.pos + 80)
            snippet = text[start:end]
        print("\nContexto próximo ao erro:\n---")
        print(snippet)
        print("\n---")
        sys.exit(1)

# carregar (substitua "catalogo.json" pelo caminho correto)
raw = load_json_safe("catalogo.json")

# aceitar tanto objeto unico quanto lista de objetos
if isinstance(raw, dict):
    items = [raw]
elif isinstance(raw, list):
    items = raw
else:
    print("Formato inesperado no JSON (esperado objeto ou lista).")
    sys.exit(1)

# --- a partir daqui segue sua lógica de embeddings / qdrant ---
model = SentenceTransformer("all-MiniLM-L6-v2")

# Exemplo de criacao/uso do Qdrant (ajuste conforme sua instalacao)
# Se estiver rodando Qdrant server local, substitua com url/porta adequados:
# qdrant = QdrantClient(url="http://localhost:6333")
# Para teste rápido em memória (se disponivel na sua versao do cliente) ajuste conforme docs.
qdrant = QdrantClient(":memory:")

qdrant.recreate_collection(
    collection_name="catalogo",
    vectors_config=VectorParams(size=384, distance="Cosine")
)

for item in items:
    aplicacoes = item.get("aplicacoes", [])
    for aplicacao in aplicacoes:
        texto = f"{item.get('descricao','')} | {aplicacao} | {item.get('fabricante','')} | {item.get('codigo_fabricante','')}"
        vector = model.encode(texto).tolist()
        qdrant.upsert(
            collection_name="catalogo",
            points=[
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload={
                        "codigo": item.get("codigo"),
                        "descricao": item.get("descricao"),
                        "aplicacao": aplicacao,
                        "fabricante": item.get("fabricante"),
                        "codigo_fabricante": item.get("codigo_fabricante"),
                        "equivalentes": item.get("equivalentes"),
                        "especificacoes": item.get("especificacoes")
                    }
                )
            ]
        )

print("Catalogo carregado no banco vetorial.")
