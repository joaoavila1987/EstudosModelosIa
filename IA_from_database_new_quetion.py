import subprocess

# --- 1. Usuario pergunta ---
pergunta = "Qual amortecedor serve na Duster 2022?"

# --- 2. Gerar embedding e buscar no Qdrant ---
vector_query = model.encode(pergunta).tolist()
resultados = qdrant.search(
    collection_name="catalogo",
    query_vector=vector_query,
    limit=3
)

# --- 3. Montar contexto com dados do catalogo ---
contexto = "\n".join([
    f"- {r.payload['descricao']} (Aplicacao: {r.payload['aplicacao']}, Codigo: {r.payload['codigo_fabricante']})"
    for r in resultados
])

prompt = f"""
Voce e um especialista em peças automotivas.

Pergunta do usuario: {pergunta}

Base de dados do catalogo (encontrada no vetor):
{contexto}

Responda somente com base nessas informacoes.
"""

# --- 4. Enviar prompt para LLaMA 3 (via Ollama) ---
resposta = subprocess.run(
    ["ollama", "run", "llama3"],
    input=prompt.encode("utf-8"),
    capture_output=True
)

# --- 5. Mostrar resposta ---
print("\n🤖 Resposta:\n", resposta.stdout.decode())
