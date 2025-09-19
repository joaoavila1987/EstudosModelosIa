import subprocess
import json

contexto = "\n".join([r.payload["descricao"] + " - " + r.payload["aplicacao"] for r in resultados])

prompt = f"""
Você é um especialista em catálogos de autopeças.
Pergunta do usuário: {pergunta}

Informações relevantes do catálogo:
{contexto}

Responda apenas com base nos dados acima.
"""

resposta = subprocess.run(
    ["ollama", "run", "llama3"],
    input=prompt.encode("utf-8"),
    capture_output=True
)

print("🤖 Resposta:", resposta.stdout.decode())
