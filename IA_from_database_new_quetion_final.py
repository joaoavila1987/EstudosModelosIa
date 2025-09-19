import json
import subprocess

# Configuracao do Ollama
LLAMA_MODEL = "llama3"

# 1) Carregar catalogo JSON em memoria
with open("catalogo.json", "r", encoding="utf-8") as f:
    catalogo = json.load(f)

def build_context():
    """Transforma o catalogo em um texto que o Llama pode usar."""
    linhas = []
    for item in catalogo:
        aplicacoes = ", ".join(item.get("aplicacao", []))
        linhas.append(f"{item['descricao']} (Codigo: {item['codigo_fabricante']}, Aplicacoes: {aplicacoes})")
    return "\n".join(linhas)

CATALOGO_TEXT = build_context()

def build_prompt(pergunta):
    return f"""Voce e um especialista em autopeças.

Catalogo de pecas disponiveis:
{CATALOGO_TEXT}

Pergunta do usuario: {pergunta}

Responda apenas com base no catalogo acima. 
Se nao encontrar informacao relevante, diga: "Nao encontrei essa informacao no catalogo".
"""

def ask_ollama(prompt):
    proc = subprocess.run(["ollama", "run", LLAMA_MODEL],
                          input=prompt.encode("utf-8"),
                          stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE)
    return proc.stdout.decode("utf-8")

# Loop interativo
if __name__ == "__main__":
    print("Chat iniciado. Digite 'sair' para encerrar.")
    while True:
        pergunta = input("Pergunta: ")
        if pergunta.lower() in ["sair", "exit", "quit"]:
            break
        prompt = build_prompt(pergunta)
        resposta = ask_ollama(prompt)
        print("\n--- Resposta ---")
        print(resposta)
        print("---------------\n")
