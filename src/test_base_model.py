import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen3-0.6B"

print("=" * 70)
print("MEDASSIST-LLM - TESTE DO MODELO BASE")
print("=" * 70)

# -------------------------------------------------------
# 1. Verificacao obrigatoria da GPU
# -------------------------------------------------------

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA nao esta disponivel. "
        "Este projeto deve executar utilizando GPU."
    )

device = torch.device("cuda")

print("\n[GPU]")
print(f"GPU detectada : {torch.cuda.get_device_name(0)}")
print(f"CUDA          : {torch.version.cuda}")
print(f"Device        : {device}")

# -------------------------------------------------------
# 2. Carregamento do tokenizer
# -------------------------------------------------------

print("\n[MODELO]")
print(f"Modelo selecionado: {MODEL_NAME}")

print("\nCarregando tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# -------------------------------------------------------
# 3. Carregamento da LLM
# -------------------------------------------------------

print("Carregando modelo na GPU...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16,
)

model = model.to(device)
model.eval()

print("Modelo carregado com sucesso.")

# -------------------------------------------------------
# 4. Verificacao de onde o modelo esta executando
# -------------------------------------------------------

model_device = next(model.parameters()).device

print(f"Modelo executando em: {model_device}")

if model_device.type != "cuda":
    raise RuntimeError("O modelo nao foi carregado na GPU.")

# -------------------------------------------------------
# 5. Prompt de teste
# -------------------------------------------------------

question = (
    "O que uma pessoa leiga deve fazer inicialmente "
    "ao encontrar um adulto desacordado?"
)

messages = [
    {
        "role": "system",
        "content": (
            "Voce e um assistente educacional. "
            "Responda em portugues de forma clara e objetiva."
        ),
    },
    {
        "role": "user",
        "content": question,
    },
]

print("\n[PROMPT]")
print(question)

# -------------------------------------------------------
# 6. Aplicacao do template de chat
# -------------------------------------------------------

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

inputs = tokenizer(
    text,
    return_tensors="pt",
).to(device)

# -------------------------------------------------------
# 7. Geracao utilizando GPU
# -------------------------------------------------------

print("\nGerando resposta utilizando GPU...")

with torch.inference_mode():
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=180,
        do_sample=False,
    )

# Remove os tokens do prompt para mostrar somente
# a resposta produzida pela LLM.
new_tokens = generated_ids[:, inputs["input_ids"].shape[1]:]

response = tokenizer.batch_decode(
    new_tokens,
    skip_special_tokens=True,
)[0]

# -------------------------------------------------------
# 8. Resultado
# -------------------------------------------------------

print("\n[RESPOSTA DO MODELO BASE]")
print("-" * 70)
print(response)
print("-" * 70)

allocated = torch.cuda.memory_allocated() / (1024 ** 3)
reserved = torch.cuda.memory_reserved() / (1024 ** 3)

print("\n[USO DA GPU]")
print(f"Memoria alocada : {allocated:.2f} GB")
print(f"Memoria reservada: {reserved:.2f} GB")

print("\nTESTE FINALIZADO COM SUCESSO.")
print("=" * 70)