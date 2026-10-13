
import re

import gradio as gr
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "Qwen/Qwen3-0.6B"
ADAPTER_PATH = "experiments/exp04/final_adapter"

if not torch.cuda.is_available():
    raise RuntimeError(
        "GPU CUDA não encontrada. Este projeto exige execução na GPU."
    )

DEVICE = "cuda:0"

print("Carregando MedAssist-LLM...")
print("GPU:", torch.cuda.get_device_name(0))

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    dtype=torch.float16,
)

model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_PATH,
)

model = model.to(DEVICE)
model.eval()

SYSTEM_PROMPT = (
    "Você é o MedAssist-LLM, um protótipo acadêmico experimental "
    "sobre primeiros socorros. Suas respostas podem conter erros. "
    "Não substitua profissionais de saúde ou serviços de emergência. "
    "Não invente procedimentos e não apresente suas respostas como "
    "orientações médicas validadas."
)


def responder(pergunta):
    pergunta = pergunta.strip()

    if not pergunta:
        return "Digite uma pergunta para continuar."

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": pergunta},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    ).to(DEVICE)

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=220,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = output[0][inputs["input_ids"].shape[1]:]

    resposta = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    )

    resposta = re.sub(
        r"&lt;think&gt;.*?&lt;/think&gt;",
        "",
        resposta,
        flags=re.DOTALL,
    )

    return resposta.strip() or "O modelo não gerou uma resposta."


with gr.Blocks(title="MedAssist-LLM") as demo:
    gr.Markdown("# MedAssist-LLM")
    gr.Markdown(
        "**Protótipo acadêmico de inteligência artificial** "
        "treinado com LoRA sobre Qwen3-0.6B."
    )
    gr.Markdown(
        "⚠️ **ATENÇÃO:** Este modelo apresentou erros graves nos testes "
        "e não deve ser utilizado para orientar emergências reais. "
        "Em uma emergência no Brasil, ligue 192 (SAMU) ou 193 "
        "(Bombeiros), conforme a situação."
    )

    pergunta = gr.Textbox(
        label="Digite sua pergunta",
        placeholder="Escreva uma pergunta de teste sobre primeiros socorros...",
        lines=3,
    )

    botao = gr.Button("Gerar resposta experimental", variant="primary")

    resposta = gr.Textbox(
        label="Resposta gerada pelo modelo experimental",
        lines=8,
        interactive=False,
    )

    botao.click(
        fn=responder,
        inputs=pergunta,
        outputs=resposta,
    )

    gr.Markdown(
        "**Aviso:** As respostas são geradas automaticamente e "
        "não foram validadas clinicamente."
    )


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True,
    )
