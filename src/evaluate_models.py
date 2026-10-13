import gc
import json
import re
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


BASE_MODEL = "Qwen/Qwen3-0.6B"

TEST_FILE = Path("data/test.jsonl")
RESULTS_DIR = Path("results")

ADAPTERS = {
    "base": None,
    "exp01": Path("experiments/exp01/final_adapter"),
    "exp02": Path("experiments/exp02/final_adapter"),
    "exp03": Path("experiments/exp03/final_adapter"),
}

MAX_NEW_TOKENS = 220


def load_test_questions():
    examples = []

    with open(TEST_FILE, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            record = json.loads(line)

            messages = record["messages"]

            question = messages[0]["content"]
            reference = messages[1]["content"]

            examples.append(
                {
                    "question": question,
                    "reference": reference,
                }
            )

    return examples


def clean_response(text):
    # Qwen3 pode produzir blocos de raciocinio <think>.
    # Eles nao devem ser apresentados como resposta final.
    text = re.sub(
        r"<think>.*?</think>",
        "",
        text,
        flags=re.DOTALL,
    )

    return text.strip()


def unload_model(model):
    del model

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_model(adapter_path=None):
    print(f"\nCarregando modelo base: {BASE_MODEL}")

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.float16,
    )

    if adapter_path is not None:
        print(f"Carregando adaptador: {adapter_path}")

        model = PeftModel.from_pretrained(
            model,
            str(adapter_path),
        )

    model = model.to("cuda")
    model.eval()

    return model


def generate_answer(model, tokenizer, question):
    messages = [
        {
            "role": "system",
            "content": (
                "Voce e o MedAssist, um assistente educacional "
                "sobre primeiros socorros. Responda em portugues, "
                "de forma clara, objetiva e segura. "
                "Nao substitua atendimento profissional."
            ),
        },
        {
            "role": "user",
            "content": question,
        },
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    ).to("cuda")

    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )

    new_tokens = generated[
        :,
        inputs["input_ids"].shape[1]:
    ]

    answer = tokenizer.batch_decode(
        new_tokens,
        skip_special_tokens=True,
    )[0]

    return clean_response(answer)


def main():
    print("=" * 70)
    print("MEDASSIST-LLM - AVALIACAO DOS MODELOS")
    print("=" * 70)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA nao detectado. A avaliacao deve utilizar GPU."
        )

    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    test_examples = load_test_questions()

    print(
        f"Perguntas de teste carregadas: "
        f"{len(test_examples)}"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
    )

    all_results = {}

    for model_name, adapter_path in ADAPTERS.items():
        print("\n" + "=" * 70)
        print(f"AVALIANDO: {model_name.upper()}")
        print("=" * 70)

        model = load_model(adapter_path)

        torch.cuda.reset_peak_memory_stats()

        start = time.time()

        model_results = []

        for index, example in enumerate(
            test_examples,
            start=1,
        ):
            print(
                f"[{model_name}] "
                f"Pergunta {index}/{len(test_examples)}"
            )

            answer = generate_answer(
                model,
                tokenizer,
                example["question"],
            )

            model_results.append(
                {
                    "id": index,
                    "question": example["question"],
                    "reference": example["reference"],
                    "answer": answer,
                }
            )

        elapsed = time.time() - start

        peak_vram = (
            torch.cuda.max_memory_allocated()
            / (1024 ** 3)
        )

        all_results[model_name] = {
            "model": BASE_MODEL,
            "adapter": (
                str(adapter_path)
                if adapter_path is not None
                else None
            ),
            "time_seconds": elapsed,
            "peak_gpu_memory_gb": peak_vram,
            "responses": model_results,
        }

        individual_file = (
            RESULTS_DIR
            / f"responses_{model_name}.json"
        )

        with open(
            individual_file,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                all_results[model_name],
                file,
                indent=4,
                ensure_ascii=False,
            )

        print(
            f"\nTempo {model_name}: "
            f"{elapsed:.2f} segundos"
        )
        print(
            f"Pico VRAM: "
            f"{peak_vram:.2f} GB"
        )

        unload_model(model)

    complete_file = (
        RESULTS_DIR
        / "all_model_responses.json"
    )

    with open(
        complete_file,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            all_results,
            file,
            indent=4,
            ensure_ascii=False,
        )

    print("\n" + "=" * 70)
    print("AVALIACAO FINALIZADA")
    print("=" * 70)

    print(
        f"Modelos avaliados : "
        f"{len(ADAPTERS)}"
    )

    print(
        f"Perguntas/modelo  : "
        f"{len(test_examples)}"
    )

    print(
        f"Total de respostas: "
        f"{len(ADAPTERS) * len(test_examples)}"
    )

    print(
        f"\nResultados salvos em: "
        f"{complete_file}"
    )

    print(
        "\nSTATUS: AVALIACAO CONCLUIDA COM SUCESSO."
    )

    print("=" * 70)


if __name__ == "__main__":
    main()