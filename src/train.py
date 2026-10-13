import argparse
import json
import os
import time

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer


MODEL_NAME = "Qwen/Qwen3-0.6B"


EXPERIMENTS = {
    "exp01": {
        "epochs": 1,
        "learning_rate": 2e-4,
        "lora_r": 8,
        "lora_alpha": 16,
    },
    "exp02": {
        "epochs": 3,
        "learning_rate": 2e-4,
        "lora_r": 16,
        "lora_alpha": 32,
    },
    "exp03": {
        "epochs": 3,
        "learning_rate": 1e-4,
        "lora_r": 32,
        "lora_alpha": 64,
    },
}


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--experiment",
        choices=EXPERIMENTS.keys(),
        default="exp01",
    )

    args = parser.parse_args()

    experiment_name = args.experiment
    config = EXPERIMENTS[experiment_name]

    output_dir = os.path.join(
        "experiments",
        experiment_name,
    )

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("MEDASSIST-LLM - FINE-TUNING COM LoRA")
    print("=" * 70)

    # --------------------------------------------------
    # GPU
    # --------------------------------------------------

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA nao foi detectado. "
            "O treinamento deve ser realizado na GPU."
        )

    gpu_name = torch.cuda.get_device_name(0)

    print("\n[GPU]")
    print(f"GPU             : {gpu_name}")
    print(f"CUDA PyTorch    : {torch.version.cuda}")
    print(
        f"VRAM total      : "
        f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
    )

    # --------------------------------------------------
    # Experimento
    # --------------------------------------------------

    print("\n[EXPERIMENTO]")
    print(f"Nome            : {experiment_name}")
    print(f"Modelo          : {MODEL_NAME}")
    print(f"Epochs          : {config['epochs']}")
    print(f"Learning rate   : {config['learning_rate']}")
    print(f"LoRA r          : {config['lora_r']}")
    print(f"LoRA alpha      : {config['lora_alpha']}")

    # --------------------------------------------------
    # Dataset
    # --------------------------------------------------

    print("\n[DATASET]")

    dataset = load_dataset(
        "json",
        data_files={
            "train": "data/train.jsonl",
            "validation": "data/validation.jsonl",
        },
    )

    print(f"Treino          : {len(dataset['train'])}")
    print(f"Validacao       : {len(dataset['validation'])}")

    # --------------------------------------------------
    # Tokenizer
    # --------------------------------------------------

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
    )

    # --------------------------------------------------
    # LoRA
    # --------------------------------------------------

    peft_config = LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    # --------------------------------------------------
    # Configuracao SFT
    # --------------------------------------------------

    training_args = SFTConfig(
        output_dir=output_dir,

        num_train_epochs=config["epochs"],
        learning_rate=config["learning_rate"],

        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,

        gradient_accumulation_steps=4,

        max_length=512,

        fp16=True,
        bf16=False,

        gradient_checkpointing=True,

        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=1,

        report_to="none",

        seed=42,

        assistant_only_loss=True,

        model_init_kwargs={
            "dtype": torch.float16,
        },
    )

    # --------------------------------------------------
    # Trainer
    # --------------------------------------------------

    trainer = SFTTrainer(
        model=MODEL_NAME,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    # --------------------------------------------------
    # Parametros treinaveis
    # --------------------------------------------------

    print("\n[PARAMETROS]")

    trainer.model.print_trainable_parameters()

    # --------------------------------------------------
    # Treinamento
    # --------------------------------------------------

    print("\n[TREINAMENTO]")
    print("Iniciando treinamento na GPU...\n")

    torch.cuda.reset_peak_memory_stats()

    start_time = time.time()

    train_result = trainer.train()

    elapsed_time = time.time() - start_time

    # --------------------------------------------------
    # Avaliacao
    # --------------------------------------------------

    print("\n[VALIDACAO]")

    eval_result = trainer.evaluate()

    # --------------------------------------------------
    # Salvar adaptador LoRA
    # --------------------------------------------------

    final_adapter_dir = os.path.join(
        output_dir,
        "final_adapter",
    )

    trainer.save_model(final_adapter_dir)
    tokenizer.save_pretrained(final_adapter_dir)

    # --------------------------------------------------
    # GPU
    # --------------------------------------------------

    peak_memory = (
        torch.cuda.max_memory_allocated()
        / (1024 ** 3)
    )

    # --------------------------------------------------
    # Resultados
    # --------------------------------------------------

    results = {
        "experiment": experiment_name,
        "model": MODEL_NAME,
        "gpu": gpu_name,
        "epochs": config["epochs"],
        "learning_rate": config["learning_rate"],
        "lora_r": config["lora_r"],
        "lora_alpha": config["lora_alpha"],
        "train_samples": len(dataset["train"]),
        "validation_samples": len(dataset["validation"]),
        "training_time_seconds": elapsed_time,
        "peak_gpu_memory_gb": peak_memory,
        "train_metrics": train_result.metrics,
        "eval_metrics": eval_result,
    }

    results_path = os.path.join(
        output_dir,
        "results.json",
    )

    with open(
        results_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=4,
            ensure_ascii=False,
        )

    print("\n" + "=" * 70)
    print("TREINAMENTO FINALIZADO")
    print("=" * 70)

    print(f"Experimento      : {experiment_name}")
    print(f"Tempo            : {elapsed_time:.2f} segundos")
    print(f"Pico VRAM        : {peak_memory:.2f} GB")

    if "train_loss" in train_result.metrics:
        print(
            f"Train loss       : "
            f"{train_result.metrics['train_loss']:.4f}"
        )

    if "eval_loss" in eval_result:
        print(
            f"Validation loss  : "
            f"{eval_result['eval_loss']:.4f}"
        )

    print(f"\nAdaptador salvo  : {final_adapter_dir}")
    print(f"Resultados       : {results_path}")

    print("\nSTATUS: EXPERIMENTO CONCLUIDO COM SUCESSO.")
    print("=" * 70)


if __name__ == "__main__":
    main()