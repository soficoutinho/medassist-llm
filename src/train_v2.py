import json
import time
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer


MODEL_NAME = "Qwen/Qwen3-0.6B"

TRAIN_FILE = "data/train_v2.jsonl"
VALIDATION_FILE = "data/validation_v2.jsonl"

EXPERIMENTS = {
    "exp04": {
        "epochs": 3,
        "learning_rate": 2e-4,
        "lora_r": 16,
        "lora_alpha": 32,
    },
    "exp05": {
        "epochs": 5,
        "learning_rate": 1e-4,
        "lora_r": 16,
        "lora_alpha": 32,
    },
    "exp06": {
        "epochs": 5,
        "learning_rate": 5e-5,
        "lora_r": 32,
        "lora_alpha": 64,
    },
}


def main():
    print("=" * 70)
    print("MEDASSIST-LLM - TREINAMENTO V2")
    print("=" * 70)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA nao detectado. "
            "O treinamento V2 deve utilizar GPU."
        )

    gpu_name = torch.cuda.get_device_name(0)

    print(f"\nGPU detectada: {gpu_name}")
    print(f"CUDA PyTorch : {torch.version.cuda}")
    print(f"Modelo base  : {MODEL_NAME}")

    print("\nCarregando dataset V2...")

    dataset = load_dataset(
        "json",
        data_files={
            "train": TRAIN_FILE,
            "validation": VALIDATION_FILE,
        },
    )

    print(
        f"Treino       : "
        f"{len(dataset['train'])} exemplos"
    )
    print(
        f"Validacao    : "
        f"{len(dataset['validation'])} exemplos"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
    )

    for experiment_name, params in EXPERIMENTS.items():
        print("\n" + "=" * 70)
        print(f"INICIANDO {experiment_name.upper()}")
        print("=" * 70)

        print(
            f"Epochs        : "
            f"{params['epochs']}"
        )
        print(
            f"Learning rate : "
            f"{params['learning_rate']}"
        )
        print(
            f"LoRA r        : "
            f"{params['lora_r']}"
        )
        print(
            f"LoRA alpha    : "
            f"{params['lora_alpha']}"
        )

        output_dir = Path(
            f"experiments/{experiment_name}"
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        lora_config = LoraConfig(
            r=params["lora_r"],
            lora_alpha=params["lora_alpha"],
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

        training_args = SFTConfig(
            output_dir=str(output_dir),

            num_train_epochs=params["epochs"],

            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,

            gradient_accumulation_steps=4,

            learning_rate=params["learning_rate"],

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

        trainer = SFTTrainer(
            model=MODEL_NAME,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            processing_class=tokenizer,
            peft_config=lora_config,
        )

        print("\nParametros treinaveis:")
        trainer.model.print_trainable_parameters()

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        print(
            f"\nIniciando treinamento "
            f"{experiment_name} na GPU..."
        )

        start_time = time.time()

        train_result = trainer.train()

        training_time = (
            time.time() - start_time
        )

        print("\nExecutando avaliacao...")

        eval_metrics = trainer.evaluate()

        peak_vram = (
            torch.cuda.max_memory_allocated()
            / (1024 ** 3)
        )

        final_adapter_dir = (
            output_dir / "final_adapter"
        )

        trainer.model.save_pretrained(
            final_adapter_dir
        )

        tokenizer.save_pretrained(
            final_adapter_dir
        )

        results = {
            "experiment": experiment_name,
            "dataset_version": "v2",
            "model": MODEL_NAME,
            "gpu": gpu_name,
            "cuda": torch.version.cuda,

            "hyperparameters": {
                "epochs": params["epochs"],
                "learning_rate":
                    params["learning_rate"],
                "lora_r":
                    params["lora_r"],
                "lora_alpha":
                    params["lora_alpha"],
                "lora_dropout": 0.05,
                "batch_size": 1,
                "gradient_accumulation_steps": 4,
                "max_length": 512,
                "precision": "fp16",
            },

            "dataset": {
                "train_samples":
                    len(dataset["train"]),
                "validation_samples":
                    len(dataset["validation"]),
            },

            "training_time_seconds":
                training_time,

            "peak_gpu_memory_gb":
                peak_vram,

            "train_metrics":
                train_result.metrics,

            "eval_metrics":
                eval_metrics,
        }

        results_file = (
            output_dir / "results.json"
        )

        with open(
            results_file,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                results,
                file,
                indent=4,
                ensure_ascii=False,
            )

        print("\n" + "-" * 70)
        print(
            f"RESULTADO {experiment_name.upper()}"
        )
        print("-" * 70)

        print(
            f"Tempo treinamento : "
            f"{training_time:.2f} segundos"
        )

        print(
            f"Pico de VRAM       : "
            f"{peak_vram:.2f} GB"
        )

        if "train_loss" in train_result.metrics:
            print(
                f"Train loss         : "
                f"{train_result.metrics['train_loss']:.4f}"
            )

        if "eval_loss" in eval_metrics:
            print(
                f"Validation loss    : "
                f"{eval_metrics['eval_loss']:.4f}"
            )

        print(
            f"Adapter final      : "
            f"{final_adapter_dir}"
        )

        print(
            f"Resultados         : "
            f"{results_file}"
        )

        del trainer

        torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print("TREINAMENTO V2 FINALIZADO")
    print("=" * 70)

    print(
        f"Experimentos executados: "
        f"{len(EXPERIMENTS)}"
    )

    print(
        "\nSTATUS: TODOS OS EXPERIMENTOS V2 "
        "FORAM CONCLUIDOS."
    )

    print("=" * 70)


if __name__ == "__main__":
    main()