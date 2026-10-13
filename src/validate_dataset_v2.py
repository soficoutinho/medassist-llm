import json
from pathlib import Path


FILES = {
    "TREINO V2": Path("data/train_v2.jsonl"),
    "VALIDACAO V2": Path("data/validation_v2.jsonl"),
    "TESTE": Path("data/test.jsonl"),
}

EXPECTED_COUNTS = {
    "TREINO V2": 120,
    "VALIDACAO V2": 15,
    "TESTE": 10,
}


def normalize_text(text):
    return " ".join(
        text.lower().strip().split()
    )


def load_and_validate(name, path):
    errors = []
    examples = []

    if not path.exists():
        return [], [
            f"{name}: arquivo nao encontrado: {path}"
        ]

    with open(path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                errors.append(
                    f"{name} - linha {line_number}: "
                    f"linha vazia."
                )
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                errors.append(
                    f"{name} - linha {line_number}: "
                    f"JSON invalido -> {error}"
                )
                continue

            if "messages" not in record:
                errors.append(
                    f"{name} - linha {line_number}: "
                    f"campo 'messages' ausente."
                )
                continue

            messages = record["messages"]

            if not isinstance(messages, list):
                errors.append(
                    f"{name} - linha {line_number}: "
                    f"'messages' precisa ser uma lista."
                )
                continue

            if len(messages) != 2:
                errors.append(
                    f"{name} - linha {line_number}: "
                    f"esperados exatamente 2 messages."
                )
                continue

            user_message = messages[0]
            assistant_message = messages[1]

            if user_message.get("role") != "user":
                errors.append(
                    f"{name} - linha {line_number}: "
                    f"primeiro role deve ser 'user'."
                )

            if assistant_message.get("role") != "assistant":
                errors.append(
                    f"{name} - linha {line_number}: "
                    f"segundo role deve ser 'assistant'."
                )

            question = user_message.get("content", "")
            answer = assistant_message.get("content", "")

            if not isinstance(question, str) or not question.strip():
                errors.append(
                    f"{name} - linha {line_number}: "
                    f"pergunta vazia ou invalida."
                )

            if not isinstance(answer, str) or not answer.strip():
                errors.append(
                    f"{name} - linha {line_number}: "
                    f"resposta vazia ou invalida."
                )

            if (
                isinstance(question, str)
                and question.strip()
                and isinstance(answer, str)
                and answer.strip()
            ):
                examples.append(
                    {
                        "line": line_number,
                        "question": question,
                        "normalized_question":
                            normalize_text(question),
                        "answer": answer,
                    }
                )

    return examples, errors


def find_duplicates(name, examples):
    errors = []
    seen = {}

    for example in examples:
        question = example["normalized_question"]

        if question in seen:
            errors.append(
                f"{name}: pergunta duplicada nas linhas "
                f"{seen[question]} e {example['line']}."
            )
        else:
            seen[question] = example["line"]

    return errors


def find_leakage(
    source_name,
    source_examples,
    target_name,
    target_examples,
):
    errors = []

    target_questions = {
        example["normalized_question"]:
        example["line"]
        for example in target_examples
    }

    for example in source_examples:
        question = example["normalized_question"]

        if question in target_questions:
            errors.append(
                f"VAZAMENTO: pergunta de {source_name} "
                f"(linha {example['line']}) tambem aparece "
                f"em {target_name} "
                f"(linha {target_questions[question]})."
            )

    return errors


def main():
    print("=" * 72)
    print("MEDASSIST-LLM - VALIDACAO DO DATASET V2")
    print("=" * 72)

    datasets = {}
    all_errors = []

    for name, path in FILES.items():
        examples, errors = load_and_validate(
            name,
            path,
        )

        datasets[name] = examples
        all_errors.extend(errors)

        print(
            f"\n{name:<15}: "
            f"{len(examples)} exemplos validos"
        )

        expected = EXPECTED_COUNTS[name]

        if len(examples) != expected:
            all_errors.append(
                f"{name}: esperado {expected} exemplos, "
                f"mas foram encontrados "
                f"{len(examples)} validos."
            )

    print("\nVerificando duplicatas...")

    for name, examples in datasets.items():
        all_errors.extend(
            find_duplicates(name, examples)
        )

    print("Verificando vazamento entre treino e validacao...")

    all_errors.extend(
        find_leakage(
            "TREINO V2",
            datasets["TREINO V2"],
            "VALIDACAO V2",
            datasets["VALIDACAO V2"],
        )
    )

    print("Verificando vazamento do teste no treino...")

    all_errors.extend(
        find_leakage(
            "TESTE",
            datasets["TESTE"],
            "TREINO V2",
            datasets["TREINO V2"],
        )
    )

    print("Verificando vazamento do teste na validacao...")

    all_errors.extend(
        find_leakage(
            "TESTE",
            datasets["TESTE"],
            "VALIDACAO V2",
            datasets["VALIDACAO V2"],
        )
    )

    print("\n" + "=" * 72)

    if all_errors:
        print("PROBLEMAS ENCONTRADOS:")
        print("=" * 72)

        for error in all_errors:
            print(f"- {error}")

        print(
            f"\nTOTAL DE PROBLEMAS: "
            f"{len(all_errors)}"
        )

        print(
            "\nSTATUS: DATASET V2 NAO FOI APROVADO."
        )

    else:
        total = sum(
            len(examples)
            for examples in datasets.values()
        )

        print("RESULTADO DA VALIDACAO")
        print("=" * 72)

        print(
            f"Treino V2     : "
            f"{len(datasets['TREINO V2'])}"
        )

        print(
            f"Validacao V2  : "
            f"{len(datasets['VALIDACAO V2'])}"
        )

        print(
            f"Teste         : "
            f"{len(datasets['TESTE'])}"
        )

        print(
            f"Total         : "
            f"{total}"
        )

        print("\nJSON                 : OK")
        print("Estrutura messages   : OK")
        print("Perguntas/respostas  : OK")
        print("Duplicatas exatas    : NAO ENCONTRADAS")
        print("Treino x validacao   : SEM VAZAMENTO EXATO")
        print("Teste x treino       : SEM VAZAMENTO EXATO")
        print("Teste x validacao    : SEM VAZAMENTO EXATO")

        print(
            "\nSTATUS: DATASET V2 VALIDADO COM SUCESSO."
        )

    print("=" * 72)


if __name__ == "__main__":
    main()