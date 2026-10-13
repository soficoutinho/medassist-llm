import json
from pathlib import Path


DATA_DIR = Path("data")

FILES = {
    "TREINO": DATA_DIR / "train.jsonl",
    "VALIDACAO": DATA_DIR / "validation.jsonl",
    "TESTE": DATA_DIR / "test.jsonl",
}


def load_jsonl(path):
    records = []

    with open(path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):

            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"JSON invalido em {path}, linha {line_number}: {error}"
                )

            records.append(record)

    return records


def validate_record(record, file_name, index):
    if "messages" not in record:
        raise ValueError(
            f"{file_name} - exemplo {index}: campo 'messages' ausente."
        )

    messages = record["messages"]

    if not isinstance(messages, list):
        raise ValueError(
            f"{file_name} - exemplo {index}: 'messages' deve ser uma lista."
        )

    if len(messages) != 2:
        raise ValueError(
            f"{file_name} - exemplo {index}: esperado user + assistant."
        )

    if messages[0].get("role") != "user":
        raise ValueError(
            f"{file_name} - exemplo {index}: primeira mensagem nao e user."
        )

    if messages[1].get("role") != "assistant":
        raise ValueError(
            f"{file_name} - exemplo {index}: segunda mensagem nao e assistant."
        )

    question = messages[0].get("content", "").strip()
    answer = messages[1].get("content", "").strip()

    if not question:
        raise ValueError(
            f"{file_name} - exemplo {index}: pergunta vazia."
        )

    if not answer:
        raise ValueError(
            f"{file_name} - exemplo {index}: resposta vazia."
        )

    return question


print("=" * 70)
print("MEDASSIST-LLM - VALIDACAO DO DATASET")
print("=" * 70)

datasets = {}

for name, path in FILES.items():

    print(f"\nValidando {name}: {path}")

    if not path.exists():
        raise FileNotFoundError(
            f"Arquivo nao encontrado: {path}"
        )

    records = load_jsonl(path)

    questions = []

    for index, record in enumerate(records, start=1):
        question = validate_record(
            record,
            name,
            index,
        )

        questions.append(question)

    datasets[name] = {
        "records": records,
        "questions": questions,
    }

    print(f"OK - {len(records)} exemplos validos.")


print("\n" + "=" * 70)
print("VERIFICACAO DE DUPLICIDADE ENTRE CONJUNTOS")
print("=" * 70)

train_questions = set(datasets["TREINO"]["questions"])
validation_questions = set(datasets["VALIDACAO"]["questions"])
test_questions = set(datasets["TESTE"]["questions"])

train_validation = train_questions & validation_questions
train_test = train_questions & test_questions
validation_test = validation_questions & test_questions

if train_validation:
    raise ValueError(
        "Existem perguntas identicas em TREINO e VALIDACAO."
    )

if train_test:
    raise ValueError(
        "Existem perguntas identicas em TREINO e TESTE."
    )

if validation_test:
    raise ValueError(
        "Existem perguntas identicas em VALIDACAO e TESTE."
    )

print("OK - Nenhuma pergunta identica encontrada entre os conjuntos.")


total = sum(
    len(dataset["records"])
    for dataset in datasets.values()
)

print("\n" + "=" * 70)
print("RESUMO")
print("=" * 70)

print(f"Treino     : {len(datasets['TREINO']['records'])}")
print(f"Validacao  : {len(datasets['VALIDACAO']['records'])}")
print(f"Teste      : {len(datasets['TESTE']['records'])}")
print(f"Total      : {total}")

print("\nSTATUS: DATASET VALIDADO COM SUCESSO.")
print("=" * 70)