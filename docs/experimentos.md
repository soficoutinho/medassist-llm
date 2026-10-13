
# MedAssist-LLM — Relatório de Experimentos

## 1. Objetivo

Desenvolver um protótipo acadêmico de modelo de linguagem
especializado em perguntas educacionais sobre primeiros socorros,
utilizando fine-tuning local com GPU.

O projeto não é destinado a orientar emergências reais.

## 2. Modelo e infraestrutura

- Modelo base: Qwen/Qwen3-0.6B
- Técnica: Supervised Fine-Tuning (SFT) com LoRA/PEFT
- GPU: NVIDIA GeForce RTX 4050 Laptop GPU
- Framework: PyTorch, Transformers, TRL e PEFT
- Precisão: FP16
- Frontend: Gradio
- Execução: ambiente local Windows com Python

## 3. Datasets

Versão V1:
- Treinamento: 30 exemplos
- Validação: 6 exemplos
- Teste: 10 perguntas

Versão V2:
- Treinamento: 120 exemplos
- Validação: 15 exemplos
- Avaliação de regressão: 10 perguntas da V1

Os exemplos foram preparados para fins acadêmicos.

A V2 foi desenvolvida após a análise de erros nos testes da V1.
Portanto, as 10 perguntas reutilizadas não constituem um conjunto
de teste final independente para a V2.

A verificação automatizada de duplicatas considerou perguntas
idênticas após normalização, não equivalência semântica.

## 4. Experimentos

| Exp. | Dataset | Épocas | Learning rate | LoRA r | Train loss | Validation loss |
|------|---------|--------|---------------|--------|------------|-----------------|
| 01 | V1 | 1 | 0.0002 | 8 | 2.5910 | 2.3233 |
| 02 | V1 | 3 | 0.0002 | 16 | 1.9023 | 1.7720 |
| 03 | V1 | 3 | 0.0001 | 32 | 1.9314 | 1.8032 |
| 04 | V2 | 3 | 0.0002 | 16 | 1.3753 | 1.0683 |
| 05 | V2 | 5 | 0.0001 | 16 | 1.2634 | 1.1085 |
| 06 | V2 | 5 | 0.00005 | 32 | 1.3400 | 1.1795 |

As perdas de validação de V1 e V2 não são diretamente
comparáveis, pois foram medidas em conjuntos diferentes.

Os resultados completos estão em:
`results/metrics_summary.csv`

## 5. Execução na GPU

| Exp. | Tempo de treinamento | Pico de memória GPU |
|------|----------------------|--------------------|
| 01 | 66.79 s | 1.54 GB |
| 02 | 41.22 s | 1.65 GB |
| 03 | 84.52 s | 1.82 GB |
| 04 | 416.56 s | 1.61 GB |
| 05 | 824.13 s | 2.39 GB |
| 06 | 1023.77 s | 2.43 GB |

Todos os treinamentos foram executados na GPU local.

## 6. Avaliação qualitativa preliminar da V2

Foram avaliadas 10 respostas de cada experimento,
comparando o conteúdo gerado com respostas de referência.

| Experimento | Adequadas | Parciais | Falhas graves |
|-------------|-----------|----------|---------------|
| Exp04 | 2 | 5 | 3 |
| Exp05 | 1 | 4 | 5 |
| Exp06 | 0 | 5 | 5 |

Essas categorias são julgamentos qualitativos exploratórios,
não pontuações clínicas validadas.

Falhas identificadas incluem:
- Omissão de RCP e DEA em situações compatíveis com PCR.
- Orientações incorretas sobre queimaduras.
- Recomendações inadequadas para engasgo com tosse eficaz.
- Orientações perigosas relacionadas a intoxicações.
- Respostas contraditórias sobre AVC e convulsões.

## 7. Seleção do modelo

O Exp04 foi selecionado para demonstração acadêmica porque
apresentou a menor validation loss da V2 e menos falhas
graves na avaliação qualitativa preliminar.

Essa escolha não representa aprovação clínica.

## 8. Limitações

- Dataset pequeno e elaborado para fins acadêmicos.
- Ausência de validação clínica independente.
- Possibilidade de respostas incorretas e perigosas.
- Teste V2 influenciado por observações anteriores.
- Configurações de geração diferentes entre avaliações
  V1 e V2, limitando comparações diretas.
- Resultados restritos ao modelo e aos cenários avaliados.

## 9. Conclusão

O projeto demonstrou a viabilidade técnica de realizar
fine-tuning local com LoRA em uma GPU RTX 4050,
comparar hiperparâmetros e disponibilizar uma interface
web funcional com Gradio.

Entretanto, os resultados também demonstraram que
menores valores de loss não garantem respostas
clinicamente corretas ou seguras.

O MedAssist-LLM deve permanecer um protótipo
experimental exclusivamente acadêmico.
