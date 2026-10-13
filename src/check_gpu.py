import platform
import torch

print("=" * 65)
print("      MEDASSIST-LLM - VERIFICACAO DO AMBIENTE DE GPU")
print("=" * 65)

print("\n[AMBIENTE]")
print(f"Sistema operacional : {platform.system()} {platform.release()}")
print(f"Python              : {platform.python_version()}")
print(f"PyTorch             : {torch.__version__}")
print(f"CUDA PyTorch        : {torch.version.cuda}")

print("\n[GPU]")
print(f"CUDA disponivel     : {torch.cuda.is_available()}")

if torch.cuda.is_available():
    gpu_id = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(gpu_id)

    print(f"GPU detectada       : {torch.cuda.get_device_name(gpu_id)}")
    print(f"Quantidade de GPUs  : {torch.cuda.device_count()}")
    print(f"Memoria total       : {properties.total_memory / (1024**3):.2f} GB")
    print(
        f"Compute Capability  : "
        f"{properties.major}.{properties.minor}"
    )

    print("\n[TESTE COMPUTACIONAL NA GPU]")

    device = torch.device("cuda")

    # Matrizes pequenas para realizar um teste seguro da GPU.
    a = torch.rand((2000, 2000), device=device)
    b = torch.rand((2000, 2000), device=device)

    torch.cuda.synchronize()

    c = torch.matmul(a, b)

    torch.cuda.synchronize()

    print(f"Tensor A             : {a.device}")
    print(f"Tensor B             : {b.device}")
    print(f"Resultado            : {c.device}")
    print(f"Dimensao resultado   : {c.shape}")
    print("\nSTATUS: GPU CONFIGURADA E FUNCIONANDO CORRETAMENTE.")

else:
    print("\nSTATUS: CUDA NAO FOI DETECTADO.")
    print("O projeto nao deve prosseguir com treinamento em CPU.")

print("\n" + "=" * 65)