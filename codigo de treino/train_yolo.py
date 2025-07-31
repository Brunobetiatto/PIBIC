import os
import argparse
import random
import torch
import numpy as np
from ultralytics import YOLO
import sys
import traceback

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def main(root_dir, output_dir, seed, modelo):
    set_seed(seed)

    try:
        for folder in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder)
            batch = -1

            if os.path.isdir(folder_path):
                print(f"Iniciando treinamento para o dataset: {folder} com batch = {batch}")
                data_file_path = os.path.join(folder_path, "data.yaml")

                if os.path.exists(data_file_path):
                    print(f"Treinando com batch = 2 e 100 épocas para o dataset: {folder}...")

                    model = YOLO(modelo)

                    model.train(
                        data=data_file_path,
                        epochs=100,
                        batch=batch,
                        name=f"{folder}",
                        project=output_dir,
                        device='cuda:0',
                        save=True,
                        exist_ok=True
                    )

                else:
                    print(f"⚠ Arquivo data.yaml não encontrado em {folder_path}, pulando esta pasta.")

    except KeyboardInterrupt:
        print("⚠ Treinamento interrompido manualmente (Ctrl + C).")
        sys.exit(1)

    except SystemExit:
        print("⚠ Treinamento interrompido pelo sistema.")
        sys.exit(1)

    except Exception as e:
        erro_msg = f"❌ Erro inesperado no treinamento: {str(e)}\n\n{traceback.format_exc()}"
        print(erro_msg)
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treinamento de IA por Fold.")

    parser.add_argument("root_dir", type=str, help="Diretório raiz contendo os folds de entrada.")
    parser.add_argument("output_dir", type=str, help="Diretório onde os resultados serão salvos.")
    parser.add_argument("--seed", type=int, default=42, help="Seed para reprodutibilidade (padrão: 42).")
    parser.add_argument("--model", type=str, required=True, help="Modelo da YOLO a ser utilizado.")

    args = parser.parse_args()

    main(args.root_dir, args.output_dir, args.seed, args.model)
