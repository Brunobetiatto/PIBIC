import os
import matplotlib.pyplot as plt

def parse_metrics_from_txt(file_path):
    """
    Lê um arquivo TXT e extrai os valores de Acurácia e mAP@50.
    Espera que o arquivo tenha linhas do tipo:
        Acurácia: 0.8095
        mAP@50: 0.9001
    Retorna uma tupla: (acuracia, map50)
    """
    acuracia = None
    map50 = None
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith("Acurácia:"):
                try:
                    acuracia = float(line.split(":", 1)[1].strip())
                except Exception as e:
                    print(f"Erro ao ler Acurácia em {file_path}: {e}")
            elif line.startswith("mAP@50:"):
                try:
                    map50 = float(line.split(":", 1)[1].strip())
                except Exception as e:
                    print(f"Erro ao ler mAP@50 em {file_path}: {e}")
    return acuracia, map50

def process_experiment(experiment_path):
    """
    Dado o caminho de um experimento (pasta), busca pelas 5 subpastas 
    (resultado_1, resultado_2, resultado_3, resultado_4, resultado_5), 
    lê os arquivos TXT e retorna as médias de Acurácia e mAP@50.
    """
    subfolder_names = ["resultado_fold_1", "resultado_fold_2", "resultado_fold_3", "resultado_fold_4", "resultado_fold_5"]
    acuracias = []
    map50s = []
    
    for sub in subfolder_names:
        sub_path = os.path.join(experiment_path, sub)
        if not os.path.isdir(sub_path):
            print(f"Subpasta {sub} não encontrada em {experiment_path}.")
            continue
        
        txt_files = [f for f in os.listdir(sub_path) if f.endswith(".txt")]
        if not txt_files:
            print(f"Nenhum arquivo TXT encontrado em {sub_path}.")
            continue
        
        txt_path = os.path.join(sub_path, txt_files[0])
        ac, m50 = parse_metrics_from_txt(txt_path)
        if ac is not None:
            acuracias.append(ac)
        if m50 is not None:
            map50s.append(m50)
    
    avg_acuracia = sum(acuracias) / len(acuracias) if acuracias else None
    avg_map50 = sum(map50s) / len(map50s) if map50s else None
    return avg_acuracia, avg_map50

def main():
    # Configuração: defina aqui os experimentos que deseja comparar.
    # Cada item do dicionário tem as chaves:
    #   - "path": caminho completo da pasta do experimento
    #   - "label": nome a ser exibido no gráfico
    experiments = [
        {"path": "/home/bruno/projects/teste_somente_cachorro_limpo", "label": "somente cachorro (300+)"},
        {"path": "/home/bruno/projects/teste_somente_gato_limpo", "label": "somente gato"}
    ]
    
    labels = []
    acuracia_values = []
    map50_values = []
    
    for exp in experiments:
        exp_path = exp["path"]
        exp_label = exp["label"]
        avg_acuracia, avg_map50 = process_experiment(exp_path)
        if avg_acuracia is None or avg_map50 is None:
            print(f"Não foi possível calcular as métricas para {exp_label} (pasta: {exp_path}).")
            continue
        labels.append(exp_label)
        acuracia_values.append(avg_acuracia)
        map50_values.append(avg_map50)
        print(f"{exp_label} -> Acurácia média: {avg_acuracia:.4f} | mAP@50 média: {avg_map50:.4f}")
    
    # Cria a pasta de saída para os gráficos, se não existir
    output_folder = "acuracia_graphic_cachorro_gato_limpo"
    os.makedirs(output_folder, exist_ok=True)
    
    # Gera o gráfico de Acurácia
    plt.figure(figsize=(10, 6), dpi=300)
    plt.bar(labels, acuracia_values, color='skyblue')
    plt.xlabel("Experimentos")
    plt.ylabel("Acurácia")
    plt.title("Comparação de Acurácia entre Experimentos")
    plt.ylim(0, 1)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.xticks(rotation=45, ha="right")  # Rotaciona os rótulos do eixo X para melhor visualização
    plt.tight_layout()
    acuracia_output = os.path.join(output_folder, "acuracia_comparison.png")
    plt.savefig(acuracia_output)
    plt.close()
    print(f"Gráfico de Acurácia salvo em: {acuracia_output}")
    
    # Gera o gráfico de mAP@50
    plt.figure(figsize=(10, 6), dpi=300)
    plt.bar(labels, map50_values, color='salmon')
    plt.xlabel("Experimentos")
    plt.ylabel("mAP@50")
    plt.title("Comparação de mAP@50 entre Experimentos")
    plt.ylim(0, 1)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.xticks(rotation=45, ha="right")  # Rotaciona os rótulos do eixo X
    plt.tight_layout()
    map50_output = os.path.join(output_folder, "map50_comparison.png")
    plt.savefig(map50_output)
    plt.close()
    print(f"Gráfico de mAP@50 salvo em: {map50_output}")

if __name__ == "__main__":
    main()

