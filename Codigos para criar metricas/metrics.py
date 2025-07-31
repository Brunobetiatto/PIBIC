import os
import csv
import argparse
import matplotlib.pyplot as plt
import math

def parse_metrics_file(filepath):
    metrics = {}
    category_counts = {}

    with open(filepath, 'r') as file:
        lines = file.readlines()

    for i, line in enumerate(lines):
        line = line.strip()

        if line.startswith("mAP50:"):
            metrics["mAP50"] = float(line.split(": ")[1])
        elif line.startswith("mAP50-95:"):
            metrics["mAP50_95"] = float(line.split(": ")[1])
        elif line.startswith("Acurácia:"):
            metrics["accuracy"] = float(line.split(": ")[1])
        elif line.startswith("Precisão:"):
            metrics["precision"] = float(line.split(": ")[1])
        elif line.startswith("Recall:"):
            metrics["recall"] = float(line.split(": ")[1])
        elif line.startswith("F1-Score:"):
            metrics["f1"] = float(line.split(": ")[1])
        elif "Corretos" in line:
            category, values = line.split(": ")
            correct, total = map(int, values.split(" ")[0].split("/"))
            category_name = category.replace(" Corretos", "")
            category_counts[category_name] = (correct, total)

    return metrics, category_counts

def process_metrics_folder(base_folder, output_csv):
    fold_data = []  # Renomeado para armazenar dados dos folds
    folder_count = 0

    sums = {
        'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0,
        'mAP50': 0, 'mAP50_95': 0
    }

    # Listas para armazenar métricas individuais para cálculo do desvio padrão
    all_accuracy = []
    all_precision = []
    all_recall = []
    all_f1 = []
    all_mAP50 = []
    all_mAP50_95 = []

    category_sums = {
        'Felinos': [0, 0],
        'Caninos': [0, 0],
        'Convexo': [0, 0],
        'Linear': [0, 0]
    }

    for subfolder in os.listdir(base_folder):
        subfolder_path = os.path.join(base_folder, subfolder)
        metrics_path = os.path.join(subfolder_path, 'metrics.txt')

        if os.path.isdir(subfolder_path) and os.path.exists(metrics_path):
            metrics, category_counts = parse_metrics_file(metrics_path)

            # Coletando métricas individuais
            acc = metrics.get('accuracy', 0)
            prec = metrics.get('precision', 0)
            rec = metrics.get('recall', 0)
            f1 = metrics.get('f1', 0)
            map50 = metrics.get('mAP50', 0)
            map50_95 = metrics.get('mAP50_95', 0)

            # Armazenando para desvio padrão
            all_accuracy.append(acc)
            all_precision.append(prec)
            all_recall.append(rec)
            all_f1.append(f1)
            all_mAP50.append(map50)
            all_mAP50_95.append(map50_95)

            # Atualizando somas
            sums['accuracy'] += acc
            sums['precision'] += prec
            sums['recall'] += rec
            sums['f1'] += f1
            sums['mAP50'] += map50
            sums['mAP50_95'] += map50_95

            for cat in category_sums:
                correct, total = category_counts.get(cat, (0, 0))
                category_sums[cat][0] += correct
                category_sums[cat][1] += total

            folder_count += 1

            fold_data.append([
                subfolder,
                acc,
                prec,
                rec,
                f1,
                map50,
                map50_95,
            ])

    def safe_avg(value): 
        return value / folder_count if folder_count > 0 else 0

    averages = [safe_avg(sums[k]) for k in ['accuracy', 'precision', 'recall', 'f1', 'mAP50', 'mAP50_95']]

    # Função para calcular desvio padrão
    def calculate_std(values):
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return math.sqrt(variance)

    # Calculando desvios padrão
    stds = [
        calculate_std(all_accuracy),
        calculate_std(all_precision),
        calculate_std(all_recall),
        calculate_std(all_f1),
        calculate_std(all_mAP50),
        calculate_std(all_mAP50_95)
    ]

    # Gravar CSV
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            'Subpasta', 'Acurácia', 'Precision', 'Recall', 'F1 Score', 'mAP@50', 'mAP@50-95'
        ])
        writer.writerows(fold_data)
        writer.writerow(["Média Geral"] + averages)
        writer.writerow(["Desvio Padrão"] + stds)  # Nova linha para desvio padrão

    print(f"CSV com resultados salvos em {output_csv}")

    # Gráfico: Felinos vs Caninos (Melhorado)
    felinos = category_sums['Felinos']
    caninos = category_sums['Caninos']
    felinos_pct = (felinos[0] / felinos[1]) * 100 if felinos[1] > 0 else 0
    caninos_pct = (caninos[0] / caninos[1]) * 100 if caninos[1] > 0 else 0

    plt.figure(figsize=(10, 6))
    bars = plt.bar(
        [f'Felinos\n({felinos[0]}/{felinos[1]})', f'Caninos\n({caninos[0]}/{caninos[1]})'],
        [felinos_pct, caninos_pct],
        color=['#FFA500', '#1E90FF'],
        edgecolor='black',
        width=0.6
    )

    plt.ylabel('Correção (%)', fontsize=12, labelpad=15)
    plt.title('Desempenho por Categoria: Felinos vs Caninos\n', fontsize=14, fontweight='bold')
    plt.ylim(0, 110)
    plt.yticks(range(0, 101, 10), fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Adiciona valores percentuais
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%',
                ha='center', va='bottom',
                fontsize=12,
                fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(base_folder, 'grafico_felinos_caninos.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Gráfico: Convexo vs Linear (Melhorado)
    convexo = category_sums['Convexo']
    linear = category_sums['Linear']
    convexo_pct = (convexo[0] / convexo[1]) * 100 if convexo[1] > 0 else 0
    linear_pct = (linear[0] / linear[1]) * 100 if linear[1] > 0 else 0

    plt.figure(figsize=(10, 6))
    bars = plt.bar(
        [f'Convexo\n({convexo[0]}/{convexo[1]})', f'Linear\n({linear[0]}/{linear[1]})'],
        [convexo_pct, linear_pct],
        color=['#2E8B57', '#9370DB'],
        edgecolor='black',
        width=0.6
    )

    plt.ylabel('Correção (%)', fontsize=12, labelpad=15)
    plt.title('Desempenho por Categoria: Convexo vs Linear\n', fontsize=14, fontweight='bold')
    plt.ylim(0, 110)
    plt.yticks(range(0, 101, 10), fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Adiciona valores percentuais
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%',
                ha='center', va='bottom',
                fontsize=12,
                fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(base_folder, 'grafico_convexo_linear.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print("Gráficos salvos com sucesso.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Processar métricas dos folds e gerar CSV e gráficos.")
    parser.add_argument('--base_folder', type=str, required=True, help='Caminho para a pasta base com os resultados.')
    parser.add_argument('--output_csv', type=str, required=True, help='Caminho completo do CSV de saída.')

    args = parser.parse_args()
    process_metrics_folder(args.base_folder, args.output_csv)
