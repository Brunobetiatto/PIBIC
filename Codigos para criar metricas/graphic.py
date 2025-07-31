import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_metric(csv_files, metric_label, metric_index, output_filename, title):
    """
    Gera um gráfico comparativo para a métrica extraída da coluna (por posição) metric_index.
    
    Parâmetros:
      - csv_files: lista de caminhos para os CSVs.
      - metric_label: rótulo da métrica para o eixo X (ex.: "mAP50").
      - metric_index: índice (base 0) da coluna da métrica no CSV.
      - output_filename: nome do arquivo de saída (ex.: "comparacao_mAP50.png").
      - title: título do gráfico.
      
    O gráfico utiliza o eixo Y para as épocas e o eixo X para os valores da métrica.
    """
    plt.figure(figsize=(12, 8), dpi=300)
    
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        
        # Usa a coluna 'epoch', se existir; caso contrário, gera uma sequência de épocas.
        if 'epoch' in df.columns:
            epochs = df['epoch']
        else:
            epochs = range(1, len(df) + 1)
        
        # Extrai os valores da métrica pela posição e converte para array NumPy
        try:
            metric_values = df.iloc[:, metric_index].to_numpy()
        except IndexError:
            print(f"Índice {metric_index} fora do intervalo no arquivo {csv_file}.")
            continue
        
        # Converte epochs para array NumPy (caso seja uma Series ou range)
        if isinstance(epochs, pd.Series):
            epochs = epochs.to_numpy()
        else:
            epochs = np.array(list(epochs))
        
        plt.plot(metric_values, epochs, marker='o', label=os.path.basename(csv_file))
    
    plt.xlabel(metric_label)
    plt.ylabel("Épocas")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close()
    print(f"Gráfico salvo em: {output_filename}")

def main():
    # Diretório onde estão os CSVs
    csv_directory = "/home/bruno/projects/csvs/csv_somente_gato_limpo"
    csv_pattern = os.path.join(csv_directory, "*.csv")
    csv_files = glob.glob(csv_pattern)
    
    if len(csv_files) == 0:
        print("Nenhum arquivo CSV encontrado. Verifique o caminho!")
        return

    # Gráfico 1: Comparação mAP50 (índice 7)
    plot_metric(csv_files, metric_label="mAP50", metric_index=6, 
                output_filename="graphic_somente_gato_limpo/comparacao_mAP50.png", 
                title="Comparação mAP50 entre folds")
    
    # Gráfico 2: Comparação mAP50-95 (índice 8)
    plot_metric(csv_files, metric_label="mAP50-95", metric_index=7, 
                output_filename="graphic_somente_gato_limpo/comparacao_mAP50-95.png", 
                title="Comparação mAP50-95 entre folds")
    
    # Gráfico 3: Comparação Box Loss (índice 1)
    plot_metric(csv_files, metric_label="Box Loss", metric_index=1, 
                output_filename="graphic_somente_gato_limpo/comparacao_box_loss.png", 
                title="Comparação Box Loss entre folds")

if __name__ == '__main__':
    main()

