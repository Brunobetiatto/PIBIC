import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Configuração de estilo corrigida
plt.style.use('seaborn-v0_8-whitegrid')  # Estilo compatível
sns.set_theme(style="whitegrid")  # Configuração adicional do Seaborn
sns.set_palette("colorblind")
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': 'Times New Roman',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.figsize': (12, 8),
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.edgecolor': '0.3',
    'axes.linewidth': 0.8
})

# Dados dos cenários
scenarios = {
    "Convexo_resultado": "Convexo",
    "SemiCompleto_resultado": "Semi-Completo",
    "augmented_Convexo_resultado": "Conv. Ampliado",
    "Linear_resultado": "Linear",
    "augmented_Linear_resultado": "Linear Ampliado",
    "augmented_SemiCompleto_res": "Semi-Comp. Ampliado"
}

metrics = ["Acurácia", "Precision", "Recall", "F1 Score", "mAP@50", "mAP@50-95"]

def extract_data_from_excel(file_path):
    """Extrai dados médios de cada cenário"""
    data = {}
    sheets = pd.read_excel(file_path, sheet_name=None)
    
    for sheet_name, scenario_name in scenarios.items():
        if sheet_name in sheets:
            df = sheets[sheet_name]
            
            # Localizar linha da média geral
            mask = df.iloc[:, 0].str.contains("Média Geral", na=False)
            if not mask.any():
                mask = df.iloc[:, 0].str.contains("Média", na=False)
                
            mean_row = df[mask]
            
            if not mean_row.empty:
                # Converter valores numéricos e converter para porcentagem
                values = mean_row.iloc[:, 1:7].values[0]
                data[scenario_name] = {
                    metric: float(str(val).replace(',', '.')) * 100  # Converter para %
                    for metric, val in zip(metrics, values)
                }
    
    return pd.DataFrame(data).T

def create_comparison_plots(df):
    """Cria gráficos comparativos para cada métrica"""
    # Ordenar cenários
    scenario_order = [
        "Convexo",
        "Semi-Completo",
        "Conv. Ampliado",
        "Linear",
        "Linear Ampliado",
        "Semi-Comp. Ampliado"
    ]
    df = df.reindex(scenario_order)
    
    # Criar gráficos individuais
    for metric in metrics:
        plt.figure(figsize=(12, 8))
        
        # Gráfico de barras com cores distintas
        colors = plt.cm.viridis(np.linspace(0, 1, len(df)))
        bars = plt.bar(
            df.index, 
            df[metric],
            color=colors,
            edgecolor='black',
            linewidth=1.2,
            alpha=0.85
        )
        
        # Adicionar valores em porcentagem
        for bar in bars:
            height = bar.get_height()
            plt.annotate(
                f'{height:.1f}%',  # Formato com 1 casa decimal e símbolo %
                xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center',
                va='bottom',
                fontsize=12,
                fontweight='bold'
            )
        
        # Linha de referência (75%)
        plt.axhline(y=75, color='r', linestyle='--', alpha=0.7, linewidth=1.5)
        plt.annotate('75%', (5.2, 75), 
                    ha='left', va='center', color='red', fontsize=10, fontweight='bold')
        
        # Configurações
        plt.title(f'Comparação de {metric} entre Cenários', 
                 fontsize=18, fontweight='bold', pad=20)
        plt.ylabel(f'{metric} (%)', fontweight='bold')  # Adicionar % no eixo Y
        plt.xlabel('Cenários', fontweight='bold')
        plt.ylim(0, 105)  # Ajustar para escala de porcentagem
        plt.xticks(rotation=15)
        plt.grid(True, alpha=0.2)
        
        # Adicionar grade horizontal mais densa
        plt.yticks(np.arange(0, 101, 10))
        
        plt.tight_layout()
        plt.savefig(f'{metric}_comparison.png', bbox_inches='tight', transparent=True)
        plt.show()

# Caminho para o arquivo Excel (ajuste conforme necessário)
file_path = "Relatorio_Final_Resultados_Comparativo.xlsx"

# Processar dados
try:
    df = extract_data_from_excel(file_path)
    print("Dados extraídos com sucesso:")
    print(df)
    
    create_comparison_plots(df)
    
except Exception as e:
    print(f"Erro: {e}")
    print("Verifique o caminho do arquivo e a estrutura das planilhas")

# Caso tenha problemas com o Excel, use esta alternativa com dados fictícios
if 'df' not in locals():
    print("\nUsando dados de exemplo para demonstração")
    data = {
        'Convexo': [0.6129, 0.6468, 0.6613, 0.6482, 0.6579, 0.5654],
        'Semi-Completo': [0.6954, 0.7318, 0.7249, 0.7265, 0.6232, 0.5329],
        'Conv. Ampliado': [0.6837, 0.7143, 0.7180, 0.7113, 0.7980, 0.6855],
        'Linear': [0.5851, 0.5924, 0.7362, 0.6413, 0.5746, 0.5070],
        'Linear Ampliado': [0.8603, 0.9129, 0.8903, 0.8971, 0.9130, 0.8497],
        'Semi-Comp. Ampliado': [0.6916, 0.7622, 0.6632, 0.7042, 0.8416, 0.7499]
    }
    
    # Converter dados fictícios para porcentagem
    df = pd.DataFrame({k: [v*100 for v in vals] for k, vals in data.items()}, index=metrics).T
    create_comparison_plots(df)