import os
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ultralytics import YOLO
from PIL import Image
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import torch
import gc
import pandas as pd
import re

# Configuração de memória GPU
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

def create_incremented_folder(base_path, fold):
    new_folder = os.path.join(base_path, f"resultado_fold_{fold}")
    os.makedirs(new_folder, exist_ok=True)
    return new_folder

def load_spreadsheet_data(excel_path):
    df_ge = pd.read_excel(excel_path, sheet_name='GE')
    df_samsung = pd.read_excel(excel_path, sheet_name='Samsung')
    df_catalogo = pd.concat([df_ge, df_samsung], ignore_index=True)
    
    image_to_species = {}
    image_to_transducer = {}
    
    for _, row in df_catalogo.iterrows():
        image_id = str(row.iloc[0]).strip()
        species = str(row.iloc[2]).strip().lower()
        transducer = str(row.iloc[10]).strip().lower()
        
        normalized_id = re.sub(r'\((\d+)\)', r'-\1-', image_id)
        normalized_id = re.sub(r'[^a-zA-Z0-9_-]', '', normalized_id).lower()
        
        image_to_species[normalized_id] = 'felino' if 'felino' in species else 'canino'
        image_to_transducer[normalized_id] = 'linear' if 'linear' in transducer else 'convexo'
    
    return image_to_species, image_to_transducer

def normalize_image_id(image_name):
    base = os.path.splitext(image_name)[0]
    base = re.sub(r'_(jpg|jpeg|png)\.rf\.[a-f0-9]+$', '', base, flags=re.IGNORECASE)
    match = re.search(r'^(.+?)([-(]\d+[-)])', base)
    if match:
        base = match.group(1) + '-' + re.search(r'\d+', match.group(2)).group() + '-'
    base = re.sub(r'[^a-zA-Z0-9_-]', '', base)
    return base.lower()

def safe_divide(a, b):
    return a / b if b > 0 else 0

def plot_confusion_matrix(cm, classes, title, filename, normalize=False):
    plt.figure(figsize=(8, 6))
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_category_accuracy(metrics, category_type, output_folder):
    if category_type == 'species':
        categories = ['Felino', 'Canino']
        corrects = [metrics['species']['felino'], metrics['species']['canino']]
        totals = [metrics['total_species']['felino'], metrics['total_species']['canino']]
        title = 'Acertos por Espécie'
        filename = 'species_accuracy.png'
        colors = ['#FF9AA2', '#FFB7B2']
    else:
        categories = ['Convexo', 'Linear']
        corrects = [metrics['transducer']['convexo'], metrics['transducer']['linear']]
        totals = [metrics['total_transducer']['convexo'], metrics['total_transducer']['linear']]
        title = 'Acertos por Transdutor'
        filename = 'transducer_accuracy.png'
        colors = ['#B5EAD7', '#C7CEEA']
    
    plt.figure(figsize=(10, 6))
    plt.bar(categories, totals, color=[f'{color}99' for color in colors], edgecolor=colors, linewidth=2, label='Total de Imagens', width=0.6)
    plt.bar(categories, corrects, color=colors, edgecolor=[c.replace('AA', '') for c in colors], linewidth=2, label='Acertos', width=0.4)
    
    for i, (correct, total) in enumerate(zip(corrects, totals)):
        plt.text(i, correct + 0.5, f'{correct}/{total}', ha='center', va='bottom', fontweight='bold')
        if total > 0:
            plt.text(i, correct/2, f'{safe_divide(correct, total)*100:.1f}%', ha='center', va='center', color='white', fontsize=12, fontweight='bold')
    
    plt.ylabel('Quantidade de Imagens')
    plt.title(title, fontsize=14)
    plt.ylim(0, max(totals)*1.2)
    plt.legend(loc='upper right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, filename))
    plt.close()

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()

    parser = argparse.ArgumentParser(description="Avaliação de modelo YOLO por fold.")
    parser.add_argument("--folds_dir", type=str, required=True)
    parser.add_argument("--weights_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    EXCEL_PATH = "C:/Users/Casa/Desktop/codigos-pibic2/codigos-pibic/Catalagoimagens09.05.2025.xlsx"
    image_to_species, image_to_transducer = load_spreadsheet_data(EXCEL_PATH)

    for fold in range(1, 6):
        print(f"\n{'='*40}\nProcessando Fold {fold}\n{'='*40}")
        
        model = YOLO(os.path.join(args.weights_dir, f"fold_{fold}/weights/best.pt"))
        data_yaml = os.path.join(args.folds_dir, f"fold_{fold}/data.yaml")
        
        metrics_results = model.val(
            data=data_yaml,
            plots=False,
            save_json=False,
            conf=0.25,
            iou=0.6
        )
        
        input_images_folder = os.path.join(args.folds_dir, f"fold_{fold}/test/images")
        input_labels_folder = os.path.join(args.folds_dir, f"fold_{fold}/test/labels")
        images = [os.path.join(input_images_folder, img) for img in os.listdir(input_images_folder) if img.endswith(('.jpg', '.png'))]
        output_folder = create_incremented_folder(args.output_dir, fold)

        metrics = {
            'total': 0,
            'found_full': 0,
            'found_partial': 0,
            'not_found': 0,
            'correct': 0,
            'species': {'felino': 0, 'canino': 0},
            'transducer': {'convexo': 0, 'linear': 0},
            'total_species': {'felino': 0, 'canino': 0},
            'total_transducer': {'convexo': 0, 'linear': 0}
        }

        y_true, y_pred = [], []
        color_map = {"0": "green", "1": "red"}

        for i, image_path in enumerate(images):
            image_name = os.path.basename(image_path)
            normalized_id = normalize_image_id(image_name)
            metrics['total'] += 1

            in_species = normalized_id in image_to_species
            in_transducer = normalized_id in image_to_transducer
            
            if in_species:
                species = image_to_species[normalized_id]
                metrics['total_species'][species] += 1
            if in_transducer:
                transducer = image_to_transducer[normalized_id]
                metrics['total_transducer'][transducer] += 1

            label_path = os.path.join(input_labels_folder, image_name.replace('.jpg', '.txt').replace('.png', '.txt'))
            real_class = None
            
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    real_class = f.readline().split()[0]

            img = np.array(Image.open(image_path))
            result = model(image_path)[0]
            
            if result.boxes:
                highest_conf_idx = result.boxes.conf.argmax()
                box = result.boxes.xyxy[highest_conf_idx]
                predicted_class = str(int(result.boxes.cls[highest_conf_idx]))
                highest_conf = float(result.boxes.conf[highest_conf_idx])

                if real_class is not None:
                    if predicted_class == real_class:
                        metrics['correct'] += 1
                        if in_species:
                            metrics['species'][species] += 1
                        if in_transducer:
                            metrics['transducer'][transducer] += 1

                    y_true.append(int(real_class))
                    y_pred.append(int(predicted_class))

                fig, ax = plt.subplots(1)
                ax.imshow(img)
                
                if real_class is not None:
                    ax.text(10, 30, f'Real: {"normal" if real_class == "0" else "abnormal"}',
                            color=color_map[real_class], fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
                
                x1, y1, x2, y2 = map(int, box)
                rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor=color_map[predicted_class], facecolor='none')
                ax.add_patch(rect)
                ax.text(x1, y1-10, f'Pred: {"normal" if predicted_class == "0" else "abnormal"} | Conf: {highest_conf:.2f}',
                        color=color_map[predicted_class], fontsize=10, bbox=dict(facecolor='white', alpha=0.5))
                
                ax.set_axis_off()
                plt.savefig(os.path.join(output_folder, f"result_{i+1}.jpg"), bbox_inches='tight')
                plt.close()

            torch.cuda.empty_cache()
            gc.collect()

        accuracy = accuracy_score(y_true, y_pred) if y_true else 0
        precision = precision_score(y_true, y_pred, average='binary') if y_true else 0
        recall = recall_score(y_true, y_pred, average='binary') if y_true else 0
        f1 = f1_score(y_true, y_pred, average='binary') if y_true else 0

        if y_true:
            cm_classes = confusion_matrix(y_true, y_pred, labels=[0, 1])
            plot_confusion_matrix(cm_classes, ['normal', 'abnormal'], 
                                'Matriz de Confusão - Classes', 
                                os.path.join(output_folder, 'confusion_matrix_classes.png'))
        
        plot_category_accuracy(metrics, 'species', output_folder)
        plot_category_accuracy(metrics, 'transducer', output_folder)
        
        with open(os.path.join(output_folder, "metrics.txt"), 'w') as f:
            f.write("=== Métricas Padrão do YOLO ===\n")
            f.write(f"mAP50: {metrics_results.box.map50:.4f}\n")
            f.write(f"mAP50-95: {metrics_results.box.map:.4f}\n\n")
            
            f.write("=== Métricas de Classificação ===\n")
            f.write(f"Acurácia: {accuracy:.4f}\n")
            f.write(f"Precisão: {precision:.4f}\n")
            f.write(f"Recall: {recall:.4f}\n")
            f.write(f"F1-Score: {f1:.4f}\n\n")
            
            f.write("=== Estatísticas por Categoria ===\n")
            f.write(f"Felinos Corretos: {metrics['species']['felino']}/{metrics['total_species']['felino']} ({safe_divide(metrics['species']['felino'], metrics['total_species']['felino'])*100:.1f}%)\n")
            f.write(f"Caninos Corretos: {metrics['species']['canino']}/{metrics['total_species']['canino']} ({safe_divide(metrics['species']['canino'], metrics['total_species']['canino'])*100:.1f}%)\n")
            f.write(f"Convexo Corretos: {metrics['transducer']['convexo']}/{metrics['total_transducer']['convexo']} ({safe_divide(metrics['transducer']['convexo'], metrics['total_transducer']['convexo'])*100:.1f}%)\n")
            f.write(f"Linear Corretos: {metrics['transducer']['linear']}/{metrics['total_transducer']['linear']} ({safe_divide(metrics['transducer']['linear'], metrics['total_transducer']['linear'])*100:.1f}%)\n")

        print(f"\nProcessamento do Fold {fold} concluído!")
        print(f"Relatórios salvos em: {output_folder}")
