import os
import cv2
import numpy as np
import random
import shutil
import yaml
import re
from pathlib import Path

# --- Funções de Crop (Adicionadas do código de augmentation) ---
def crop_and_find_component(image, threshold=5):
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    _, binary_mask = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return image, (0, 0, image.shape[1], image.shape[0])

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    cropped_image = image[y:y+h, x:x+w]
    
    return cropped_image, (x, y, w, h)

def adjust_coordinates(coords_pairs, orig_size, bbox):
    orig_w, orig_h = orig_size
    x, y, w, h = bbox
    new_coords = []
    
    for coord_x, coord_y in coords_pairs:
        abs_x = coord_x * orig_w
        abs_y = coord_y * orig_h
        
        adj_x = (abs_x - x) / w
        adj_y = (abs_y - y) / h
        
        adj_x = max(0, min(1, adj_x))
        adj_y = max(0, min(1, adj_y))
        
        new_coords.append((adj_x, adj_y))
    
    return new_coords
# --- Fim das funções de crop ---

def load_yaml(yaml_path):
    with open(yaml_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def get_image_label_pairs(split_dir: Path):
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"
    if not images_dir.exists() or not labels_dir.exists():
        print(f"Diretórios {images_dir} ou {labels_dir} não existem.")
        return []
    image_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))])
    pairs = []
    for img in image_files:
        base = os.path.splitext(img)[0]
        label_file = base + ".txt"
        if label_file in os.listdir(labels_dir):
            pairs.append((images_dir / img, labels_dir / label_file))
        else:
            print(f"Label {label_file} não encontrado para a imagem {img}.")
    return pairs

def extract_group_key(filename: str) -> str:
    name = os.path.splitext(filename)[0]
    if re.match(r'^[A-Za-z]_', name):
        name = name[2:]
    return name.split('-', 1)[0]

def create_folds(base_dir, num_folds=5, seed=42):
    base_dir = Path(base_dir)
    data_yaml_path = base_dir / "data.yaml"
    if not data_yaml_path.exists():
        raise FileNotFoundError("Arquivo data.yaml não encontrado no diretório base.")
    
    all_pairs = []
    splits_dirs = ["train", "valid", "test"]
    for sp in splits_dirs:
        all_pairs.extend(get_image_label_pairs(base_dir / sp))
    print(f"Total de imagens encontradas: {len(all_pairs)}")
    
    groups = {}
    for img_path, lbl_path in all_pairs:
        key = extract_group_key(img_path.name)
        groups.setdefault(key, []).append((img_path, lbl_path))
    group_items = list(groups.values())
    total_groups = len(group_items)
    print(f"Total de grupos formados: {total_groups}")
    
    random.seed(seed)
    random.shuffle(group_items)
    
    base_size = total_groups // num_folds
    remainder = total_groups % num_folds
    splits = []
    idx = 0
    for fold_idx in range(num_folds):
        size = base_size + (1 if fold_idx < remainder else 0)
        splits.append(group_items[idx: idx + size])
        idx += size
    
    folds_dir = base_dir / "folds"
    folds_dir.mkdir(exist_ok=True)
    
    for fold in range(num_folds):
        train_idxs = [(fold + i) % num_folds for i in range(3)]
        valid_idx = (fold + 3) % num_folds
        test_idx  = (fold + 4) % num_folds
        
        train_pairs = [
            pair
            for gi in train_idxs
            for group in splits[gi]
            for pair in group
        ]
        valid_pairs = [
            pair
            for group in splits[valid_idx]
            for pair in group
        ]
        test_pairs = [
            pair
            for group in splits[test_idx]
            for pair in group
        ]
        
        fold_path = folds_dir / f"fold_{fold+1}"
        for sp in splits_dirs:
            (fold_path / sp / "images").mkdir(parents=True, exist_ok=True)
            (fold_path / sp / "labels").mkdir(parents=True, exist_ok=True)
        
        # --- Função Modificada para Aplicar Crop ---
        def process_and_save_pair(img_path, lbl_path, split_name):
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"Failed to read image: {img_path}")
                return

            with open(lbl_path, 'r') as f:
                label_line = f.readline().strip()
            parts = label_line.split()
            if not parts:
                print(f"Empty label in {lbl_path}")
                return
            class_id = parts[0]
            try:
                coords = list(map(float, parts[1:]))
            except:
                print(f"Error parsing coordinates in {lbl_path}")
                return
            coords_pairs = [(coords[i], coords[i+1]) for i in range(0, len(coords), 2)]

            cropped_image, bbox = crop_and_find_component(image)
            orig_size = (image.shape[1], image.shape[0])
            adjusted_coords = adjust_coordinates(coords_pairs, orig_size, bbox)

            new_img_path = fold_path / split_name / "images" / img_path.name
            cv2.imwrite(str(new_img_path), cropped_image)

            new_lbl_path = fold_path / split_name / "labels" / lbl_path.name
            with open(new_lbl_path, 'w') as f:
                adjusted_flat = [str(coord) for pair in adjusted_coords for coord in pair]
                line = class_id + ' ' + ' '.join(adjusted_flat)
                f.write(line + '\n')
        # --- Fim da modificação ---
        
        # Processar cada par com a nova lógica
        for img, lbl in train_pairs:
            process_and_save_pair(img, lbl, "train")
        for img, lbl in valid_pairs:
            process_and_save_pair(img, lbl, "valid")
        for img, lbl in test_pairs:
            process_and_save_pair(img, lbl, "test")
        
        shutil.copy(data_yaml_path, fold_path / "data.yaml")
        print(
            f"Fold {fold+1}: "
            f"{len(train_pairs)} imgs treino, "
            f"{len(valid_pairs)} imgs validação, "
            f"{len(test_pairs)} imgs teste."
        )
    
    print("Todas as folds foram criadas com sucesso.")

if __name__ == '__main__':
    base_dataset_path = "C:/Users/Casa/Desktop/codigos-pibic2/codigos-pibic/dataset_linear"
    create_folds(base_dataset_path)