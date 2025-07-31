import os
import cv2
import numpy as np
import random
import shutil
import argparse
import re
import pandas as pd
from collections import defaultdict

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def crop_and_find_component(image, threshold=5):
    """
    Crops an image based on a threshold and finds the connected component.
    Returns:
      - cropped_image: The cropped image
      - bbox: The bounding box of the component (x, y, w, h)
    """
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    # Apply thresholding to create a binary mask
    _, binary_mask = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return image, (0, 0, image.shape[1], image.shape[0])

    # Find largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    cropped_image = image[y:y+h, x:x+w]
    
    return cropped_image, (x, y, w, h)

def adjust_coordinates(coords_pairs, orig_size, bbox):
    """
    Adjust normalized coordinates after cropping
    """
    orig_w, orig_h = orig_size
    x, y, w, h = bbox
    new_coords = []
    
    for coord_x, coord_y in coords_pairs:
        # Convert to absolute coordinates
        abs_x = coord_x * orig_w
        abs_y = coord_y * orig_h
        
        # Adjust based on crop
        adj_x = (abs_x - x) / w
        adj_y = (abs_y - y) / h
        
        # Clamp to [0, 1]
        adj_x = max(0, min(1, adj_x))
        adj_y = max(0, min(1, adj_y))
        
        new_coords.append((adj_x, adj_y))
    
    return new_coords
# ======== FIM DO NOVO CÓDIGO ========


def apply_random_brightness(image, brightness_range=(2, 2)):
    if random.random() < 0.0:
        brightness = random.uniform(*brightness_range)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float64)
        hsv[:,:,2] = np.clip(hsv[:,:,2] * brightness, 0, 255)
        image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return image

def apply_random_flip(image, coords_pairs):
    if random.random() < 0.0:
        image = cv2.flip(image, 1)
        return image, [(1.0 - x, y) for x, y in coords_pairs]
    return image, coords_pairs

def rotate_image(image, angle, coords_pairs):
    if random.random() < 0.0:
        angle = random.uniform(-30, 30)
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_image = cv2.warpAffine(image, matrix, (w, h))
        coords_abs = np.array([(x * w, y * h) for x, y in coords_pairs])
        coords_homo = np.hstack([coords_abs, np.ones((len(coords_abs), 1))])
        rotated_coords = matrix.dot(coords_homo.T).T
        rotated_coords_normalized = [(x / w, y / h) for x, y in rotated_coords]
        if not all(0 <= x <= 1 and 0 <= y <= 1 for x, y in rotated_coords_normalized):
            return image, coords_pairs
        return rotated_image, rotated_coords_normalized
    return image, coords_pairs

def apply_augmentations(image, coords_pairs):
    # Aplicar novo crop baseado em threshold antes das outras transformações
    orig_size = (image.shape[1], image.shape[0])
    cropped_img, bbox = crop_and_find_component(image)
    adjusted_coords = adjust_coordinates(coords_pairs, orig_size, bbox)
    
    # Aplicar outras transformações
    image = apply_random_brightness(cropped_img)
    image, adjusted_coords = apply_random_flip(image, adjusted_coords)
    image, adjusted_coords = rotate_image(image, 0, adjusted_coords)
    
    return image, adjusted_coords

def write_new_label(class_id, coords_pairs, label_file_path):
    new_label = [class_id] + [coord for pair in coords_pairs for coord in pair]
    with open(label_file_path, 'w') as f:
        f.write(' '.join(map(str, new_label)) + '\n')

def read_label_file(label_file_path):
    with open(label_file_path, 'r') as f:
        return f.readline().strip()

def find_next_fold(output_dir):
    i = 1
    while True:
        fold_path = os.path.join(output_dir, f"fold_{i}")
        if not os.path.exists(fold_path):
            return fold_path
        i += 1

def extract_base_name(image_file):
    base = os.path.splitext(image_file)[0]
    base = re.sub(r'_(jpg|jpeg|png)\.rf\.[a-f0-9]+$', '', base, flags=re.IGNORECASE)
    match = re.search(r'^(.+?)([-(]\d+[-)])', base)
    if match:
        base = match.group(1) + '-' + re.search(r'\d+', match.group(2)).group() + '-'
    base = re.sub(r'[^a-zA-Z0-9_-]', '', base)
    return base.lower()

# === Carrega planilha e prepara o dicionário ===
excel_path = "/home/bruno/projects/datasets/datasets-final/Catalago_imagens_meu.xlsx"
df_ge = pd.read_excel(excel_path, sheet_name='GE')
df_samsung = pd.read_excel(excel_path, sheet_name='Samsung')
df_catalogo = pd.concat([df_ge, df_samsung], ignore_index=True)

image_type_lookup = {}
for _, row in df_catalogo.iterrows():
    nome = str(row.iloc[0]).strip()
    tipo_raw = str(row.iloc[10]).strip().lower()
    
    if 'microconvexo' in tipo_raw or 'convexo' in tipo_raw:
        tipo = 'convexo'
    elif 'linear' in tipo_raw:
        tipo = 'linear'
    else:
        tipo = 'convexo'
    
    image_type_lookup[nome.lower()] = tipo

def determine_image_type(raw_name):
    return image_type_lookup.get(raw_name.lower(), 'convexo')

def process_directory(image_dir, label_dir, output_image_dir, output_label_dir, target_per_class):
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    class_subcount = {'0': {'convexo': 0, 'linear': 0}, '1': {'convexo': 0, 'linear': 0}}
    available_images = defaultdict(list)

    for image_file in sorted(os.listdir(image_dir)):
        base_name = extract_base_name(image_file)
        tipo = determine_image_type(base_name)
        
        label_path = os.path.join(label_dir, os.path.splitext(image_file)[0] + '.txt')
        if not os.path.exists(label_path):
            continue

        try:
            label = read_label_file(label_path)
            class_id = label.split()[0]
            if class_id not in ['0', '1']:
                continue
            available_images[f"{class_id}_{tipo}"].append(image_file)
        except Exception as e:
            print(f"Erro ao processar {image_file}: {e}")

    target_per_subclass = target_per_class // 2

    for class_tipo, files in available_images.items():
        class_id, tipo = class_tipo.split('_')
        needed = target_per_subclass
        available = len(files)
        if available == 0:
            continue
            
        copies_per_image = needed // available
        remainder = needed % available

        for i, image_file in enumerate(files):
            image_path = os.path.join(image_dir, image_file)
            label_path = os.path.join(label_dir, os.path.splitext(image_file)[0] + '.txt')

            try:
                image = cv2.imread(image_path)
                if image is None:
                    continue
                    
                coords = list(map(float, read_label_file(label_path).split()[1:]))
                coords_pairs = [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]
                copies = copies_per_image + 1 if i < remainder else copies_per_image

                for copy_num in range(copies):
                    # Aplica novo corte e ajusta coordenadas
                    augmented_image, augmented_coords = apply_augmentations(image.copy(), coords_pairs.copy())
                    
                    count = class_subcount[class_id][tipo]
                    base_filename = os.path.splitext(image_file)[0]
                    out_img = os.path.join(output_image_dir, f'{base_filename}_aug{copy_num}.jpg')
                    out_lbl = os.path.join(output_label_dir, f'{base_filename}_aug{copy_num}.txt')
                    
                    cv2.imwrite(out_img, augmented_image)
                    write_new_label(class_id, augmented_coords, out_lbl)
                    
                    class_subcount[class_id][tipo] += 1
                    if class_subcount[class_id][tipo] >= target_per_subclass:
                        break
            except Exception as e:
                print(f"Erro ao processar {image_file}: {e}")

    for class_id in class_subcount:
        for tipo in class_subcount[class_id]:
            print(f"Classe {class_id}, Tipo {tipo}: {class_subcount[class_id][tipo]} imagens aumentadas (target: {target_per_subclass})")

def copy_test_images(test_image_dir, test_label_dir, output_test_image_dir, output_test_label_dir):
    os.makedirs(output_test_image_dir, exist_ok=True)
    os.makedirs(output_test_label_dir, exist_ok=True)
    for file in os.listdir(test_image_dir):
        shutil.copy(os.path.join(test_image_dir, file), os.path.join(output_test_image_dir, file))
    for file in os.listdir(test_label_dir):
        shutil.copy(os.path.join(test_label_dir, file), os.path.join(output_test_label_dir, file))
    print("Imagens de teste copiadas.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_fold", type=str, help="Caminho da pasta do fold de entrada")
    parser.add_argument("output_dir", type=str, help="Caminho da pasta onde os dados aumentados serão salvos")
    parser.add_argument("--target_per_class_train", type=int, default=500)
    parser.add_argument("--target_per_class_valid", type=int, default=100)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        set_seed(args.seed)

    output_fold_path = find_next_fold(args.output_dir)
    os.makedirs(output_fold_path, exist_ok=True)

    train_image_dir = os.path.join(args.input_fold, "train/images")
    train_label_dir = os.path.join(args.input_fold, "train/labels")
    valid_image_dir = os.path.join(args.input_fold, "valid/images")
    valid_label_dir = os.path.join(args.input_fold, "valid/labels")
    test_image_dir = os.path.join(args.input_fold, "test/images")
    test_label_dir = os.path.join(args.input_fold, "test/labels")

    output_train_image_dir = os.path.join(output_fold_path, "train/images")
    output_train_label_dir = os.path.join(output_fold_path, "train/labels")
    output_valid_image_dir = os.path.join(output_fold_path, "valid/images")
    output_valid_label_dir = os.path.join(output_fold_path, "valid/labels")
    output_test_image_dir = os.path.join(output_fold_path, "test/images")
    output_test_label_dir = os.path.join(output_fold_path, "test/labels")

    print(f"Criando imagens aumentadas em: {output_fold_path}")
    process_directory(train_image_dir, train_label_dir, output_train_image_dir, output_train_label_dir, args.target_per_class_train)
    process_directory(valid_image_dir, valid_label_dir, output_valid_image_dir, output_valid_label_dir, args.target_per_class_valid)
    copy_test_images(test_image_dir, test_label_dir, output_test_image_dir, output_test_label_dir)

    data_yaml_path = os.path.join(args.input_fold, "data.yaml")
    if os.path.exists(data_yaml_path):
        shutil.copy(data_yaml_path, os.path.join(output_fold_path, "data.yaml"))
        print("Arquivo data.yaml copiado.")

if __name__ == "__main__":
    main()
