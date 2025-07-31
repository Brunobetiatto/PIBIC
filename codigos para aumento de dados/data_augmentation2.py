import os
import cv2
import numpy as np
import random
import shutil
import argparse
import re
from collections import defaultdict

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, matrix, (w, h))
    return rotated_image, matrix

def rotate_coordinates(coords, matrix, w, h):
    coords_abs = np.array([(x * w, y * h) for x, y in coords])
    ones = np.ones(shape=(len(coords_abs), 1))
    coords_homo = np.hstack([coords_abs, ones])
    rotated_coords = matrix.dot(coords_homo.T).T
    rotated_coords_normalized = [(x / w, y / h) for x, y in rotated_coords]
    return rotated_coords_normalized

def write_new_label(class_id, rotated_coords, label_file_path):
    new_label = [class_id] + [coord for pair in rotated_coords for coord in pair]
    with open(label_file_path, 'w') as f:
        label_line = ' '.join(map(str, new_label))
        f.write(label_line + '\n')

def read_label_file(label_file_path):
    with open(label_file_path, 'r') as f:
        return f.readline().strip()

def find_next_fold(output_dir):
    i = 1
    while True:
        fold_name = f"fold_{i}"
        fold_path = os.path.join(output_dir, fold_name)
        if not os.path.exists(fold_path):
            return fold_path
        i += 1

def extract_base_name(image_file):
    base = re.sub(r'(\..*|_jpg|\.rf\.[a-f0-9]+)$', '', image_file)
    base = re.sub(r'-+$', '', base)
    if '-' in base:
        parts = base.split('-')
        base = f"{parts[0]}-{parts[1]}"
    return base

def determine_image_type(base_name):
    if base_name.upper().startswith('L'):
        return 'linear'
    return 'microconvexo'

def process_directory(image_dir, label_dir, output_image_dir, output_label_dir, target_per_class):
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    class_subcount = {
        '0': {'microconvexo': 0, 'linear': 0},
        '1': {'microconvexo': 0, 'linear': 0},
    }

    available_images = defaultdict(list)

    image_files = sorted(os.listdir(image_dir))
    for image_file in image_files:
        base_name = extract_base_name(image_file)
        tipo = determine_image_type(base_name)
        label_path = os.path.join(label_dir, os.path.splitext(image_file)[0] + '.txt')

        if not os.path.exists(label_path):
            continue

        try:
            label = read_label_file(label_path)
            label_parts = label.split()
            class_id = label_parts[0]

            if class_id not in ['0', '1']:
                continue

            available_images[f"{class_id}_{tipo}"].append(image_file)

        except Exception as e:
            print(f"Erro ao processar {image_file}: {str(e)}")

    target_per_subclass = target_per_class // 2

    for class_tipo, files in available_images.items():
        class_id, tipo = class_tipo.split('_')
        needed = target_per_subclass
        available = len(files)

        print(f"\nProcessando classe {class_id}, tipo {tipo}:")
        print(f"Necessário: {needed}, Disponível: {available}")

        if available == 0:
            continue

        copies_per_image = needed // available
        remainder = needed % available

        for i, image_file in enumerate(files):
            base_name = extract_base_name(image_file)
            image_path = os.path.join(image_dir, image_file)
            label_path = os.path.join(label_dir, os.path.splitext(image_file)[0] + '.txt')

            try:
                image = cv2.imread(image_path)
                if image is None:
                    continue

                (h, w) = image.shape[:2]
                label = read_label_file(label_path)
                label_parts = label.split()
                coords = list(map(float, label_parts[1:]))
                coords_pairs = [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]

                copies = copies_per_image + 1 if i < remainder else copies_per_image

                for copy_num in range(copies):
                    angle = random.uniform(0, 0)
                    rotated_image, rotation_matrix = rotate_image(image, angle)
                    rotated_coords = rotate_coordinates(coords_pairs, rotation_matrix, w, h)

                    valid = all(0 <= x <= 1 and 0 <= y <= 1 for x, y in rotated_coords)
                    if not valid:
                        continue

                    current_count = class_subcount[class_id][tipo]
                    output_image_path = os.path.join(output_image_dir, f'{class_id}_{tipo}_{current_count}.jpg')
                    output_label_path = os.path.join(output_label_dir, f'{class_id}_{tipo}_{current_count}.txt')

                    cv2.imwrite(output_image_path, rotated_image)
                    write_new_label(class_id, rotated_coords, output_label_path)

                    class_subcount[class_id][tipo] += 1

                    if class_subcount[class_id][tipo] >= target_per_subclass:
                        break

            except Exception as e:
                print(f"Erro ao processar {image_file}: {str(e)}")

    print("\nResultado final:")
    for class_id in class_subcount:
        for tipo in class_subcount[class_id]:
            print(f"Classe {class_id}, Tipo {tipo}: {class_subcount[class_id][tipo]}/{target_per_subclass}")

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
    parser.add_argument("--target_per_class_train", type=int, default=830)
    parser.add_argument("--target_per_class_valid", type=int, default=300)
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
    process_directory(train_image_dir, train_label_dir, output_train_image_dir, output_train_label_dir,
                      args.target_per_class_train)
    process_directory(valid_image_dir, valid_label_dir, output_valid_image_dir, output_valid_label_dir,
                      args.target_per_class_valid)
    copy_test_images(test_image_dir, test_label_dir, output_test_image_dir, output_test_label_dir)

    data_yaml_path = os.path.join(args.input_fold, "data.yaml")
    if os.path.exists(data_yaml_path):
        shutil.copy(data_yaml_path, os.path.join(output_fold_path, "data.yaml"))
        print("Arquivo data.yaml copiado.")

if __name__ == "__main__":
    main()
