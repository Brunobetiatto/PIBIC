# Projeto PIBIC – Implementação de IA no Diagnóstico Ultrassonográfico em Cães

## 📄 Relatório Final do PIBIC

Para quem quiser se aprofundar, disponibilizamos o **Relatório Final** completo em PDF:

➡️ [Relatório Final (PDF)](docs/Modelo-Relatorio_Final.pdf)

---

## 📂 Estrutura do Projeto

O repositório está organizado em pastas que cobrem todo o fluxo do seu projeto:


### Descrição de cada pasta

- **Base de dados/**  
  Contém as imagens ultrassonográficas e arquivos de anotação (bounding boxes e labels) usadas para treinar e avaliar o modelo.

- **codigos para k-fold (repartição de dados)/**  
  Script que divide o dataset em *k* folds para validação cruzada, garantindo uma avaliação mais robusta.

- **codigos para aumento de dados/**  
  Implementa técnicas de data augmentation (sem crops/carvas) para enriquecer o conjunto de treino com variações de brilho, rotação, flip, etc.

- **codigo de treino/**  
  Executa o treinamento do modelo de detecção YOLOv11-nano, salvando checkpoints e logs de performance a cada época.

- **Codigos para criar métricas/**  
  Processa os resultados de inferência em arquivos de métricas brutas ('metrics.txt') e prepara dados para análise.

- **Métri­cas/**  
  Armazena o CSV final com estatísticas agregadas (precisão, recall, F1-score, mAP) e pastas com gráficos que ilustram o desempenho por categoria e experimento.

---

_Nas próximas seções, veremos como rodar cada script passo a passo.

## ⚙️ Pré-requisitos e Execução

Antes de rodar qualquer script, verifique se você tem:

- **Python 3.8+**  
- **CUDA 11.0+** e GPU NVIDIA (recomendado para treinamento YOLO)  
- **pip**

### 1. Clone o repositório  


### 2. Crie e ative um ambiente virtual

python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows

### 3. Instale as dependências

pip install -r requirements.txt

| Logo                                                                                                          | Tecnologia           | Uso principal                                         |
| ------------------------------------------------------------------------------------------------------------- | -------------------- | ----------------------------------------------------- |
| <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/python/python-original.svg"   width="24"/>       | **Python 3.8+**      | Linguagem principal, scripts de processamento         |
| <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/opencv/opencv-original.svg"   width="24"/>       | **OpenCV**           | Leitura, transformação e escrita de imagens           |
| <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/pytorch/pytorch-original.svg"   width="24"/>     | **PyTorch**          | Backbone de treinamento e inferência do modelo YOLO   |
| <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/pandas/pandas-original.svg"     width="24"/>     | **Pandas**           | Manipulação de planilhas e geração de CSV             |
| <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/numpy/numpy-original.svg"       width="24"/>     | **NumPy**            | Operações numéricas e matrizes                        |
| <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/matplotlib/matplotlib-original.svg" width="24"/> | **Matplotlib**       | Plotagem de métricas e gráficos                       |
| <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/pillow/pillow-original.svg"     width="24"/>     | **Pillow**           | Processamento básico de imagens (RandAugment seguro)  |
| ![YOLOv11-nano](https://img.shields.io/badge/YOLOv11-nano-orange)                                             | **Ultralytics YOLO** | Detecção de objetos em US – arquitetura leve e rápida |

---

## 🏋️‍♂️ Script de Treinamento por Fold (`train_yolo.py`)

Este script automatiza o treinamento do modelo YOLOv11-nano em múltiplos folds (pastas de dados), salvando checkpoints e logs de cada experimento em pastas separadas.

---

### 📋 O que ele faz

1. **Define a seed** para tornar os resultados reprodutíveis ('set_seed').
2. **Percorre cada subpasta** em 'root_dir' (cada fold é uma pasta com um 'data.yaml').
3. Para cada fold:
   - Carrega o arquivo 'data.yaml'.
   - Instancia o modelo YOLO via 'ultralytics.YOLO(modelo)'.
   - Executa 'model.train(...)' com 100 épocas e batch definido (pode ser ajustado).
   - Salva resultados em 'output_dir/<nome-do-fold>/'.
4. Trata erros comuns (interrupção manual, falta de arquivo YAML ou exceções inesperadas) com mensagens claras no console.

---

### 🛠️ Parâmetros de Entrada

| Argumento       | Tipo   | Descrição                                                    | Padrão |
|-----------------|--------|--------------------------------------------------------------|--------|
| 'root_dir'      | 'str'  | Caminho para a pasta que contém os folds (cada fold em subpasta). | —      |
| 'output_dir'    | 'str'  | Onde salvar logs, pesos e gráficos gerados pelo treinamento.    | —      |
| '--seed'        | 'int'  | Semente para NumPy, PyTorch e random (reprodutibilidade).       | '42'   |
| '--model'       | 'str'  | Caminho ou nome do modelo YOLO (ex: 'yolov11n') a ser usado.    | —      |

---

### ⚙️ Como usar

1. **Preparar os folds**  
   - Cada pasta em 'root_dir' deve conter um 'data.yaml' com:
        yaml
     train: ../images/train
     val:   ../images/val
     nc: 2
     names: ['normal','alterado']
  
2. **Executar o treinamento**  
      bash
   python train_yolo.py \
     /caminho/para/folds/ \
     /caminho/para/resultados/ \
     --seed 123 \
     --model yolov11n
---

## ✨ Script de Aumento de Dados Customizado (albumentation_1.py)

Este script gera novas amostras de imagem e arquivos de label para treino e validação, aplicando transformações seguras (sem deformar via crop) e mantendo a consistência dos bounding boxes.

---

### 📋 O que ele faz

- Define a seed para reprodutibilidade (função `set_seed`).
- Recorta cada imagem ao redor do maior componente (linha de interesse) e ajusta as coordenadas originais de caixa.
- Aplica transformações opcionais:
  - brilho aleatório  
  - flip horizontal  
  - rotação suave (± 30°)
- Gera cópias por classe até atingir a quantidade desejada para treino e validação.
- Copia todo o conjunto de teste sem alterações.
- Replica o arquivo `data.yaml` no novo fold.

---

### 🔑 Principais funções

- `set_seed(seed)`: define sementes de `random` e `numpy`.  
- `crop_and_find_component(imagem, threshold=5)`: converte para escala de cinza, aplica threshold e encontra o maior contorno, retornando imagem recortada e bbox.  
- `adjust_coordinates(pares_de_coords, tamanho_original, bbox)`: ajusta coordenadas normalizadas da imagem original para a região recortada.  
- `apply_random_brightness(imagem)`: altera brilho via HSV (controlável por probabilidade interna).  
- `apply_random_flip(imagem, pares_de_coords)`: faz flip horizontal e atualiza as coordenadas X.  
- `rotate_image(imagem, pares_de_coords)`: aplica rotação suave e recalcula coordenadas, descartando transformações que saiam do frame.  
- `process_directory(...)`: lê imagens e labels de `train/` e `valid/`, calcula quantas cópias gerar, salva imagens aumentadas e novos labels.  
- `copy_test_images(...)`: replica pastas `test/images` e `test/labels` sem modificações.  
- `main()`: lê argumentos, cria um novo `fold_X` em `output_dir`, executa `process_directory` e `copy_test_images`, e copia o `data.yaml`.

---

### ⚙️ Parâmetros de entrada

| Argumento                   | Tipo | Descrição                                                        |
| --------------------------- | ---- | ---------------------------------------------------------------- |
| `input_fold`                | str  | Pasta do fold original contendo subpastas `train`, `valid` e `test` |
| `output_dir`                | str  | Pasta onde o novo fold será criado (ex: `augmented_folds/`)      |
| `--target_per_class_train`  | int  | Quantidade de imagens por classe em `train` (padrão: 500)        |
| `--target_per_class_valid`  | int  | Quantidade de imagens por classe em `valid` (padrão: 100)        |
| `--seed`                    | int  | Semente para reprodutibilidade (opcional)                        |

---

### 🚀 Como usar

1. Preparar as pastas de entrada:  
   - O fold original deve conter:  
     - `train/images`  
     - `train/labels`  
     - `valid/images`  
     - `valid/labels`  
     - `test/images`  
     - `test/labels`  
     - `data.yaml`

2. Executar o comando no terminal:  
   ```bash
   python albumentation_1.py \
     <caminho/para/fold_original> \
     <caminho/para/folds_aumentados> \
     --target_per_class_train 500 \
     --target_per_class_valid 100 \
     --seed 42
---

## 🧪 Script de Avaliação e Visualização de Resultados (`teste_pasta.py`)

Este script carrega pesos treinados do YOLO, avalia o modelo em imagens de teste por fold, gera métricas clássicas de classificação e detecção, plota matrizes de confusão e gráficos de acurácia por categoria, e salva figuras e relatórios em pastas separadas.

---

### 📋 O que ele faz

- Carrega catálogo Excel para mapear cada imagem a  
  - espécie (`felino` ou `canino`)  
  - tipo de transdutor (`convexo` ou `linear`)  
- Para cada fold (1 a 5):  
  1. Cria pasta `resultado_fold_<n>`  
  2. Avalia o modelo com `model.val(...)` do ultralytics YOLO  
  3. Itera sobre as imagens de teste, realizando:  
     - Normalização do ID da imagem (`normalize_image_id`)  
     - Inferência YOLO para obter bbox e classe com maior confiança  
     - Comparação com label real (se existir) para atualizar contadores  
     - Desenho da imagem com bbox, legenda de classe real e prevista  
     - Salvamento de figura em `resultado_fold_<n>/result_<i>.jpg`  
  4. Calcula métricas de classificação (acurácia, precisão, recall, F1)  
  5. Plota e salva:  
     - Matriz de confusão de classes normal/abnormal (`confusion_matrix_classes.png`)  
     - Barras de acurácia por espécie (`species_accuracy.png`)  
     - Barras de acurácia por transdutor (`transducer_accuracy.png`)  
  6. Gera `metrics.txt` com:  
     - mAP50 e mAP50-95 do YOLO  
     - Métricas de classificação  
     - Estatísticas de acerto por espécie e transdutor  

---

### 🔑 Principais funções

- `load_spreadsheet_data(excel_path)`  
  Lê planilhas GE e Samsung, cria dicionários `image_to_species` e `image_to_transducer`.

- `normalize_image_id(image_name)`  
  Remove sufixos e normaliza o ID para lookup no catálogo.

- `create_incremented_folder(base_path, fold)`  
  Cria `resultado_fold_<fold>` sem sobrescrever se já existir.

- `plot_confusion_matrix(cm, classes, title, filename, normalize=False)`  
  Gera e salva mapa de calor de matriz de confusão.

- `plot_category_accuracy(metrics, category_type, output_folder)`  
  Plota barras de total x acertos para `species` ou `transducer`.

- Loop principal em `__main__`:  
  - Analisa argumentos `--folds_dir`, `--weights_dir`, `--output_dir`  
  - Para cada fold: avaliação, plotagens, salvamento de imagens e métricas.

---

### ⚙️ Parâmetros de entrada

| Argumento     | Tipo | Descrição                                                      |
| ------------- | ---- | -------------------------------------------------------------- |
| `--folds_dir`     | str  | Pasta com subpastas `fold_<n>/test/images`, `test/labels` e `data.yaml` |
| `--weights_dir`   | str  | Pasta com `fold_<n>/weights/best.pt` para cada fold           |
| `--output_dir`    | str  | Pasta onde serão criados `resultado_fold_<n>/`                |

---

### 🚀 Como usar

1. Certifique-se de ter os arquivos:  
   - `<folds_dir>/fold_<n>/test/images` e `.../test/labels`  
   - `<folds_dir>/fold_<n>/data.yaml`  
   - `<weights_dir>/fold_<n>/weights/best.pt`  

2. Execute no terminal:  
   ```bash
   python evaluate_and_plot.py \
     --folds_dir caminho/para/folds \
     --weights_dir caminho/para/weights \
     --output_dir caminho/para/resultados

## 🗂️ Script de Criação de Folds com Crop (kfolds.py)

Este script divide o dataset original em N folds estratificados por grupo, aplica crop automático em cada imagem e ajusta as coordenadas dos bounding boxes.

---

### 📋 O que ele faz

- Carrega o arquivo `data.yaml` para obter configurações gerais.  
- Varre as pastas `train`, `valid` e `test`, coletando pares imagem–label.  
- Agrupa imagens por chave extraída via regex (função `extract_group_key`) para manter série correlacionadas juntas.  
- Embaralha aleatoriamente os grupos e os divide em `num_folds` partes quase iguais.  
- Para cada fold i (1…N):  
  - Define quais grupos vão para treino (3 folds), validação (1 fold) e teste (1 fold).  
  - Cria pasta `folds/fold_i/{train,valid,test}/{images,labels}`.  
  - Em cada par imagem–label, aplica:  
    - `crop_and_find_component()` para recortar ao redor do maior componente  
    - `adjust_coordinates()` para recalcular as coordenadas normalizadas no crop  
    - Salva a imagem recortada e o novo arquivo de label no respectivo split.  
  - Copia o `data.yaml` original para `fold_i/`.  
- Ao final, imprime resumo de quantas imagens em cada split e confirma criação de todos os folds.

---

### 🔑 Principais funções

- `load_yaml(yaml_path)` — lê `data.yaml` e retorna dicionário.  
- `get_image_label_pairs(split_dir)` — retorna lista de tuplas (imagem, label) para um split.  
- `extract_group_key(filename)` — limpa sufixos e retorna chave de agrupamento para manter séries juntas.  
- `crop_and_find_component(image, threshold=5)` — do módulo de augment, encontra maior contorno e recorta.  
- `adjust_coordinates(coords_pairs, orig_size, bbox)` — recalcula coordenadas após o crop.  
- `create_folds(base_dir, num_folds, seed)` — coordena todo o processo de agrupamento, divisão e salvamento.  

---

### ⚙️ Parâmetros de entrada

| Parâmetro    | Tipo | Descrição                                                      |
| ------------ | ---- | -------------------------------------------------------------- |
| base_dir     | str  | Pasta raiz do dataset contendo `train/`, `valid/`, `test/` e `data.yaml` |
| num_folds    | int  | Quantidade de folds a gerar (padrão 5)                         |
| seed         | int  | Semente para embaralhamento e reprodutibilidade (padrão 42)    |

---

### 🚀 Como usar

1. Ajuste o caminho `base_dataset_path` no bloco `if __name__ == '__main__'` ou altere para usar `argparse`.  
2. Execute no terminal:
- python create_folds.py
- ou, se tiver `argparse` integrado:

  3. Após execução, será criada a pasta `folds/` com:
- `fold_1` … `fold_5`  
- Em cada `fold_i`:  
  - `train/images`, `train/labels`  
  - `valid/images`, `valid/labels`  
  - `test/images`,  `test/labels`  
  - cópia de `data.yaml`  

---

💡 **Dica**  
Altere `threshold` em `crop_and_find_component` para ajustar a sensibilidade de recorte, e experimente mudar `num_folds` conforme necessidade de validação cruzada.  


