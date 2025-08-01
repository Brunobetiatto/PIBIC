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

## ✨ Script de Aumento de Dados Customizado (arquivo: albumentation_1.py)
Este script gera novas amostras de imagem e labels para treino e validação, aplicando transformações seguras (sem crop que deforme) e mantendo a consistência dos bounding boxes.

📋 O que ele faz
Define a seed para reprodutibilidade (função set_seed).

Recorta cada imagem ao redor do maior componente (linha de interesse) e ajusta as coordenadas originais de caixa.

Aplica transformações opcionais:
• Brilho aleatório
• Flip horizontal
• Rotação suave até ±30°

Gera N cópias por classe até atingir o target definido para treino e validação.

Copia o conjunto de teste sem alterações.

Mantém a mesma estrutura de pastas e replica o data.yaml.

🔑 Principais funções
  - set_seed(seed) – ajusta as sementes de random e numpy para resultados reproduzíveis.
  - crop_and_find_component(imagem, threshold=5) – converte para escala de cinza, aplica threshold, encontra o maior contorno e retorna a caixa delimitadora e a imagem recortada.
  - adjust_coordinates(pares_de_coords, tamanho_original, bbox) – transforma coordenadas normalizadas da imagem original para a região recortada.
  - apply_random_brightness(imagem) – ajusta o brilho via HSV (com probabilidade interna).
  - apply_random_flip(imagem, pares_de_coords) – inverte horizontalmente e atualiza as coordenadas X.
  - rotate_image(imagem, pares_de_coords) – rotaciona suavemente e recalcula as coordenadas, descartando rotações que saiam do frame.
  - process_directory(…) – lê imagens e labels das pastas train e valid, calcula quantas cópias gerar por imagem e salva as imagens aumentadas e novos arquivos de label.
  - copy_test_images(…) – replica as pastas test/images e test/labels sem modificações.
  - main() – lê argumentos via argparse, cria uma nova pasta fold_X em output_dir, chama process_directory e copy_test_images e copia o data.yaml para o novo fold.

⚙️ Parâmetros de entrada
| Argumento                   | Tipo | Descrição                                                        |
| --------------------------- | ---- | ---------------------------------------------------------------- |
| input\_fold                 | str  | Pasta do fold original com subpastas train, valid e test         |
| output\_dir                 | str  | Pasta onde o novo fold será criado (ex: augmented\_folds/)       |
| --target\_per\_class\_train | int  | Quantidade de imagens desejada por classe em train (padrão: 500) |
| --target\_per\_class\_valid | int  | Quantidade de imagens desejada por classe em valid (padrão: 100) |
| --seed                      | int  | Semente para reprodutibilidade (opcional)                        |


#🚀 Como utilizar
Preparar as pastas de entrada:
• fold original deve conter subpastas train/images, train/labels, valid/images, valid/labels, test/images, test/labels e um arquivo data.yaml.

#Executar o comando:
python albumentation_1.py <caminho_para_fold_original> <caminho_para_folds_aumentados> --target_per_class_train 500 --target_per_class_valid 100 --seed 42

O script criará uma pasta fold_1 (ou próxima disponível) em <caminho_para_folds_aumentados> com a mesma estrutura, contendo imagens originais e aumentadas, labels ajustadas e o data.yaml copiado.

#💡 Dica rápida
Ajuste o threshold em crop_and_find_component para recortes mais finos, ou altere as probabilidades internas (valores em if random.random() < ...) para controlar com que frequência cada transformação ocorre.

<!-- Nas próximas seções você verá: 
- Descrição do projeto  
- Estrutura do repositório  
- Como rodar os notebooks e scripts  
- Principais resultados e gráficos  
- Tecnologias utilizadas  
- Contato e referências  
-->
