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
  Processa os resultados de inferência em arquivos de métricas brutas (`metrics.txt`) e prepara dados para análise.

- **Métri­cas/**  
  Armazena o CSV final com estatísticas agregadas (precisão, recall, F1-score, mAP) e pastas com gráficos que ilustram o desempenho por categoria e experimento.

---

_Nas próximas seções, veremos como rodar cada script passo a passo e como interpretar os resultados gerados._  


<!-- Nas próximas seções você verá: 
- Descrição do projeto  
- Estrutura do repositório  
- Como rodar os notebooks e scripts  
- Principais resultados e gráficos  
- Tecnologias utilizadas  
- Contato e referências  
-->
