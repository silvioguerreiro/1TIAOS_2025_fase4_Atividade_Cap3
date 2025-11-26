# FIAP - Faculdade de Informática e Administração Paulista


<a href= "https://www.fiap.com.br/"><img width="2385" height="642" alt="logo-fiap" src="https://github.com/user-attachments/assets/62285a6c-34fe-4206-8a85-7ad584c6908b" alt="FIAP - Faculdade de Informática e Admnistração Paulista" border="0" width=40% height=40%></a>
</p>

<br>


# 📊 Fase 4 – 1TIAOS – Capítulo 3

## **(IR ALÉM) Implementando Algoritmos de Machine Learning com Scikit-learn**

### 👨‍💻 Aluno

* [**Silvio Prestes Guerreiro Junior**](https://www.linkedin.com/in/silvio-guerreiro-junior/)
* **Matrícula:** RM567958
* **Grupo 25**

### 👩‍🏫 Professores

* **Tutor(a):** Sabrina Otoni
* **Coordenador(a):** André Godoi Chiovato

# Projeto Seeds – Classificação Automática de Grãos de Trigo

Este repositório implementa a atividade **“Da Terra ao Código: Automatizando a Classificação de Grãos com Machine Learning”** da FIAP (Fase 4 – Capítulo 3, IR ALÉM).

O objetivo é aplicar a metodologia **CRISP‑DM** para desenvolver um modelo de aprendizado de máquina capaz de **classificar automaticamente grãos de trigo** em três variedades (Kama, Rosa e Canadian) a partir de suas características físicas, substituindo (ou complementando) a triagem manual feita por especialistas.

---

## 🧭 Índice

- [Contexto do Problema](#-contexto-do-problema)
- [Dataset Utilizado](#-dataset-utilizado)
- [Metodologia e Organização do Notebook](#-metodologia-e-organização-do-notebook)
- [Modelos de Machine Learning](#-modelos-de-machine-learning)
- [Conclusões e Relatório Executivo](#-conclusões-e-relatório-executivo)
- [Estrutura de Pastas](#-estrutura-de-pastas)
- [Como Executar o Código](#-como-executar-o-código)
- [Dependências e Tecnologias](#-dependências-e-tecnologias)
- [Histórico de Lançamentos](#-histórico-de-lançamentos)
- [Licença](#-licença)
- [Autor](#-autor)

---

## 🌾 Contexto do Problema

Em cooperativas agrícolas de pequeno porte, a classificação dos grãos de trigo costuma ser:

- manual  
- demorada  
- sujeita a erro humano  
- difícil de padronizar e escalar

A proposta deste projeto é **automatizar a classificação de variedades de trigo** usando *Machine Learning*, apoiando a cooperativa fictícia **Farm Tech Solutions** na tomada de decisão: armazenagem, precificação e atendimento a especificações de clientes.

---

## 📊 Dataset Utilizado

O projeto utiliza o **Seeds Dataset** (UCI Machine Learning Repository), disponibilizado neste repositório como `seeds_dataset.txt`. :contentReference[oaicite:1]{index=1}  

Características:

- **210 amostras** de grãos de trigo  
- **3 classes (variedades)**:
  - 1 – Kama  
  - 2 – Rosa  
  - 3 – Canadian  
- **7 atributos numéricos (medidas geométricas)**:
  1. Área  
  2. Perímetro  
  3. Compacidade  
  4. Comprimento do núcleo  
  5. Largura do núcleo  
  6. Assimetria  
  7. Comprimento do sulco do núcleo (*groove_length*)

A última coluna representa a **classe** (variedade do trigo).

---

## 🔄 Metodologia e Organização do Notebook

Todo o desenvolvimento está concentrado em:

- `Seeds_Notebook.ipynb` – notebook principal com a solução completa.

O notebook foi estruturado seguindo o **CRISP‑DM**:

1. **Entendimento do Negócio**  
   - Cenário da cooperativa Farm Tech Solutions  
   - Problemas da classificação manual  
   - Objetivo de automação e métricas-alvo

2. **Entendimento e Preparação dos Dados**  
   - Carregamento do `seeds_dataset.txt`  
   - Renomeação das colunas  
   - Estatísticas descritivas (`describe()`)  
   - Verificação de valores ausentes  
   - Visualizações:
     - histogramas  
     - boxplots  
     - scatter plots  
     - matriz de correlação  
   - Discussão e aplicação de **padronização** (`StandardScaler`) para modelos sensíveis à escala.

3. **Modelagem Básica**  
   - Divisão treino/teste (70% / 30%)  
   - Treinamento inicial de:
     - **K‑Nearest Neighbors (KNN)**
     - **Support Vector Machine (SVM)**
     - **Random Forest**
   - Avaliação com:
     - acurácia  
     - precision, recall, F1‑score  
     - matriz de confusão para cada modelo

4. **Otimização de Hiperparâmetros**  
   - Uso de **GridSearchCV** para buscar a melhor configuração de hiperparâmetros  
   - Re-treino do modelo campeão com os parâmetros otimizados  
   - Reavaliação em teste

5. **Modelagem Avançada**  
   - Criação de **pipelines** (pré-processamento + modelo)  
   - **Validação cruzada (5-fold)** para comparação robusta entre algoritmos  
   - Aplicação de **PCA** para visualização em 2D da separação entre classes

6. **Conclusões e Relatório Executivo**  
   - Síntese dos resultados técnicos  
   - Interpretação em linguagem de negócio  
   - Recomendações práticas para a cooperativa

---

## 🤖 Modelos de Machine Learning

Três algoritmos supervisionados de classificação foram treinados e comparados:

- **KNN (K-Nearest Neighbors)**  
  Classifica um grão pela maioria dos vizinhos mais próximos no espaço de atributos.

- **SVM (Support Vector Machine)**  
  Busca hiperplanos de decisão que maximizam a margem entre as classes.

- **Random Forest**  
  Conjunto de árvores de decisão que captura relações não lineares e permite medir a importância das features.

Todos os modelos foram avaliados com:

- Acurácia  
- Precision, Recall e F1‑score (médias ponderadas)  
- Matrizes de confusão  
- Validação cruzada (5-fold) com pipelines

---

## 📈 Conclusões e Relatório Executivo

### 1. Síntese do Experimento

- **Problema:** automatizar a classificação de grãos de trigo para reduzir esforço humano e padronizar decisões.  
- **Dados:** 210 amostras, 7 atributos geométricos, 3 variedades (Kama, Rosa, Canadian).  
- **Abordagem:** CRISP‑DM, com forte foco em EDA, comparação de algoritmos e otimização.  
- **Modelos testados:** KNN, SVM e Random Forest.  
- **Avaliação:** conjunto de teste separado (30% do dataset) + validação cruzada (5-fold) com pipelines.

### 2. Desempenho dos Modelos

Após a fase de validação e otimização:

- 🥇 **Modelo Campeão: SVM (Support Vector Machine)**  
  - **Acurácia média em validação cruzada:** ~**93,20%**  
  - **Acurácia no conjunto de teste (dados inéditos):** ~**87,30%**

Esses resultados indicam um modelo:

- estável (boa performance em diferentes folds),  
- com boa capacidade de generalização,  
- adequado para substituir parte da classificação manual feita por especialistas.

### 3. Comportamento por Variedade (Matriz de Confusão)

No conjunto de teste (63 amostras), a matriz de confusão do SVM mostra:

- **Canadian (Classe 3):**  
  - ~**95% de acerto** (20/21 amostras corretamente classificadas)  
  - Variedade mais fácil para o modelo.

- **Rosa (Classe 2):**  
  - ~**90% de acerto** (19/21 amostras)  
  - Modelo bastante confiável.

- **Kama (Classe 1):**  
  - ~**76% de acerto** (16/21 amostras)  
  - Maior fonte de erro: parte dos grãos Kama é confundida com Rosa ou Canadian.

> **Insight:** a variedade **Kama** tem características geométricas mais “intermediárias” ou maior variabilidade interna, o que torna sua classificação mais difícil e exige atenção especial em produção.

### 4. Insights sobre as Características

A análise de importância das features (Random Forest) e os gráficos derivados do PCA indicam:

- **Atributos mais importantes:**
  - **Comprimento do Sulco do Núcleo (*groove_length*)**  
  - **Perímetro**  
  - **Área**
- **Atributos menos relevantes:**
  - **Compacidade**  
  - **Assimetria**

Implica em termos de negócio e engenharia:

- Sensores/câmeras devem ser escolhidos e calibrados para medir com **alta precisão**:
  - o contorno do grão,  
  - a área projetada,  
  - e a geometria do sulco.

- Em cenários com limitação de hardware, compacidade e assimetria podem ter prioridade menor.

### 5. Avaliação de Viabilidade

**Pergunta central:**  
> “É viável substituir, ao menos parcialmente, a classificação manual por um modelo de IA confiável?”

**Resposta:**  
✅ **Sim, a automação é tecnicamente viável e recomendável.**

O modelo SVM:

- mantém acurácia próxima a 90%;  
- reduz variabilidade entre avaliadores humanos;  
- é rápido o suficiente para operar em linhas de triagem em tempo quase real.

### 6. Recomendações para a Farm Tech Solutions

1. **Automação Parcial Imediata**
   - Utilizar o modelo SVM como primeira etapa de decisão para todas as variedades.  
   - Encaminhar apenas casos de **baixa confiança** ou amostras classificadas como **Kama** para revisão manual.

2. **Investimento em Hardware**
   - Priorizar sensores de visão/câmeras que permitam extrair:
     - área,  
     - perímetro,  
     - comprimento do sulco.  
   - Avaliar trade‑offs de custo versus resolução necessária.

3. **Fluxo Operacional Sugerido**
   1. Grão passa por câmera/sensor em esteira.  
   2. Imagem é processada para extrair features geométricas.  
   3. Features são enviadas ao modelo SVM.  
   4. Modelo retorna:
      - classe prevista (Kama, Rosa, Canadian);  
      - score de confiança.  
   5. Grãos com baixa confiança ou Kama são rotulados para conferência manual.

---

## 📁 Estrutura de Pastas

```text
FASE4_ATIVIDADE_CAP3/
│
├─ .venv/                  # Ambiente virtual Python 
├─ requirements.txt        # Lista de dependências do projeto
├─ seeds_dataset.txt       # Seeds Dataset (dados de entrada)
└─ Seeds_Notebook.ipynb    # Notebook Jupyter com toda a análise e modelagem
````

---

## ⚙️ Como Executar o Código

### 1. Clonar o repositório

```bash
git clone https://github.com/SEU-USUARIO/FASE4_ATIVIDADE_CAP3.git
cd FASE4_ATIVIDADE_CAP3
```

### 2. (Opcional, mas recomendado) Criar e ativar o ambiente virtual

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

### 3. Instalar as dependências

```bash
pip install -r requirements.txt
```

### 4. Abrir o notebook

Usando Jupyter:

```bash
jupyter notebook Seeds_Notebook.ipynb
```

Ou diretamente pelo **VS Code**, com a extensão *Jupyter* instalada, abrindo o arquivo `Seeds_Notebook.ipynb` e executando as células na ordem.

---

## 🧩 Dependências e Tecnologias

Principais ferramentas e bibliotecas utilizadas (versões especificadas em `requirements.txt`):

* **Python 3.x**
* **Jupyter / VS Code (Jupyter extension)**
* `pandas`, `numpy` – manipulação e análise de dados
* `scikit-learn` – modelos de Machine Learning (KNN, SVM, Random Forest, GridSearchCV, pipelines, validação cruzada) 
* `matplotlib`, `seaborn` – visualização de dados
* Outras libs de apoio descritas em `requirements.txt` (ipykernel, scipy etc.). 

---

## 📝 Histórico de Lançamentos

* **v1.0.0 – Entrega FIAP (Capítulo 3 – IR ALÉM)**

  * Notebook `Seeds_Notebook.ipynb` finalizado
  * EDA completa (gráficos, estatísticas, correlação)
  * Implementação de KNN, SVM e Random Forest
  * Validação cruzada + GridSearchCV
  * Conclusões e relatório executivo


---

## 📄 Licença

Este projeto foi desenvolvido exclusivamente para fins acadêmicos – FIAP. Qualquer uso, modificação ou redistribuição deve seguir as diretrizes institucionais e de propriedade intelectual aplicáveis.

---

```
