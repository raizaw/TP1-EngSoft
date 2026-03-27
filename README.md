# [SampaImóveis Predictor: Análise e Previsão de Valor Venal em São Paulo]

**Uma prova de conceito de pipeline de Ciência de Dados utilizando dados reais do GeoSampa.**

## Objetivos e Principais Hipóteses

[Utilizando os dados abertos do GeoSampa da Prefeitura de São Paulo, este projeto visa desenvolver um pipeline end-to-end de Ciência de Dados para analisar e prever o valor venal por metro quadrado de imóveis na capital. O objetivo é criar uma ferramenta de apoio à decisão para o mercado imobiliário, validando se variáveis urbanas e socioeconômicas conseguem explicar variações de preços em dados públicos reais. O projeto simula um cenário de demanda real, exercitando práticas ágeis e construção de portfólio técnico em equipe. 

A viabilidade do modelo será testada através das seguintes hipóteses, vinculadas ao backlog da sprint:
* (1) A proximidade a estações de transporte público e grandes áreas verdes impacta positivamente e quantificavelmente o valor por metro quadrado;
* (2) O tipo de zoneamento urbano exerce uma influência significativa e mensurável na valorização imobiliária;
* (3) A integração de características estruturais dos lotes com variáveis socioeconômicas da região (como indicadores sociodemográficos do setor censitário) permite gerar predições com acurácia satisfatória.
* (4) A presença de infraestruturas urbanas de grande impacto, como linhas de alta tensão ou torres de transmissão, exerce uma influência negativa e detectável no valor venal de lotes adjacentes.

O resultado final será um dashboard interativo para visualização das análises e previsões do modelo, simulando uma demanda real do mercado imobiliário urbano.]


## Dataset Escolhido

Para cumprir o requisito de utilizar dados reais, selecionamos o seguinte conjunto:

* **Origem:** [GeoSampa (Portal de Dados Abertos da Prefeitura de São Paulo)](http://geosampa.prefeitura.sp.gov.br/).
* **Tamanho Estimado:** [Milhões de registros de lotes (o tamanho exato depende do recorte temporal e geográfico, mas será um dataset robusto, exigindo técnicas de chunking ou processamento eficiente em memória, como Pandas/Polars)].
* **Quantidade de Features (Candidatas):** [Prevemos o uso de 15 a 25 features em potencial, incluindo características estruturais (área do lote, ano de construção), dados de localização (coordenadas geográficas, bairro), tipo de zoneamento, e variáveis derivadas de proximidade a pontos de interesse (distância a metrô, parques, etc.)].


## Membros da Equipe e Papéis

Nossa equipe é composta por quatro estudantes, com os seguintes papéis definidos para a simulação ágil do projeto:

* **[Felipe Damasceno]:** [Coleta e Pré-processamento de Dados] - *Responsável pela extração dos dados do GeoSampa, tratamento de dados nulos/inválidos, e engenharia de features básicas.* (Papel: Data)
* **[Luís Henrique Emediato]:** [Engenharia de Features e Modelagem] - *Responsável por criar variáveis geoespaciais avançadas (ex: distâncias), selecionar features, treinar e otimizar os modelos de ML.* (Papel: Model)
* **[Mateus Antinossi]:** [Avaliação e Análise de Métricas] - *Responsável por definir e medir as métricas de performance (ex: MAE, RMSE, R² para regressão), validar o modelo e analisar o overfitting.* (Papel: Eval)
* **[Raiza Wunsch]:** [Desenvolvimento do Dashboard e Visualização] - *Responsável por integrar as predições do modelo em um dashboard interativo e criar as visualizações de dados para a review da sprint.* (Papel: Visualization)


## Pilha de Tecnologias

Para a implementação do pipeline, utilizaremos as seguintes tecnologias:

* **Gerenciamento de Projeto:** GitHub (incluindo GitHub Projects para o backlog).
* **Linguagem Principal:** Python.
* **Manipulação de Dados e Geo:** Pandas, NumPy, [Geopandas (para arquivos .shp do GeoSampa)].
* **Machine Learning:** Scikit-learn, [a considerar: XGBoost ou LightGBM para maior performance em dados tabulares].
* **Visualização:** Matplotlib, Seaborn, Plotly.
* **Dashboarding:** [Streamlit] ou [Dash].
* **[Opcional] Outros:** [Docker (para reprodutibilidade do pipeline)].

