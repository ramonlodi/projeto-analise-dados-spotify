# Análise de Músicas Populares no Spotify

Análise exploratória, análise estatística e Machine Learning sobre as músicas mais populares do Spotify,
reunidas em um único notebook e um relatório completo.

Trabalho desenvolvido no Instituto Federal de Santa Catarina (IFSC), curso de Sistemas de Informação,
nas disciplinas de Big Data e Probabilidade e Estatística.

**Autor:** Ramon Lodi de Sousa

## O que há aqui

| Parte | Conteúdo | Seções do notebook |
|---|---|---|
| **I. Dados** | Contexto, dataset, carregamento, visão geral e qualidade dos dados | 1–6 |
| **II. Análise exploratória** | Análises univariada, bivariada e multivariada | 7–9 |
| **III. Análise estatística** | Média, mediana, moda, variância, desvio padrão e dados agrupados | 10–12 |
| **IV. Machine Learning** | Classificação do gênero musical com CRISP-DM (4 modelos + baseline) | 13 |
| | Conclusão geral, limitações e extensões | 14 |

O fio condutor: as Partes I–III mostram que as características sonoras quase não explicam a popularidade
entre músicas já populares; a Parte IV pergunta se elas ao menos permitem prever o gênero.

## Principais resultados

- **Perfil predominante:** músicas energéticas, dançantes, ~120 BPM e ~3,5 min; o pop domina os gêneros.
- **Popularidade:** correlação muito fraca com todos os atributos sonoros (entre −0,14 e +0,08).
- **Gênero:** previsível acima do acaso, mas não de forma confiável com ~ 45% de acurácia (Random Forest e
  Regressão Logística, empatados) contra ~ 24% do baseline "sempre pop".
- Erros concentrados em gêneros pequenos; o pop atua como "classe atratora".

Detalhes, gráficos e ressalvas em [`docs/RELATORIO.md`](docs/RELATORIO.md).

## Estrutura do repositório

```
.
├── notebooks/
│   └── analise_spotify.ipynb   # notebook unificado (Partes I a IV)
├── docs/
│   └── RELATORIO.md            # relatório com descrição e análises
├── images/                     # gráficos usados no relatório
├── data/
│   └── README.md               # como obter o CSV (não versionado)
├── requirements.txt
└── README.md
```

## Como executar

**1. Obtenha o dataset.** Baixe em
<https://www.kaggle.com/datasets/solomonameh/spotify-music-dataset> apenas o arquivo
`high_popularity_spotify_data.csv` (1.686 linhas × 29 colunas).

**2. Escolha onde rodar.**

- **Kaggle:** adicione o dataset ao notebook. O caminho é detectado automaticamente.
- **Local:**
  ```bash
  python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
  pip install -r requirements.txt
  # coloque o CSV em data/high_popularity_spotify_data.csv
  jupyter notebook notebooks/analise_spotify.ipynb
  ```
- **Google Colab:** envie o notebook, crie a pasta `data/` no ambiente e envie o CSV para ela antes de rodar.

O notebook procura o CSV em `/kaggle/input/...`, em `data/` e em `../data/`, nessa ordem.

## Observações importantes

- O notebook é entregue sem saídas (limpo). Execute-o para gerar todos os números e gráficos.
- Os valores e gráficos do relatório vêm das execuções originais dos três trabalhos sobre o dataset real.
  Reexecutar com outra versão do scikit-learn pode alterar levemente os números da Parte IV.
- As Partes II–III usam as 1.686 linhas do dataset; a Parte IV usa faixas únicas (evita vazamento de dados),
  o que é explicado na seção 13.1 do notebook.
- A seção 14 do relatório lista as inconsistências encontradas entre os trabalhos e como foram tratadas.

## Fontes

- Dataset: [Spotify Music Dataset — Kaggle](https://www.kaggle.com/datasets/solomonameh/spotify-music-dataset) (arquivo `high_popularity_spotify_data.csv`, atualizado em 2025).
- Vídeo de apresentação (Big Data): <https://www.loom.com/share/4d6563cc3bc14d7fb57818a299ce1074>
