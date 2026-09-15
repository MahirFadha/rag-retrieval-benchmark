# Evaluasi Komparatif BM25, Multilingual-E5, dan Hybrid RRF pada Retrieval-Augmented Generation

Repository ini berisi dataset benchmark, implementasi metode retrieval, konfigurasi eksperimen, pipeline Retrieval-Augmented Generation (RAG), hasil evaluasi, dan analisis statistik yang digunakan dalam penelitian:

> **Evaluasi Komparatif BM25, Multilingual-E5, dan Hybrid RRF pada Retrieval-Augmented Generation untuk Katalog Produk dan Layanan**

Penelitian membandingkan tiga pendekatan retrieval:

1. **BM25** sebagai lexical retrieval.
2. **Multilingual-E5** sebagai dense semantic retrieval.
3. **Hybrid Retrieval** menggunakan Reciprocal Rank Fusion (RRF).

Evaluasi dilakukan pada dua tingkat, yaitu **retrieval performance** dan pengaruh metode retrieval terhadap **kualitas jawaban RAG**.

---

## 1. Gambaran Penelitian

Eksperimen menggunakan katalog produk dan layanan sebagai knowledge base untuk sistem Retrieval-Augmented Generation.

Alur utama penelitian adalah:

```text
Data Katalog
    ↓
Canonical Corpus
    ↓
Benchmark Queries + Relevance Judgment
    ↓
┌──────────────┬─────────────────┐
│     BM25     │ Multilingual-E5 │
└──────────────┴─────────────────┘
          ↓
     Hybrid RRF
          ↓
   Retrieval Ranking
          ↓
 Retrieval Evaluation
          ↓
 Top-5 Context per Method
          ↓
   RAG Generation
          ↓
    RAG Evaluation
          ↓
 Statistical Analysis
```

Seluruh metode retrieval menggunakan corpus dan query benchmark yang sama.

Pada tahap RAG, setiap metode menggunakan generator, system prompt, dan konfigurasi generasi yang sama sehingga perbedaan konteks retrieval menjadi variabel utama yang dibandingkan.

---

## 2. Dataset

Benchmark terdiri dari:

- **145 dokumen katalog**
- **40 benchmark queries**
- **1 relevant document untuk setiap query**
- Binary relevance judgment

### Distribusi dokumen

| Domain     |  Jumlah |
| ---------- | ------: |
| Produk AC  |      53 |
| Layanan AC |      71 |
| CCTV       |      21 |
| **Total**  | **145** |

### Kategori query

Sebanyak 40 query dibagi secara seimbang menjadi empat kategori:

| Kategori         | Jumlah |
| ---------------- | -----: |
| Exact            |     10 |
| Semantic         |     10 |
| Multi-constraint |     10 |
| Fine-grained     |     10 |
| **Total**        | **40** |

Distribusi domain query:

| Domain     | Jumlah |
| ---------- | -----: |
| Product AC |     12 |
| Service AC |     16 |
| CCTV       |     12 |
| **Total**  | **40** |

Setiap query memiliki tepat satu dokumen relevan. Karena itu, dalam desain benchmark ini nilai `Recall@K` secara numerik sama dengan `Hit@K`.

### Sumber Data

Dataset diproses dari informasi katalog produk dan layanan yang tersedia secara publik pada website resmi perusahaan.

**Sumber:**  
TODO: tambahkan URL website resmi perusahaan

Dataset pada repository merupakan data yang telah diproses menjadi canonical corpus untuk kebutuhan eksperimen retrieval dan RAG.

> Catatan: ketersediaan informasi pada website publik tidak selalu identik dengan lisensi open data. Pengguna repository tetap disarankan memperhatikan ketentuan penggunaan dari sumber data asli.

---

## 3. Metode Retrieval

### 3.1 BM25

Lexical retrieval menggunakan implementasi `BM25Okapi` dari:

```text
rank-bm25==0.2.2
```

Konfigurasi utama:

```text
k1      = 1.5
b       = 0.75
epsilon = 0.25
```

Preprocessing teks mencakup normalisasi Unicode, lowercase, serta tokenisasi berbasis pola alfanumerik.

Tidak digunakan:

- stemming,
- stopword removal,
- synonym expansion,
- query boosting,
- metadata filtering.

---

### 3.2 Multilingual-E5

Dense semantic retrieval menggunakan model:

```text
intfloat/multilingual-e5-base
```

Representasi input menggunakan prefix:

```text
query:
passage:
```

Embedding diperoleh menggunakan:

- mean pooling,
- L2 normalization,
- cosine similarity.

Seluruh embedding dokumen dibandingkan secara **exact similarity search** tanpa Approximate Nearest Neighbor (ANN) atau vector database.

Seluruh dokumen pada canonical corpus berada di bawah batas 512 token model. Panjang maksimum dokumen setelah representasi final adalah **477 token**.

---

### 3.3 Hybrid Retrieval

Hybrid retrieval menggabungkan ranking BM25 dan multilingual-E5 menggunakan **Reciprocal Rank Fusion (RRF)**.

Formula:

```text
RRF(d) =
    1 / (K + rank_BM25(d))
    +
    1 / (K + rank_E5(d))
```

dengan:

```text
K = 60
BM25 weight = 1
E5 weight   = 1
```

Fusion dilakukan terhadap full ranking seluruh corpus.

Tidak digunakan additional reranker.

---

## 4. Retrieval Evaluation

Retrieval dievaluasi menggunakan:

- Hit@1
- Hit@3
- Hit@5
- Recall@1
- Recall@3
- Recall@5
- MRR@5
- Mean Target Rank

### Hasil Retrieval Keseluruhan

| Method          |     Hit@1 |     Hit@3 |     Hit@5 |      MRR@5 | Mean Target Rank |
| --------------- | --------: | --------: | --------: | ---------: | ---------------: |
| BM25            | **0.775** |     0.875 |     0.925 | **0.8279** |             2.45 |
| Multilingual-E5 |     0.650 | **0.900** | **0.950** |     0.7737 |             1.95 |
| Hybrid RRF      |     0.725 | **0.900** |     0.925 |     0.8133 |         **1.75** |

Secara deskriptif:

- BM25 memberikan performa ranking awal terbaik.
- Multilingual-E5 memberikan cakupan Top-5 tertinggi.
- Hybrid RRF menghasilkan mean target rank terbaik.

Tidak ditemukan satu metode yang superior pada seluruh metrik.

---

## 5. RAG Evaluation

Setiap metode retrieval mengambil **Top-5 documents** sebagai konteks untuk RAG.

Generator menggunakan konfigurasi yang sama untuk seluruh metode sehingga eksperimen membandingkan efek konteks retrieval, bukan perbedaan model generator.

### Generator

Model:

```text
Gemini 3.6 Flash
```

System prompt dan konfigurasi generator dapat dilihat pada:

```text
benchmark/config/
```

Total jawaban yang dihasilkan:

```text
40 queries × 3 retrieval methods = 120 RAG responses
```

---

## 6. Evaluasi Jawaban RAG

Evaluasi dilakukan menggunakan tiga dimensi:

- **Faithfulness**
- **Answer Relevancy**
- **Answer Accuracy**

Pipeline evaluasi menggunakan Ragas dan LLM-as-a-judge dengan konfigurasi evaluator yang tersedia pada:

```text
benchmark/config/
```

### Hasil RAG Keseluruhan

| Method          | Faithfulness | Answer Relevancy | Answer Accuracy |
| --------------- | -----------: | ---------------: | --------------: |
| BM25            |       0.9380 |           0.8418 |          0.7813 |
| Multilingual-E5 |   **0.9667** |           0.8446 |      **0.7938** |
| Hybrid RRF      |       0.9647 |       **0.8451** |      **0.7938** |

Nilai tertinggi pada tabel merupakan hasil deskriptif dan tidak secara otomatis menunjukkan perbedaan yang signifikan secara statistik.

---

## 7. Statistical Analysis

Untuk membandingkan performa metode retrieval digunakan:

- Cochran's Q
- Friedman Test
- Wilcoxon Signed-Rank Test
- McNemar Test
- Holm correction
- Effect size

Untuk evaluasi RAG digunakan:

- descriptive statistics,
- Friedman Test,
- pairwise Wilcoxon Signed-Rank Test,
- Holm correction,
- Kendall's W,
- rank-biserial correlation.

Analisis tambahan dilakukan berdasarkan:

- query category,
- query domain,
- target presence in Top-5,
- target rank,
- diagnostic cases.

Pada eksperimen ini tidak ditemukan perbedaan signifikan secara statistik antara ketiga metode pada metrik retrieval utama maupun kualitas jawaban RAG.

---

## 8. Struktur Repository

```text
.
├── benchmark/
│   ├── config/
│   │   ├── rag_evaluator_config_v1.json
│   │   ├── rag_generator_config_v2.json
│   │   └── rag_system_prompt_v1.txt
│   │
│   ├── data/
│   │   ├── canonical/
│   │   ├── evaluation/
│   │   └── rag/
│   │
│   ├── generation/
│   │   └── gemini_generator.py
│   │
│   ├── retrieval/
│   │   ├── bm25_retriever.py
│   │   ├── e5_retriever.py
│   │   └── hybrid_retriever.py
│   │
│   ├── scripts/
│   │
│   └── results/
│       ├── analysis/
│       │   ├── diagnostic_cases/
│       │   ├── retrieval_statistics/
│       │   ├── statistics/
│       │   └── stratified/
│       ├── evaluation/
│       └── rag/
│
├── .gitignore
├── README.md
└── requirements.txt
```

### Folder utama

`benchmark/config/`  
Berisi konfigurasi generator, evaluator, dan system prompt yang digunakan dalam eksperimen.

`benchmark/data/canonical/`  
Berisi canonical corpus serta benchmark queries dan relevance judgment.

`benchmark/data/rag/`  
Berisi input yang dibentuk dari hasil retrieval untuk tahap RAG.

`benchmark/data/evaluation/`  
Berisi dataset dan artefak untuk evaluasi jawaban RAG.

`benchmark/retrieval/`  
Implementasi BM25, multilingual-E5, dan Hybrid RRF.

`benchmark/generation/`  
Implementasi generator RAG.

`benchmark/scripts/`  
Script untuk menjalankan pipeline eksperimen, validasi artefak, evaluasi, dan analisis statistik.

`benchmark/results/`  
Berisi hasil retrieval, hasil generasi/evaluasi RAG, serta analisis statistik.

---

## 9. Instalasi

### Clone repository

```bash
git clone https://github.com/MahirFadha/TODO-NAMA-REPOSITORY.git
cd TODO-NAMA-REPOSITORY
```

### Membuat virtual environment

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

Linux/macOS:

```bash
source venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## 10. Environment Variables

API key tidak disimpan dalam repository.

Buat file:

```text
.env
```

pada root project apabila diperlukan oleh generator/evaluator.

Contoh:

```env
GOOGLE_API_KEY=YOUR_API_KEY
```

File `.env` telah dimasukkan ke `.gitignore` dan **tidak boleh di-commit ke repository**.

---

## 11. Menjalankan Eksperimen

### Retrieval Benchmark

```bash
python -m benchmark.scripts.run_retrieval_benchmark
```

Tahap ini menjalankan:

```text
BM25
Multilingual-E5
Hybrid RRF
```

untuk seluruh benchmark queries dan menghasilkan retrieval rankings serta retrieval metrics.

### RAG Generation

Input RAG dibentuk dari Top-5 hasil retrieval:

```bash
python -m benchmark.scripts.build_rag_generation_inputs
```

Kemudian jawaban RAG dihasilkan menggunakan:

```bash
python -m benchmark.scripts.run_rag_generation
```

### RAG Evaluation

Dataset evaluasi dibentuk dari hasil generasi dan reference answer.

Evaluasi akhir dijalankan melalui:

```bash
python -m benchmark.scripts.run_rag_evaluation_final_v1
```

### Statistical Analysis

Analisis retrieval:

```bash
python -m benchmark.scripts.analyze_retrieval_statistics_v1
```

Analisis RAG:

```bash
python -m benchmark.scripts.analyze_rag_statistics_v1
```

Stratified analysis:

```bash
python -m benchmark.scripts.analyze_rag_stratified_v1
```

Diagnostic analysis:

```bash
python -m benchmark.scripts.analyze_diagnostic_cases_v1
```

---

## 12. Reproducibility

Eksperimen menggunakan mekanisme frozen artifacts untuk menjaga konsistensi antara:

- canonical corpus,
- benchmark query set,
- retrieval results,
- RAG generation inputs,
- generated responses,
- reference answers,
- evaluation dataset,
- evaluator configuration,
- evaluation results.

Beberapa artefak dilengkapi dengan SHA-256 checksum sehingga perubahan terhadap data atau konfigurasi dapat dideteksi.

Tujuannya adalah memastikan bahwa seluruh tahap analisis berasal dari artefak eksperimen yang sama dan tidak berubah setelah proses evaluasi dimulai.

---

## 13. Data and Code Availability

Dataset benchmark, kode implementasi, konfigurasi eksperimen, hasil retrieval, output RAG, hasil evaluasi, dan analisis statistik yang digunakan dalam penelitian tersedia pada repository ini.

Data katalog pada benchmark berasal dari informasi produk dan layanan yang dapat diakses secara publik melalui website resmi perusahaan dan telah diproses menjadi representasi yang digunakan dalam eksperimen.

Repository ini juga menyediakan artefak yang diperlukan untuk menelusuri alur eksperimen dari canonical corpus hingga hasil analisis statistik.

---

## 14. Research Scope and Limitations

Benchmark terdiri dari 145 dokumen dan bersifat domain-specific pada katalog produk dan layanan AC serta CCTV.

Oleh karena itu, hasil penelitian menggambarkan karakteristik ketiga metode retrieval pada corpus dan konfigurasi yang diuji dan tidak dimaksudkan untuk menunjukkan bahwa suatu metode selalu lebih baik pada seluruh domain atau dataset.

Selain itu:

- setiap query memiliki satu relevant document;
- corpus relatif kecil;
- dense retrieval menggunakan exact cosine similarity;
- RAG menggunakan Top-5 retrieved documents;
- tidak digunakan reranker;
- evaluasi utama menggunakan automatic evaluation.

Penelitian selanjutnya dapat memperluas ukuran corpus dan query, menggunakan multiple relevant documents atau graded relevance, menguji variasi Top-K, menambahkan reranking, serta melakukan human evaluation.

---
