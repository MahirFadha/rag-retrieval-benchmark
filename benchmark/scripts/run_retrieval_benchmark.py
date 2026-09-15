import csv
import hashlib
import json
import platform
from collections import defaultdict
from importlib.metadata import version
from pathlib import Path
from statistics import mean

import torch
import transformers

from benchmark.retrieval.bm25_retriever import (
    BM25Retriever,
)

from benchmark.retrieval.e5_retriever import (
    E5Retriever,
)


# ============================================================
# 1. CONFIG
# ============================================================

CORPUS_FILE = Path(
    "benchmark/data/canonical/"
    "aire_catalog_final_v1.jsonl"
)

QUERY_FILE = Path(
    "benchmark/data/canonical/"
    "aire_benchmark_queries_v1.jsonl"
)

RESULT_DIR = Path(
    "benchmark/results"
)


RANKINGS_FILE = (
    RESULT_DIR
    / "retrieval_rankings_v1.csv"
)

QUERY_METRICS_FILE = (
    RESULT_DIR
    / "retrieval_query_metrics_v1.csv"
)

OVERALL_METRICS_FILE = (
    RESULT_DIR
    / "retrieval_metrics_overall_v1.csv"
)

CATEGORY_METRICS_FILE = (
    RESULT_DIR
    / "retrieval_metrics_by_category_v1.csv"
)

DOMAIN_METRICS_FILE = (
    RESULT_DIR
    / "retrieval_metrics_by_domain_v1.csv"
)

METADATA_FILE = (
    RESULT_DIR
    / "retrieval_benchmark_metadata_v1.json"
)


EXPECTED_CORPUS_SHA256 = (
    "7e5c041917e43b32298ecf7dc02b571e"
    "11ffe6cf5c77b52dcf8b531d6b4609be"
)

EXPECTED_QUERY_SHA256 = (
    "dfbf771a3f7dc00d465be4e2605f9cca"
    "e7b0e2147933954d367cfcd0751bfb0d"
)


EXPECTED_DOCUMENTS = 145
EXPECTED_QUERIES = 40


TOP_K_VALUES = (
    1,
    3,
    5,
)

MAX_OUTPUT_K = 5

RRF_K = 60


# ============================================================
# 2. SHA256
# ============================================================

def sha256_file(
    path: Path,
) -> str:

    hasher = hashlib.sha256()

    with path.open(
        "rb"
    ) as file:

        while True:

            chunk = file.read(
                1024 * 1024
            )

            if not chunk:
                break

            hasher.update(
                chunk
            )

    return hasher.hexdigest()


# ============================================================
# 3. VERIFY FROZEN ARTIFACTS
# ============================================================

def verify_frozen_artifacts():

    print(
        "Memverifikasi frozen artifacts..."
    )

    corpus_hash = sha256_file(
        CORPUS_FILE
    )

    query_hash = sha256_file(
        QUERY_FILE
    )

    if (
        corpus_hash
        != EXPECTED_CORPUS_SHA256
    ):
        raise RuntimeError(
            "SHA256 corpus berubah.\n"
            f"Expected: "
            f"{EXPECTED_CORPUS_SHA256}\n"
            f"Actual  : {corpus_hash}"
        )

    if (
        query_hash
        != EXPECTED_QUERY_SHA256
    ):
        raise RuntimeError(
            "SHA256 query set berubah.\n"
            f"Expected: "
            f"{EXPECTED_QUERY_SHA256}\n"
            f"Actual  : {query_hash}"
        )

    print(
        "✅ Corpus SHA256 valid."
    )

    print(
        "✅ Query SHA256 valid."
    )

    return (
        corpus_hash,
        query_hash,
    )


# ============================================================
# 4. LOAD QUERY SET
# ============================================================

def load_queries():

    queries = []

    with QUERY_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            query = json.loads(
                line
            )

            queries.append(
                query
            )

    return queries


# ============================================================
# 5. VALIDATE BENCHMARK
# ============================================================

def validate_benchmark(
    queries,
    corpus_records,
):

    if (
        len(corpus_records)
        != EXPECTED_DOCUMENTS
    ):
        raise RuntimeError(
            "Jumlah corpus tidak sesuai.\n"
            f"Expected: "
            f"{EXPECTED_DOCUMENTS}\n"
            f"Actual  : "
            f"{len(corpus_records)}"
        )

    if (
        len(queries)
        != EXPECTED_QUERIES
    ):
        raise RuntimeError(
            "Jumlah query tidak sesuai.\n"
            f"Expected: "
            f"{EXPECTED_QUERIES}\n"
            f"Actual  : "
            f"{len(queries)}"
        )

    query_ids = [
        query["query_id"]
        for query in queries
    ]

    if (
        len(query_ids)
        != len(set(query_ids))
    ):
        raise RuntimeError(
            "Duplicate query_id ditemukan."
        )

    corpus_ids = {
        record["id"]
        for record in corpus_records
    }

    for query in queries:

        relevant_ids = query.get(
            "relevant_ids",
            [],
        )

        if len(relevant_ids) != 1:

            raise RuntimeError(
                f"{query['query_id']} "
                "tidak memiliki tepat "
                "1 relevant_id."
            )

        relevant_id = (
            relevant_ids[0]
        )

        if (
            relevant_id
            not in corpus_ids
        ):
            raise RuntimeError(
                f"{query['query_id']}: "
                f"{relevant_id} "
                "tidak ada di corpus."
            )

    print(
        "✅ Struktur benchmark valid."
    )

    print(
        f"Dokumen: "
        f"{len(corpus_records)}"
    )

    print(
        f"Query   : "
        f"{len(queries)}"
    )


# ============================================================
# 6. RRF FUSION
# ============================================================

def fuse_rrf(
    bm25_results,
    e5_results,
    record_map,
    rrf_k=60,
):
    """
    Reciprocal Rank Fusion.

    Menggunakan full ranking seluruh corpus.

    Formula:

    RRF(d) =
        1 / (K + rank_BM25)
        +
        1 / (K + rank_E5)

    Weight BM25 = 1
    Weight E5   = 1
    """

    bm25_rank = {
        result["id"]:
            result["rank"]

        for result
        in bm25_results
    }

    e5_rank = {
        result["id"]:
            result["rank"]

        for result
        in e5_results
    }

    fused = []

    for document_id in record_map:

        rank_bm25 = bm25_rank[
            document_id
        ]

        rank_e5 = e5_rank[
            document_id
        ]

        bm25_rrf = (
            1.0
            / (
                rrf_k
                + rank_bm25
            )
        )

        e5_rrf = (
            1.0
            / (
                rrf_k
                + rank_e5
            )
        )

        rrf_score = (
            bm25_rrf
            + e5_rrf
        )

        record = record_map[
            document_id
        ]

        fused.append({
            "id":
                document_id,

            "type":
                record["type"],

            "domain":
                record["domain"],

            "name":
                record["name"],

            "bm25_rank":
                rank_bm25,

            "e5_rank":
                rank_e5,

            "rrf_score":
                rrf_score,
        })

    # Deterministic ranking:
    #
    # score descending
    # lalu ID ascending.
    fused.sort(
        key=lambda result: (
            -result[
                "rrf_score"
            ],
            result[
                "id"
            ],
        )
    )

    for rank, result in enumerate(
        fused,
        start=1,
    ):

        result[
            "rank"
        ] = rank

    return fused


# ============================================================
# 7. FIND TARGET RANK
# ============================================================

def find_target_rank(
    results,
    relevant_id,
):

    for result in results:

        if (
            result["id"]
            == relevant_id
        ):

            return result[
                "rank"
            ]

    raise RuntimeError(
        f"Relevant ID tidak ditemukan: "
        f"{relevant_id}"
    )


# ============================================================
# 8. CALCULATE QUERY METRICS
# ============================================================

def calculate_metrics(
    target_rank,
):

    metrics = {}

    for k in TOP_K_VALUES:

        hit = int(
            target_rank <= k
        )

        # Karena setiap query hanya
        # punya satu relevant document:
        recall = hit

        metrics[
            f"hit@{k}"
        ] = hit

        metrics[
            f"recall@{k}"
        ] = recall

    if (
        target_rank
        <= 5
    ):

        mrr_5 = (
            1.0
            / target_rank
        )

    else:

        mrr_5 = 0.0

    metrics[
        "mrr@5"
    ] = mrr_5

    return metrics


# ============================================================
# 9. AGGREGATE METRICS
# ============================================================

def aggregate_metrics(
    rows,
    group_fields,
):

    grouped = defaultdict(
        list
    )

    for row in rows:

        key = tuple(
            row[field]
            for field
            in group_fields
        )

        grouped[key].append(
            row
        )

    output = []

    for key, group_rows in sorted(
        grouped.items()
    ):

        result = {}

        for (
            field,
            value,
        ) in zip(
            group_fields,
            key,
        ):

            result[field] = value

        result[
            "query_count"
        ] = len(group_rows)

        for metric in [
            "hit@1",
            "hit@3",
            "hit@5",
            "recall@1",
            "recall@3",
            "recall@5",
            "mrr@5",
        ]:

            result[metric] = mean(
                row[metric]
                for row
                in group_rows
            )

        result[
            "mean_target_rank"
        ] = mean(
            row["target_rank"]
            for row
            in group_rows
        )

        output.append(
            result
        )

    return output


# ============================================================
# 10. SAVE CSV
# ============================================================

def save_csv(
    path,
    rows,
    fieldnames,
):

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


# ============================================================
# 11. MAIN
# ============================================================

def main():

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "=" * 70
    )

    print(
        "AIRE OPTIMA "
        "RETRIEVAL BENCHMARK V1"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # VERIFY FROZEN INPUT
    # --------------------------------------------------------

    (
        corpus_hash,
        query_hash,
    ) = verify_frozen_artifacts()

    # --------------------------------------------------------
    # LOAD QUERY
    # --------------------------------------------------------

    queries = load_queries()

    # --------------------------------------------------------
    # BUILD BM25
    # --------------------------------------------------------

    print(
        "\nMembangun BM25..."
    )

    bm25 = BM25Retriever(
        corpus_path=CORPUS_FILE,
    )

    # --------------------------------------------------------
    # BUILD E5
    # --------------------------------------------------------

    print(
        "\nMembangun multilingual-E5..."
    )

    e5 = E5Retriever(
        corpus_path=CORPUS_FILE,
    )

    # --------------------------------------------------------
    # VALIDATE
    # --------------------------------------------------------

    validate_benchmark(
        queries=queries,
        corpus_records=bm25.records,
    )

    bm25_ids = [
        record["id"]
        for record
        in bm25.records
    ]

    e5_ids = [
        record["id"]
        for record
        in e5.records
    ]

    if bm25_ids != e5_ids:

        raise RuntimeError(
            "Urutan corpus BM25 dan E5 "
            "tidak identik."
        )

    document_count = len(
        bm25.records
    )

    record_map = {
        record["id"]: record
        for record
        in bm25.records
    }

    # --------------------------------------------------------
    # OUTPUT STORAGE
    # --------------------------------------------------------

    ranking_rows = []

    query_metric_rows = []

    # --------------------------------------------------------
    # RUN 40 QUERIES
    # --------------------------------------------------------

    print(
        "\nMemulai benchmark "
        f"{len(queries)} query..."
    )

    for query_number, query in enumerate(
        queries,
        start=1,
    ):

        query_id = query[
            "query_id"
        ]

        category = query[
            "category"
        ]

        domain = query[
            "domain"
        ]

        question = query[
            "question"
        ]

        relevant_id = query[
            "relevant_ids"
        ][0]

        hard_negative_ids = set(
            query.get(
                "hard_negative_ids",
                [],
            )
        )

        # ----------------------------------------------------
        # BM25 FULL RANKING
        # ----------------------------------------------------

        bm25_results = bm25.search(
            query=question,
            top_k=document_count,
        )

        # ----------------------------------------------------
        # E5 FULL RANKING
        # ----------------------------------------------------

        e5_results = e5.search(
            query=question,
            top_k=document_count,
        )

        # ----------------------------------------------------
        # HYBRID RRF
        # ----------------------------------------------------

        hybrid_results = fuse_rrf(
            bm25_results=
                bm25_results,

            e5_results=
                e5_results,

            record_map=
                record_map,

            rrf_k=
                RRF_K,
        )

        method_results = {
            "bm25":
                bm25_results,

            "e5":
                e5_results,

            "hybrid_rrf":
                hybrid_results,
        }

        # ----------------------------------------------------
        # EVALUATE EACH METHOD
        # ----------------------------------------------------

        rank_display = {}

        for (
            method,
            results,
        ) in method_results.items():

            target_rank = (
                find_target_rank(
                    results,
                    relevant_id,
                )
            )

            rank_display[
                method
            ] = target_rank

            metrics = (
                calculate_metrics(
                    target_rank
                )
            )

            # --------------------------------------------
            # QUERY-LEVEL METRICS
            # --------------------------------------------

            query_metric_rows.append({
                "query_id":
                    query_id,

                "category":
                    category,

                "domain":
                    domain,

                "method":
                    method,

                "relevant_id":
                    relevant_id,

                "target_rank":
                    target_rank,

                "hit@1":
                    metrics[
                        "hit@1"
                    ],

                "hit@3":
                    metrics[
                        "hit@3"
                    ],

                "hit@5":
                    metrics[
                        "hit@5"
                    ],

                "recall@1":
                    metrics[
                        "recall@1"
                    ],

                "recall@3":
                    metrics[
                        "recall@3"
                    ],

                "recall@5":
                    metrics[
                        "recall@5"
                    ],

                "mrr@5":
                    metrics[
                        "mrr@5"
                    ],
            })

            # --------------------------------------------
            # STORE TOP-5 RANKINGS
            # --------------------------------------------

            for result in results[
                :MAX_OUTPUT_K
            ]:

                result_id = result[
                    "id"
                ]

                if method == "bm25":

                    score = result[
                        "score"
                    ]

                    component_bm25_rank = (
                        result[
                            "rank"
                        ]
                    )

                    component_e5_rank = ""

                elif method == "e5":

                    score = result[
                        "score"
                    ]

                    component_bm25_rank = ""

                    component_e5_rank = (
                        result[
                            "rank"
                        ]
                    )

                else:

                    score = result[
                        "rrf_score"
                    ]

                    component_bm25_rank = (
                        result[
                            "bm25_rank"
                        ]
                    )

                    component_e5_rank = (
                        result[
                            "e5_rank"
                        ]
                    )

                ranking_rows.append({
                    "query_id":
                        query_id,

                    "category":
                        category,

                    "domain":
                        domain,

                    "method":
                        method,

                    "question":
                        question,

                    "relevant_id":
                        relevant_id,

                    "result_rank":
                        result[
                            "rank"
                        ],

                    "result_id":
                        result_id,

                    "result_name":
                        result[
                            "name"
                        ],

                    "score":
                        score,

                    "is_relevant":
                        (
                            result_id
                            == relevant_id
                        ),

                    "is_hard_negative":
                        (
                            result_id
                            in hard_negative_ids
                        ),

                    "bm25_rank":
                        component_bm25_rank,

                    "e5_rank":
                        component_e5_rank,
                })

        # ----------------------------------------------------
        # PROGRESS
        # ----------------------------------------------------

        print(
            f"[{query_number:02d}/"
            f"{len(queries)}] "
            f"{query_id} | "
            f"BM25={rank_display['bm25']} | "
            f"E5={rank_display['e5']} | "
            f"Hybrid="
            f"{rank_display['hybrid_rrf']}"
        )

    # ========================================================
    # 12. AGGREGATE
    # ========================================================

    overall_metrics = (
        aggregate_metrics(
            query_metric_rows,
            group_fields=[
                "method"
            ],
        )
    )

    category_metrics = (
        aggregate_metrics(
            query_metric_rows,
            group_fields=[
                "method",
                "category",
            ],
        )
    )

    domain_metrics = (
        aggregate_metrics(
            query_metric_rows,
            group_fields=[
                "method",
                "domain",
            ],
        )
    )

    # ========================================================
    # 13. SAVE RAW RANKINGS
    # ========================================================

    save_csv(
        RANKINGS_FILE,
        ranking_rows,
        [
            "query_id",
            "category",
            "domain",
            "method",
            "question",
            "relevant_id",
            "result_rank",
            "result_id",
            "result_name",
            "score",
            "is_relevant",
            "is_hard_negative",
            "bm25_rank",
            "e5_rank",
        ],
    )

    # ========================================================
    # 14. SAVE QUERY METRICS
    # ========================================================

    save_csv(
        QUERY_METRICS_FILE,
        query_metric_rows,
        [
            "query_id",
            "category",
            "domain",
            "method",
            "relevant_id",
            "target_rank",
            "hit@1",
            "hit@3",
            "hit@5",
            "recall@1",
            "recall@3",
            "recall@5",
            "mrr@5",
        ],
    )

    metric_fields = [
        "method",
        "query_count",
        "hit@1",
        "hit@3",
        "hit@5",
        "recall@1",
        "recall@3",
        "recall@5",
        "mrr@5",
        "mean_target_rank",
    ]

    # ========================================================
    # 15. SAVE OVERALL
    # ========================================================

    save_csv(
        OVERALL_METRICS_FILE,
        overall_metrics,
        metric_fields,
    )

    # ========================================================
    # 16. SAVE CATEGORY
    # ========================================================

    save_csv(
        CATEGORY_METRICS_FILE,
        category_metrics,
        [
            "method",
            "category",
            "query_count",
            "hit@1",
            "hit@3",
            "hit@5",
            "recall@1",
            "recall@3",
            "recall@5",
            "mrr@5",
            "mean_target_rank",
        ],
    )

    # ========================================================
    # 17. SAVE DOMAIN
    # ========================================================

    save_csv(
        DOMAIN_METRICS_FILE,
        domain_metrics,
        [
            "method",
            "domain",
            "query_count",
            "hit@1",
            "hit@3",
            "hit@5",
            "recall@1",
            "recall@3",
            "recall@5",
            "mrr@5",
            "mean_target_rank",
        ],
    )

    # ========================================================
    # 18. METADATA
    # ========================================================

    try:
        rank_bm25_version = (
            version(
                "rank-bm25"
            )
        )

    except Exception:
        rank_bm25_version = (
            "unknown"
        )

    metadata = {
        "benchmark":
            "Aire Optima Retrieval Benchmark v1",

        "corpus_file":
            str(CORPUS_FILE),

        "corpus_sha256":
            corpus_hash,

        "query_file":
            str(QUERY_FILE),

        "query_sha256":
            query_hash,

        "document_count":
            document_count,

        "query_count":
            len(queries),

        "methods": {
            "bm25": {
                "algorithm":
                    "BM25Okapi",

                "k1":
                    bm25.k1,

                "b":
                    bm25.b,

                "epsilon":
                    bm25.epsilon,
            },

            "e5": {
                "model":
                    e5.model_name,

                "max_tokens":
                    512,

                "embedding":
                    "mean pooling + "
                    "L2 normalization",

                "ranking":
                    "exact cosine similarity",

                "device":
                    str(
                        e5.device
                    ),
            },

            "hybrid_rrf": {
                "rrf_k":
                    RRF_K,

                "bm25_weight":
                    1.0,

                "e5_weight":
                    1.0,

                "candidate_pool":
                    document_count,

                "reranker":
                    None,
            },
        },

        "metrics": [
            "Hit@1",
            "Hit@3",
            "Hit@5",
            "Recall@1",
            "Recall@3",
            "Recall@5",
            "MRR@5",
        ],

        "relevance": {
            "type":
                "binary",

            "relevant_documents_per_query":
                1,
        },

        "environment": {
            "python":
                platform.python_version(),

            "torch":
                torch.__version__,

            "transformers":
                transformers.__version__,

            "rank-bm25":
                rank_bm25_version,

            "cuda_available":
                torch.cuda.is_available(),

            "cuda_device":
                (
                    torch.cuda.get_device_name(
                        0
                    )
                    if (
                        torch.cuda.is_available()
                    )
                    else None
                ),
        },
    }

    with METADATA_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            ensure_ascii=False,
            indent=2,
        )

    # ========================================================
    # 19. PRINT FINAL SUMMARY
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "OVERALL RETRIEVAL METRICS"
    )

    print(
        "=" * 70
    )

    print(
        f"{'METHOD':<14}"
        f"{'H@1':>8}"
        f"{'H@3':>8}"
        f"{'H@5':>8}"
        f"{'MRR@5':>10}"
        f"{'MEAN RANK':>12}"
    )

    for row in overall_metrics:

        print(
            f"{row['method']:<14}"
            f"{row['hit@1']:>8.4f}"
            f"{row['hit@3']:>8.4f}"
            f"{row['hit@5']:>8.4f}"
            f"{row['mrr@5']:>10.4f}"
            f"{row['mean_target_rank']:>12.2f}"
        )

    print(
        "\n✅ RETRIEVAL BENCHMARK SELESAI"
    )

    print(
        f"\nRanking:\n"
        f"{RANKINGS_FILE}"
    )

    print(
        f"\nQuery metrics:\n"
        f"{QUERY_METRICS_FILE}"
    )

    print(
        f"\nOverall:\n"
        f"{OVERALL_METRICS_FILE}"
    )

    print(
        f"\nBy category:\n"
        f"{CATEGORY_METRICS_FILE}"
    )

    print(
        f"\nBy domain:\n"
        f"{DOMAIN_METRICS_FILE}"
    )

    print(
        f"\nMetadata:\n"
        f"{METADATA_FILE}"
    )


if __name__ == "__main__":
    main()