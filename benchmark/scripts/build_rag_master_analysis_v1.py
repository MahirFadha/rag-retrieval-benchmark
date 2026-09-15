import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


# ============================================================
# 1. PATHS
# ============================================================

EVALUATION_DATASET_FILE = Path(
    "benchmark/data/evaluation/"
    "rag_evaluation_dataset_v1.jsonl"
)

RAG_SCORES_FILE = Path(
    "benchmark/results/evaluation/"
    "rag_evaluation_scores_v1.jsonl"
)

RAG_SCORES_MANIFEST_FILE = Path(
    "benchmark/results/evaluation/"
    "rag_evaluation_scores_manifest_v1.json"
)

OUTPUT_JSONL_FILE = Path(
    "benchmark/results/analysis/"
    "rag_master_analysis_v1.jsonl"
)

OUTPUT_CSV_FILE = Path(
    "benchmark/results/analysis/"
    "rag_master_analysis_v1.csv"
)

OUTPUT_MANIFEST_FILE = Path(
    "benchmark/results/analysis/"
    "rag_master_analysis_manifest_v1.json"
)


# ============================================================
# 2. FROZEN SOURCE HASHES
# ============================================================

EXPECTED_DATASET_SHA256 = (
    "fe1582a06378e9aa087551a2ee01b4c5"
    "76fb14a38cdb51cc7057439a4e1b27fd"
)

EXPECTED_SCORES_SHA256 = (
    "0d66e699ec1f419c132850b13d935b9e"
    "1498a4626235671a6c11667e37ea400b"
)

EXPECTED_SCORES_MANIFEST_SHA256 = (
    "c8d5827e72c898c54df97d15baaec710"
    "310dd60433857308179908580595de91"
)


# ============================================================
# 3. EXPECTED COUNTS
# ============================================================

EXPECTED_DATASET_ROWS = 120
EXPECTED_SCORE_ROWS = 360
EXPECTED_MASTER_ROWS = 120
EXPECTED_QUERY_COUNT = 40


METHODS = [
    "bm25",
    "e5",
    "hybrid_rrf",
]

METRICS = [
    "faithfulness",
    "answer_relevancy",
    "answer_accuracy",
]

CATEGORIES = [
    "exact",
    "semantic",
    "multi_constraint",
    "fine_grained",
]

DOMAINS = [
    "product_ac",
    "service_ac",
    "cctv",
]


# ============================================================
# 4. HELPERS
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


def load_json(
    path: Path,
):

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(
            file
        )


def load_jsonl(
    path: Path,
):

    records = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        for (
            line_number,
            line,
        ) in enumerate(
            file,
            start=1,
        ):

            line = line.strip()

            if not line:
                continue

            try:

                records.append(
                    json.loads(
                        line
                    )
                )

            except json.JSONDecodeError as exc:

                raise RuntimeError(
                    f"Invalid JSON pada "
                    f"{path}, "
                    f"line {line_number}."
                ) from exc

    return records


def atomic_write_json(
    path: Path,
    data,
):

    temporary_path = (
        path.with_suffix(
            path.suffix + ".tmp"
        )
    )

    with temporary_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
            default=str,
        )

        file.flush()

    temporary_path.replace(
        path
    )


def normalize_category(
    value,
):

    value = str(
        value
    ).strip().lower()

    replacements = {
        "multi-constraint":
            "multi_constraint",

        "multi constraint":
            "multi_constraint",

        "fine-grained":
            "fine_grained",

        "fine grained":
            "fine_grained",
    }

    return replacements.get(
        value,
        value,
    )


def normalize_domain(
    value,
):

    value = str(
        value
    ).strip().lower()

    replacements = {
        "product ac":
            "product_ac",

        "service ac":
            "service_ac",

        "product-ac":
            "product_ac",

        "service-ac":
            "service_ac",
    }

    return replacements.get(
        value,
        value,
    )


# ============================================================
# 5. MAIN
# ============================================================

def main():

    print(
        "=" * 72
    )

    print(
        "BUILD RAG MASTER ANALYSIS V1"
    )

    print(
        "=" * 72
    )

    # ========================================================
    # REQUIRED FILES
    # ========================================================

    required_files = [
        EVALUATION_DATASET_FILE,
        RAG_SCORES_FILE,
        RAG_SCORES_MANIFEST_FILE,
    ]

    for path in required_files:

        if not path.exists():

            raise FileNotFoundError(
                f"File tidak ditemukan: "
                f"{path}"
            )

    OUTPUT_JSONL_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # SOURCE HASH AUDIT
    # ========================================================

    dataset_hash = sha256_file(
        EVALUATION_DATASET_FILE
    )

    scores_hash = sha256_file(
        RAG_SCORES_FILE
    )

    scores_manifest_hash = sha256_file(
        RAG_SCORES_MANIFEST_FILE
    )

    print(
        "\nSOURCE HASH AUDIT"
    )

    print(
        f"evaluation_dataset : "
        f"{dataset_hash}"
    )

    print(
        f"rag_scores         : "
        f"{scores_hash}"
    )

    print(
        f"scores_manifest    : "
        f"{scores_manifest_hash}"
    )

    if (
        dataset_hash
        !=
        EXPECTED_DATASET_SHA256
    ):

        raise RuntimeError(
            "Evaluation dataset hash berubah."
        )

    if (
        scores_hash
        !=
        EXPECTED_SCORES_SHA256
    ):

        raise RuntimeError(
            "Frozen RAG scores hash berubah."
        )

    if (
        scores_manifest_hash
        !=
        EXPECTED_SCORES_MANIFEST_SHA256
    ):

        raise RuntimeError(
            "Frozen scores manifest hash berubah."
        )

    print(
        "\n✅ Semua source hash valid."
    )

    # ========================================================
    # LOAD SOURCES
    # ========================================================

    dataset = load_jsonl(
        EVALUATION_DATASET_FILE
    )

    scores = load_jsonl(
        RAG_SCORES_FILE
    )

    scores_manifest = load_json(
        RAG_SCORES_MANIFEST_FILE
    )

    if (
        scores_manifest.get(
            "status"
        )
        !=
        "FROZEN"
    ):

        raise RuntimeError(
            "RAG scores belum berstatus FROZEN."
        )

    if (
        len(dataset)
        !=
        EXPECTED_DATASET_ROWS
    ):

        raise RuntimeError(
            f"Evaluation dataset bukan "
            f"{EXPECTED_DATASET_ROWS} rows."
        )

    if (
        len(scores)
        !=
        EXPECTED_SCORE_ROWS
    ):

        raise RuntimeError(
            f"RAG scores bukan "
            f"{EXPECTED_SCORE_ROWS} rows."
        )

    # ========================================================
    # DATASET INDEX
    # ========================================================

    dataset_by_pair = {}

    for row in dataset:

        query_id = (
            row[
                "query_id"
            ]
        )

        method = (
            row[
                "method"
            ]
        )

        pair = (
            query_id,
            method,
        )

        if (
            pair
            in dataset_by_pair
        ):

            raise RuntimeError(
                f"Duplicate dataset pair: "
                f"{pair}"
            )

        dataset_by_pair[
            pair
        ] = row

    if (
        len(dataset_by_pair)
        !=
        EXPECTED_MASTER_ROWS
    ):

        raise RuntimeError(
            "Unique query-method pair "
            "bukan 120."
        )

    # ========================================================
    # SCORE INDEX
    #
    # (query_id, method)
    #     → metric → value
    # ========================================================

    scores_by_pair = defaultdict(
        dict
    )

    seen_score_triples = set()

    for score in scores:

        query_id = (
            score[
                "query_id"
            ]
        )

        method = (
            score[
                "method"
            ]
        )

        metric = (
            score[
                "metric"
            ]
        )

        triple = (
            query_id,
            method,
            metric,
        )

        if (
            triple
            in seen_score_triples
        ):

            raise RuntimeError(
                f"Duplicate score triple: "
                f"{triple}"
            )

        seen_score_triples.add(
            triple
        )

        if (
            method
            not in METHODS
        ):

            raise RuntimeError(
                f"Unknown method: "
                f"{method}"
            )

        if (
            metric
            not in METRICS
        ):

            raise RuntimeError(
                f"Unknown metric: "
                f"{metric}"
            )

        try:

            value = float(
                score[
                    "value"
                ]
            )

        except (
            TypeError,
            ValueError,
        ) as exc:

            raise RuntimeError(
                f"Non-numeric score: "
                f"{triple}"
            ) from exc

        if not math.isfinite(
            value
        ):

            raise RuntimeError(
                f"Non-finite score: "
                f"{triple}"
            )

        scores_by_pair[
            (
                query_id,
                method,
            )
        ][
            metric
        ] = value

    if (
        len(
            seen_score_triples
        )
        !=
        EXPECTED_SCORE_ROWS
    ):

        raise RuntimeError(
            "Unique score triples "
            "bukan 360."
        )

    # ========================================================
    # BUILD MASTER TABLE
    # ========================================================

    master_rows = []

    category_counts = Counter()
    domain_counts = Counter()
    method_counts = Counter()

    top1_counts = Counter()
    top3_counts = Counter()
    top5_counts = Counter()

    # deterministic order:
    #
    # Q001:
    # bm25
    # e5
    # hybrid_rrf
    #
    # ...
    # Q040
    # ========================================================

    for query_number in range(
        1,
        EXPECTED_QUERY_COUNT + 1,
    ):

        query_id = (
            f"Q{query_number:03d}"
        )

        for method in METHODS:

            pair = (
                query_id,
                method,
            )

            if (
                pair
                not in dataset_by_pair
            ):

                raise RuntimeError(
                    f"Missing dataset pair: "
                    f"{pair}"
                )

            if (
                pair
                not in scores_by_pair
            ):

                raise RuntimeError(
                    f"Missing score pair: "
                    f"{pair}"
                )

            row = (
                dataset_by_pair[
                    pair
                ]
            )

            metric_scores = (
                scores_by_pair[
                    pair
                ]
            )

            if (
                set(
                    metric_scores.keys()
                )
                !=
                set(
                    METRICS
                )
            ):

                raise RuntimeError(
                    f"{pair} tidak memiliki "
                    "3 metric score."
                )

            # =================================================
            # CATEGORY / DOMAIN
            # =================================================

            category = (
                normalize_category(
                    row[
                        "category"
                    ]
                )
            )

            domain = (
                normalize_domain(
                    row[
                        "domain"
                    ]
                )
            )

            if (
                category
                not in CATEGORIES
            ):

                raise RuntimeError(
                    f"Unknown category: "
                    f"{category}"
                )

            if (
                domain
                not in DOMAINS
            ):

                raise RuntimeError(
                    f"Unknown domain: "
                    f"{domain}"
                )

            # =================================================
            # RETRIEVAL INFORMATION
            # =================================================

            retrieved_ids = (
                row[
                    "retrieved_ids"
                ]
            )

            if not isinstance(
                retrieved_ids,
                list,
            ):

                raise RuntimeError(
                    f"{pair}: retrieved_ids "
                    "bukan list."
                )

            if (
                len(
                    retrieved_ids
                )
                != 5
            ):

                raise RuntimeError(
                    f"{pair}: retrieved_ids "
                    "bukan Top-5."
                )

            relevant_id = (
                row[
                    "relevant_id"
                ]
            )

            target_in_top5_source = bool(
                row[
                    "target_in_top5"
                ]
            )

            target_rank_source = (
                row.get(
                    "target_rank_top5"
                )
            )

            # =================================================
            # RECOMPUTE TARGET POSITION FROM RETRIEVED IDS
            #
            # Jangan hanya percaya flag source.
            # =================================================

            if (
                relevant_id
                in retrieved_ids
            ):

                computed_rank = (
                    retrieved_ids.index(
                        relevant_id
                    )
                    + 1
                )

            else:

                computed_rank = None

            computed_in_top5 = (
                computed_rank
                is not None
                and
                computed_rank <= 5
            )

            if (
                computed_in_top5
                !=
                target_in_top5_source
            ):

                raise RuntimeError(
                    f"{pair}: target_in_top5 "
                    "mismatch."
                )

            if (
                computed_rank
                is None
            ):

                if (
                    target_rank_source
                    is not None
                ):

                    raise RuntimeError(
                        f"{pair}: target rank "
                        "seharusnya null."
                    )

            else:

                if (
                    int(
                        target_rank_source
                    )
                    !=
                    computed_rank
                ):

                    raise RuntimeError(
                        f"{pair}: target rank "
                        "mismatch."
                    )

            # =================================================
            # DERIVED RETRIEVAL FEATURES
            # =================================================

            target_in_top1 = (
                computed_rank == 1
            )

            target_in_top3 = (
                computed_rank is not None
                and
                computed_rank <= 3
            )

            target_in_top5 = (
                computed_in_top5
            )

            reciprocal_rank_at5 = (
                (
                    1.0
                    /
                    computed_rank
                )
                if (
                    computed_rank
                    is not None
                )
                else
                0.0
            )

            # With exactly one relevant document:
            #
            # Hit@K == Recall@K
            # =================================================

            hit_at1 = int(
                target_in_top1
            )

            hit_at3 = int(
                target_in_top3
            )

            hit_at5 = int(
                target_in_top5
            )

            recall_at1 = (
                hit_at1
            )

            recall_at3 = (
                hit_at3
            )

            recall_at5 = (
                hit_at5
            )

            # =================================================
            # MASTER ROW
            # =================================================

            master_row = {
                "query_id":
                    query_id,

                "method":
                    method,

                "category":
                    category,

                "domain":
                    domain,

                "question":
                    row[
                        "question"
                    ],

                "relevant_id":
                    relevant_id,

                "retrieved_ids":
                    retrieved_ids,

                "target_rank_top5":
                    computed_rank,

                "target_in_top1":
                    target_in_top1,

                "target_in_top3":
                    target_in_top3,

                "target_in_top5":
                    target_in_top5,

                "hit_at1":
                    hit_at1,

                "hit_at3":
                    hit_at3,

                "hit_at5":
                    hit_at5,

                "recall_at1":
                    recall_at1,

                "recall_at3":
                    recall_at3,

                "recall_at5":
                    recall_at5,

                "reciprocal_rank_at5":
                    reciprocal_rank_at5,

                "faithfulness":
                    metric_scores[
                        "faithfulness"
                    ],

                "answer_relevancy":
                    metric_scores[
                        "answer_relevancy"
                    ],

                "answer_accuracy":
                    metric_scores[
                        "answer_accuracy"
                    ],
            }

            master_rows.append(
                master_row
            )

            # =================================================
            # COUNTERS
            # =================================================

            method_counts[
                method
            ] += 1

            category_counts[
                category
            ] += 1

            domain_counts[
                domain
            ] += 1

            top1_counts[
                method
            ] += hit_at1

            top3_counts[
                method
            ] += hit_at3

            top5_counts[
                method
            ] += hit_at5

    # ========================================================
    # MASTER COUNT AUDIT
    # ========================================================

    if (
        len(master_rows)
        !=
        EXPECTED_MASTER_ROWS
    ):

        raise RuntimeError(
            "Master rows bukan 120."
        )

    # ========================================================
    # EXPECTED DISTRIBUTION
    # ========================================================

    for method in METHODS:

        if (
            method_counts[
                method
            ]
            != 40
        ):

            raise RuntimeError(
                f"{method} bukan "
                "40 rows."
            )

    expected_category_counts = {
        "exact":
            30,

        "semantic":
            30,

        "multi_constraint":
            30,

        "fine_grained":
            30,
    }

    if (
        dict(
            category_counts
        )
        !=
        expected_category_counts
    ):

        raise RuntimeError(
            "Category distribution "
            "tidak sesuai.\n"
            f"Actual: "
            f"{dict(category_counts)}"
        )

    expected_domain_counts = {
        "product_ac":
            36,

        "service_ac":
            48,

        "cctv":
            36,
    }

    if (
        dict(
            domain_counts
        )
        !=
        expected_domain_counts
    ):

        raise RuntimeError(
            "Domain distribution "
            "tidak sesuai.\n"
            f"Actual: "
            f"{dict(domain_counts)}"
        )

    # ========================================================
    # EXPECTED RETRIEVAL TOP-K AUDIT
    #
    # Dari frozen retrieval result sebelumnya.
    # ========================================================

    expected_top1 = {
        "bm25":
            31,

        "e5":
            26,

        "hybrid_rrf":
            29,
    }

    expected_top3 = {
        "bm25":
            35,

        "e5":
            36,

        "hybrid_rrf":
            36,
    }

    expected_top5 = {
        "bm25":
            37,

        "e5":
            38,

        "hybrid_rrf":
            37,
    }

    if (
        dict(
            top1_counts
        )
        !=
        expected_top1
    ):

        raise RuntimeError(
            "Hit@1 audit mismatch.\n"
            f"Actual: "
            f"{dict(top1_counts)}"
        )

    if (
        dict(
            top3_counts
        )
        !=
        expected_top3
    ):

        raise RuntimeError(
            "Hit@3 audit mismatch.\n"
            f"Actual: "
            f"{dict(top3_counts)}"
        )

    if (
        dict(
            top5_counts
        )
        !=
        expected_top5
    ):

        raise RuntimeError(
            "Hit@5 audit mismatch.\n"
            f"Actual: "
            f"{dict(top5_counts)}"
        )

    # ========================================================
    # WRITE JSONL
    # ========================================================

    with OUTPUT_JSONL_FILE.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as file:

        for row in master_rows:

            file.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                +
                "\n"
            )

    # ========================================================
    # WRITE CSV
    # ========================================================

    csv_fields = [
        "query_id",
        "method",
        "category",
        "domain",
        "question",
        "relevant_id",
        "retrieved_ids",
        "target_rank_top5",
        "target_in_top1",
        "target_in_top3",
        "target_in_top5",
        "hit_at1",
        "hit_at3",
        "hit_at5",
        "recall_at1",
        "recall_at3",
        "recall_at5",
        "reciprocal_rank_at5",
        "faithfulness",
        "answer_relevancy",
        "answer_accuracy",
    ]

    with OUTPUT_CSV_FILE.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=csv_fields,
        )

        writer.writeheader()

        for row in master_rows:

            csv_row = dict(
                row
            )

            csv_row[
                "retrieved_ids"
            ] = json.dumps(
                row[
                    "retrieved_ids"
                ],
                ensure_ascii=False,
            )

            writer.writerow(
                csv_row
            )

    # ========================================================
    # OUTPUT HASHES
    # ========================================================

    master_jsonl_hash = (
        sha256_file(
            OUTPUT_JSONL_FILE
        )
    )

    master_csv_hash = (
        sha256_file(
            OUTPUT_CSV_FILE
        )
    )

    # ========================================================
    # DESCRIPTIVE CHECK
    # ========================================================

    metric_means = {}

    for method in METHODS:

        method_rows = [
            row
            for row
            in master_rows
            if (
                row[
                    "method"
                ]
                ==
                method
            )
        ]

        metric_means[
            method
        ] = {}

        for metric in METRICS:

            values = [
                row[
                    metric
                ]
                for row
                in method_rows
            ]

            metric_means[
                method
            ][
                metric
            ] = (
                math.fsum(
                    values
                )
                /
                len(
                    values
                )
            )

    # ========================================================
    # MANIFEST
    # ========================================================

    manifest = {
        "artifact":
            (
                "Aire Optima "
                "RAG Master Analysis v1"
            ),

        "status":
            "FROZEN",

        "description":
            (
                "Deterministic 120-row analytical table "
                "joining the frozen 120 query-method RAG "
                "evaluation dataset with the frozen 360 "
                "RAG metric scores. One row represents one "
                "query-method pair."
            ),

        "row_count":
            len(
                master_rows
            ),

        "query_count":
            EXPECTED_QUERY_COUNT,

        "methods":
            METHODS,

        "metrics":
            METRICS,

        "sources": {
            "evaluation_dataset": {
                "file":
                    str(
                        EVALUATION_DATASET_FILE
                    ),

                "sha256":
                    dataset_hash,
            },

            "rag_scores": {
                "file":
                    str(
                        RAG_SCORES_FILE
                    ),

                "sha256":
                    scores_hash,
            },

            "rag_scores_manifest": {
                "file":
                    str(
                        RAG_SCORES_MANIFEST_FILE
                    ),

                "sha256":
                    scores_manifest_hash,
            },
        },

        "outputs": {
            "jsonl": {
                "file":
                    str(
                        OUTPUT_JSONL_FILE
                    ),

                "sha256":
                    master_jsonl_hash,
            },

            "csv": {
                "file":
                    str(
                        OUTPUT_CSV_FILE
                    ),

                "sha256":
                    master_csv_hash,
            },
        },

        "distribution": {
            "method_counts":
                dict(
                    method_counts
                ),

            "category_counts":
                dict(
                    category_counts
                ),

            "domain_counts":
                dict(
                    domain_counts
                ),
        },

        "retrieval_audit": {
            "hit_at1_count":
                dict(
                    top1_counts
                ),

            "hit_at3_count":
                dict(
                    top3_counts
                ),

            "hit_at5_count":
                dict(
                    top5_counts
                ),
        },

        "rag_metric_means":
            metric_means,

        "derivation_rules": {
            "target_rank_top5":
                (
                    "1-indexed position of relevant_id "
                    "inside retrieved_ids; null when absent "
                    "from Top-5."
                ),

            "target_in_top1":
                (
                    "True only when target_rank_top5 == 1."
                ),

            "target_in_top3":
                (
                    "True when target_rank_top5 is "
                    "between 1 and 3."
                ),

            "target_in_top5":
                (
                    "True when relevant_id occurs "
                    "in retrieved_ids."
                ),

            "reciprocal_rank_at5":
                (
                    "1 / target_rank_top5 when relevant "
                    "document occurs in Top-5; otherwise 0."
                ),

            "hit_recall_equivalence":
                (
                    "Because each query has exactly one "
                    "binary relevant document, Hit@K and "
                    "Recall@K are numerically identical."
                ),
        },

        "freeze_rule":
            (
                "Do not modify this master artifact after "
                "creation. All downstream statistical "
                "analyses must read this frozen JSONL."
            ),
    }

    atomic_write_json(
        OUTPUT_MANIFEST_FILE,
        manifest,
    )

    manifest_hash = (
        sha256_file(
            OUTPUT_MANIFEST_FILE
        )
    )

    # ========================================================
    # PRINT
    # ========================================================

    print(
        "\n"
        + "=" * 72
    )

    print(
        "MASTER TABLE AUDIT"
    )

    print(
        "=" * 72
    )

    print(
        f"Rows             : "
        f"{len(master_rows)}"
    )

    print(
        f"Queries          : "
        f"{EXPECTED_QUERY_COUNT}"
    )

    print(
        "\nMETHOD COUNTS"
    )

    for method in METHODS:

        print(
            f"{method:<15}: "
            f"{method_counts[method]}"
        )

    print(
        "\nCATEGORY COUNTS"
    )

    for category in CATEGORIES:

        print(
            f"{category:<20}: "
            f"{category_counts[category]}"
        )

    print(
        "\nDOMAIN COUNTS"
    )

    for domain in DOMAINS:

        print(
            f"{domain:<15}: "
            f"{domain_counts[domain]}"
        )

    print(
        "\nRETRIEVAL CHECK"
    )

    for method in METHODS:

        print(
            f"{method:<12} "
            f"Hit@1 "
            f"{top1_counts[method]:>2}/40 "
            f"| Hit@3 "
            f"{top3_counts[method]:>2}/40 "
            f"| Hit@5 "
            f"{top5_counts[method]:>2}/40"
        )

    print(
        "\nRAG MEAN CHECK"
    )

    for method in METHODS:

        print(
            f"\n{method}"
        )

        for metric in METRICS:

            print(
                f"  "
                f"{metric:<20}: "
                f"{metric_means[method][metric]:.6f}"
            )

    print(
        "\n"
        + "=" * 72
    )

    print(
        "MASTER ANALYSIS FROZEN"
    )

    print(
        "=" * 72
    )

    print(
        "\nJSONL:"
    )

    print(
        OUTPUT_JSONL_FILE
    )

    print(
        master_jsonl_hash
    )

    print(
        "\nCSV:"
    )

    print(
        OUTPUT_CSV_FILE
    )

    print(
        master_csv_hash
    )

    print(
        "\nMANIFEST:"
    )

    print(
        OUTPUT_MANIFEST_FILE
    )

    print(
        manifest_hash
    )

    print(
        "\nSTATUS: FROZEN"
    )


if __name__ == "__main__":
    main()