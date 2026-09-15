import csv
import hashlib
import json
import math
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

MASTER_JSONL_FILE = Path(
    "benchmark/results/analysis/"
    "rag_master_analysis_v1.jsonl"
)

MASTER_CSV_FILE = Path(
    "benchmark/results/analysis/"
    "rag_master_analysis_v1.csv"
)

MASTER_MANIFEST_FILE = Path(
    "benchmark/results/analysis/"
    "rag_master_analysis_manifest_v1.json"
)

OUTPUT_DIR = Path(
    "benchmark/results/analysis/"
    "diagnostic_cases"
)

QUERY_CASES_CSV = (
    OUTPUT_DIR
    / "diagnostic_query_cases_v1.csv"
)

RETRIEVAL_FAILURES_CSV = (
    OUTPUT_DIR
    / "diagnostic_retrieval_failures_v1.csv"
)

SUMMARY_JSON = (
    OUTPUT_DIR
    / "diagnostic_case_analysis_summary_v1.json"
)

MANIFEST_FILE = (
    OUTPUT_DIR
    / "diagnostic_case_analysis_manifest_v1.json"
)


# ============================================================
# FROZEN HASHES
# ============================================================

EXPECTED_MASTER_JSONL_SHA256 = (
    "211936a8cbfe06e1f1b658a745371599"
    "f0362c8da5f80344c1ade4f695e95280"
)

EXPECTED_MASTER_CSV_SHA256 = (
    "cbfccaeadbb72ce549cedcac154e93e3"
    "cabda64a761275cf945259363901c4cb"
)

EXPECTED_MASTER_MANIFEST_SHA256 = (
    "e6eed1b98ac9b03baad57072bb184eed"
    "df0f0d8d4b63fe535b1af8b08bf4a52d"
)


# ============================================================
# CONSTANTS
# ============================================================

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

EXPECTED_ROWS = 120
EXPECTED_QUERIES = 40


# ============================================================
# HELPERS
# ============================================================

def sha256_file(path):

    hasher = hashlib.sha256()

    with path.open("rb") as file:

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


def load_json(path):

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(
            file
        )


def load_jsonl(path):

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
                    f"Invalid JSON line "
                    f"{line_number}"
                ) from exc

    return records


def write_json(
    path,
    data,
):

    with path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as file:

        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )

        file.write("\n")


def write_csv(
    path,
    records,
):

    if not records:

        raise RuntimeError(
            f"No records for {path}"
        )

    fields = list(
        records[0].keys()
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fields,
            lineterminator="\n",
        )

        writer.writeheader()

        writer.writerows(
            records
        )


def finite(
    value,
    label,
):

    value = float(value)

    if not math.isfinite(value):

        raise RuntimeError(
            f"Non-finite: {label}"
        )

    return value


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 78
    )

    print(
        "DIAGNOSTIC CASE ANALYSIS V1"
    )

    print(
        "=" * 78
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # SOURCE AUDIT
    # ========================================================

    for path in [
        MASTER_JSONL_FILE,
        MASTER_CSV_FILE,
        MASTER_MANIFEST_FILE,
    ]:

        if not path.exists():

            raise FileNotFoundError(
                f"Missing: {path}"
            )

    jsonl_hash = sha256_file(
        MASTER_JSONL_FILE
    )

    csv_hash = sha256_file(
        MASTER_CSV_FILE
    )

    manifest_hash = sha256_file(
        MASTER_MANIFEST_FILE
    )

    print("\nSOURCE HASH AUDIT")

    print(
        f"master_jsonl    : "
        f"{jsonl_hash}"
    )

    print(
        f"master_csv      : "
        f"{csv_hash}"
    )

    print(
        f"master_manifest : "
        f"{manifest_hash}"
    )

    if (
        jsonl_hash
        !=
        EXPECTED_MASTER_JSONL_SHA256
    ):

        raise RuntimeError(
            "Master JSONL changed."
        )

    if (
        csv_hash
        !=
        EXPECTED_MASTER_CSV_SHA256
    ):

        raise RuntimeError(
            "Master CSV changed."
        )

    if (
        manifest_hash
        !=
        EXPECTED_MASTER_MANIFEST_SHA256
    ):

        raise RuntimeError(
            "Master manifest changed."
        )

    master_manifest = load_json(
        MASTER_MANIFEST_FILE
    )

    if (
        master_manifest.get(
            "status"
        )
        !=
        "FROZEN"
    ):

        raise RuntimeError(
            "Master not frozen."
        )

    rows = load_jsonl(
        MASTER_JSONL_FILE
    )

    if len(rows) != EXPECTED_ROWS:

        raise RuntimeError(
            "Master must contain 120 rows."
        )

    print(
        "\n✅ Frozen master valid."
    )

    # ========================================================
    # QUERY INDEX
    # ========================================================

    query_index = {}

    for row in rows:

        query_id = row[
            "query_id"
        ]

        method = row[
            "method"
        ]

        query_index.setdefault(
            query_id,
            {}
        )

        if (
            method
            in
            query_index[
                query_id
            ]
        ):

            raise RuntimeError(
                f"Duplicate "
                f"{query_id}/{method}"
            )

        query_index[
            query_id
        ][
            method
        ] = row

    query_ids = sorted(
        query_index.keys()
    )

    if (
        len(query_ids)
        !=
        EXPECTED_QUERIES
    ):

        raise RuntimeError(
            "Expected 40 queries."
        )

    for query_id in query_ids:

        if (
            set(
                query_index[
                    query_id
                ].keys()
            )
            !=
            set(
                METHODS
            )
        ):

            raise RuntimeError(
                f"{query_id} incomplete."
            )

    # ========================================================
    # BUILD QUERY-LEVEL DIAGNOSTIC TABLE
    # ========================================================

    query_cases = []

    for query_id in query_ids:

        method_rows = (
            query_index[
                query_id
            ]
        )

        reference_row = (
            method_rows[
                "bm25"
            ]
        )

        hit1_values = [
            int(
                method_rows[
                    method
                ][
                    "hit_at1"
                ]
            )

            for method
            in METHODS
        ]

        hit5_values = [
            int(
                method_rows[
                    method
                ][
                    "hit_at5"
                ]
            )

            for method
            in METHODS
        ]

        # ====================================================
        # TOP-5 CASE TYPES
        # ====================================================

        if sum(hit5_values) == 0:

            top5_case = (
                "all_methods_miss"
            )

        elif sum(hit5_values) == 3:

            top5_case = (
                "all_methods_hit"
            )

        else:

            top5_case = (
                "mixed_hit_miss"
            )

        # ====================================================
        # HIT@1 DISAGREEMENT
        # ====================================================

        hit1_disagreement = (
            len(
                set(
                    hit1_values
                )
            )
            >
            1
        )

        # ====================================================
        # RANK SPREAD
        #
        # Only among methods whose target appears Top-5.
        # ====================================================

        available_ranks = [
            int(
                method_rows[
                    method
                ][
                    "target_rank_top5"
                ]
            )

            for method
            in METHODS

            if (
                method_rows[
                    method
                ][
                    "target_rank_top5"
                ]
                is not None
            )
        ]

        if len(available_ranks) >= 2:

            rank_spread = (
                max(
                    available_ranks
                )
                -
                min(
                    available_ranks
                )
            )

        else:

            rank_spread = None

        # ====================================================
        # METRIC SPREADS
        # ====================================================

        metric_values = {}

        metric_spreads = {}

        for metric in METRICS:

            values = {
                method:
                    finite(
                        method_rows[
                            method
                        ][
                            metric
                        ],
                        (
                            f"{query_id}/"
                            f"{method}/"
                            f"{metric}"
                        ),
                    )

                for method
                in METHODS
            }

            metric_values[
                metric
            ] = values

            metric_spreads[
                metric
            ] = (
                max(
                    values.values()
                )
                -
                min(
                    values.values()
                )
            )

        # ====================================================
        # ROW
        # ====================================================

        query_cases.append(
            {
                "query_id":
                    query_id,

                "category":
                    reference_row[
                        "category"
                    ],

                "domain":
                    reference_row[
                        "domain"
                    ],

                "question":
                    reference_row[
                        "question"
                    ],

                "relevant_id":
                    reference_row[
                        "relevant_id"
                    ],

                "top5_case":
                    top5_case,

                "hit1_disagreement":
                    hit1_disagreement,

                "rank_spread_top5":
                    rank_spread,

                "bm25_rank":
                    method_rows[
                        "bm25"
                    ][
                        "target_rank_top5"
                    ],

                "e5_rank":
                    method_rows[
                        "e5"
                    ][
                        "target_rank_top5"
                    ],

                "hybrid_rank":
                    method_rows[
                        "hybrid_rrf"
                    ][
                        "target_rank_top5"
                    ],

                "bm25_faithfulness":
                    metric_values[
                        "faithfulness"
                    ][
                        "bm25"
                    ],

                "e5_faithfulness":
                    metric_values[
                        "faithfulness"
                    ][
                        "e5"
                    ],

                "hybrid_faithfulness":
                    metric_values[
                        "faithfulness"
                    ][
                        "hybrid_rrf"
                    ],

                "faithfulness_spread":
                    metric_spreads[
                        "faithfulness"
                    ],

                "bm25_relevancy":
                    metric_values[
                        "answer_relevancy"
                    ][
                        "bm25"
                    ],

                "e5_relevancy":
                    metric_values[
                        "answer_relevancy"
                    ][
                        "e5"
                    ],

                "hybrid_relevancy":
                    metric_values[
                        "answer_relevancy"
                    ][
                        "hybrid_rrf"
                    ],

                "relevancy_spread":
                    metric_spreads[
                        "answer_relevancy"
                    ],

                "bm25_accuracy":
                    metric_values[
                        "answer_accuracy"
                    ][
                        "bm25"
                    ],

                "e5_accuracy":
                    metric_values[
                        "answer_accuracy"
                    ][
                        "e5"
                    ],

                "hybrid_accuracy":
                    metric_values[
                        "answer_accuracy"
                    ][
                        "hybrid_rrf"
                    ],

                "accuracy_spread":
                    metric_spreads[
                        "answer_accuracy"
                    ],
            }
        )

    # ========================================================
    # ALL RETRIEVAL FAILURE ROWS
    #
    # Expected total:
    #
    # BM25   3
    # E5     2
    # Hybrid 3
    #
    # total = 8
    # ========================================================

    retrieval_failures = []

    for row in rows:

        if (
            row[
                "target_in_top5"
            ]
        ):

            continue

        retrieval_failures.append(
            {
                "query_id":
                    row[
                        "query_id"
                    ],

                "method":
                    row[
                        "method"
                    ],

                "category":
                    row[
                        "category"
                    ],

                "domain":
                    row[
                        "domain"
                    ],

                "question":
                    row[
                        "question"
                    ],

                "relevant_id":
                    row[
                        "relevant_id"
                    ],

                "retrieved_ids":
                    json.dumps(
                        row[
                            "retrieved_ids"
                        ],
                        ensure_ascii=False,
                    ),

                "faithfulness":
                    finite(
                        row[
                            "faithfulness"
                        ],
                        "failure faithfulness",
                    ),

                "answer_relevancy":
                    finite(
                        row[
                            "answer_relevancy"
                        ],
                        "failure relevancy",
                    ),

                "answer_accuracy":
                    finite(
                        row[
                            "answer_accuracy"
                        ],
                        "failure accuracy",
                    ),
            }
        )

    if (
        len(
            retrieval_failures
        )
        !=
        8
    ):

        raise RuntimeError(
            "Expected exactly "
            "8 method-query "
            "Top-5 failures, "
            f"got {len(retrieval_failures)}."
        )

    # ========================================================
    # DIAGNOSTIC GROUPS
    # ========================================================

    all_methods_miss = [
        row
        for row
        in query_cases
        if (
            row[
                "top5_case"
            ]
            ==
            "all_methods_miss"
        )
    ]

    mixed_hit_miss = [
        row
        for row
        in query_cases
        if (
            row[
                "top5_case"
            ]
            ==
            "mixed_hit_miss"
        )
    ]

    hit1_disagreements = [
        row
        for row
        in query_cases
        if (
            row[
                "hit1_disagreement"
            ]
        )
    ]

    # ========================================================
    # SORTED SPREADS
    #
    # No arbitrary threshold:
    # store all 40 queries, sorted descending.
    # ========================================================

    top_accuracy_spread = sorted(
        query_cases,
        key=lambda row:
            (
                -row[
                    "accuracy_spread"
                ],
                row[
                    "query_id"
                ],
            ),
    )

    top_relevancy_spread = sorted(
        query_cases,
        key=lambda row:
            (
                -row[
                    "relevancy_spread"
                ],
                row[
                    "query_id"
                ],
            ),
    )

    top_faithfulness_spread = sorted(
        query_cases,
        key=lambda row:
            (
                -row[
                    "faithfulness_spread"
                ],
                row[
                    "query_id"
                ],
            ),
    )

    # ========================================================
    # WRITE CSVs
    # ========================================================

    write_csv(
        QUERY_CASES_CSV,
        query_cases,
    )

    write_csv(
        RETRIEVAL_FAILURES_CSV,
        retrieval_failures,
    )

    # ========================================================
    # SUMMARY JSON
    # ========================================================

    summary = {
        "artifact":
            (
                "Aire Optima "
                "Diagnostic Case Analysis v1"
            ),

        "status":
            "COMPLETE_NOT_YET_FROZEN",

        "source_master_sha256":
            jsonl_hash,

        "counts": {
            "queries":
                len(
                    query_cases
                ),

            "all_methods_miss_top5":
                len(
                    all_methods_miss
                ),

            "mixed_hit_miss_top5":
                len(
                    mixed_hit_miss
                ),

            "hit1_disagreement_queries":
                len(
                    hit1_disagreements
                ),

            "method_query_top5_failures":
                len(
                    retrieval_failures
                ),
        },

        "all_methods_miss_top5":
            all_methods_miss,

        "mixed_hit_miss_top5":
            mixed_hit_miss,

        "retrieval_failures":
            retrieval_failures,

        "top_10_accuracy_spread":
            top_accuracy_spread[
                :10
            ],

        "top_10_relevancy_spread":
            top_relevancy_spread[
                :10
            ],

        "top_10_faithfulness_spread":
            top_faithfulness_spread[
                :10
            ],

        "selection_policy": {
            "all_methods_miss":
                (
                    "All queries for which none "
                    "of the three retrieval methods "
                    "retrieved the canonical relevant "
                    "document in Top-5."
                ),

            "mixed_hit_miss":
                (
                    "All queries for which at least "
                    "one retrieval method retrieved "
                    "the target in Top-5 and at least "
                    "one did not."
                ),

            "retrieval_failures":
                (
                    "All method-query observations "
                    "where target_in_top5=False. "
                    "No cases are omitted."
                ),

            "metric_spread":
                (
                    "All queries are deterministically "
                    "ranked by max(method score) minus "
                    "min(method score). Top-10 is "
                    "reported only for compact review."
                ),
        },
    }

    write_json(
        SUMMARY_JSON,
        summary,
    )

    # ========================================================
    # OUTPUT HASHES
    # ========================================================

    output_hashes = {
        "query_cases_csv":
            sha256_file(
                QUERY_CASES_CSV
            ),

        "retrieval_failures_csv":
            sha256_file(
                RETRIEVAL_FAILURES_CSV
            ),

        "summary_json":
            sha256_file(
                SUMMARY_JSON
            ),
    }

    # ========================================================
    # FREEZE MANIFEST
    # ========================================================

    freeze_manifest = {
        "artifact":
            (
                "Aire Optima "
                "Diagnostic Case Analysis v1"
            ),

        "status":
            "FROZEN",

        "source": {
            "master_jsonl":
                jsonl_hash,

            "master_csv":
                csv_hash,

            "master_manifest":
                manifest_hash,
        },

        "case_selection":
            (
                "Deterministic and exhaustive "
                "for retrieval failures and "
                "retrieval disagreement cases; "
                "metric-spread cases are sorted "
                "without manual selection."
            ),

        "outputs":
            output_hashes,

        "freeze_rule":
            (
                "Do not modify diagnostic case "
                "selection after inspection. "
                "Paper examples must be drawn "
                "from this frozen artifact."
            ),
    }

    write_json(
        MANIFEST_FILE,
        freeze_manifest,
    )

    freeze_manifest_hash = (
        sha256_file(
            MANIFEST_FILE
        )
    )

    # ========================================================
    # PRINT
    # ========================================================

    print(
        "\n"
        + "=" * 78
    )

    print(
        "DIAGNOSTIC COUNTS"
    )

    print(
        "=" * 78
    )

    print(
        f"Queries                  : "
        f"{len(query_cases)}"
    )

    print(
        f"All methods miss Top-5   : "
        f"{len(all_methods_miss)}"
    )

    print(
        f"Mixed Top-5 hit/miss     : "
        f"{len(mixed_hit_miss)}"
    )

    print(
        f"Hit@1 disagreement       : "
        f"{len(hit1_disagreements)}"
    )

    print(
        f"Method-query Top5 failure: "
        f"{len(retrieval_failures)}"
    )

    # ========================================================
    # ALL-METHOD MISS
    # ========================================================

    print(
        "\n"
        + "=" * 78
    )

    print(
        "ALL METHODS MISS TOP-5"
    )

    print(
        "=" * 78
    )

    if not all_methods_miss:

        print(
            "None"
        )

    else:

        for row in all_methods_miss:

            print(
                f"\n{row['query_id']} "
                f"| {row['category']} "
                f"| {row['domain']}"
            )

            print(
                f"Question: "
                f"{row['question']}"
            )

            print(
                "Ranks: "
                f"BM25={row['bm25_rank']} "
                f"E5={row['e5_rank']} "
                f"Hybrid={row['hybrid_rank']}"
            )

            print(
                "Accuracy: "
                f"BM25={row['bm25_accuracy']:.3f} "
                f"E5={row['e5_accuracy']:.3f} "
                f"Hybrid={row['hybrid_accuracy']:.3f}"
            )

    # ========================================================
    # MIXED TOP-5 CASES
    # ========================================================

    print(
        "\n"
        + "=" * 78
    )

    print(
        "MIXED TOP-5 HIT / MISS"
    )

    print(
        "=" * 78
    )

    if not mixed_hit_miss:

        print(
            "None"
        )

    else:

        for row in mixed_hit_miss:

            print(
                f"\n{row['query_id']} "
                f"| {row['category']} "
                f"| {row['domain']}"
            )

            print(
                "Ranks: "
                f"BM25={row['bm25_rank']} "
                f"E5={row['e5_rank']} "
                f"Hybrid={row['hybrid_rank']}"
            )

            print(
                "Accuracy: "
                f"BM25={row['bm25_accuracy']:.3f} "
                f"E5={row['e5_accuracy']:.3f} "
                f"Hybrid={row['hybrid_accuracy']:.3f}"
            )

            print(
                "Relevancy: "
                f"BM25={row['bm25_relevancy']:.3f} "
                f"E5={row['e5_relevancy']:.3f} "
                f"Hybrid={row['hybrid_relevancy']:.3f}"
            )

    # ========================================================
    # ALL 8 RETRIEVAL FAILURES
    # ========================================================

    print(
        "\n"
        + "=" * 78
    )

    print(
        "ALL METHOD-QUERY TOP-5 FAILURES"
    )

    print(
        "=" * 78
    )

    for row in retrieval_failures:

        print(
            f"{row['query_id']} "
            f"{row['method']:<11} "
            f"| F={row['faithfulness']:.3f} "
            f"R={row['answer_relevancy']:.3f} "
            f"A={row['answer_accuracy']:.3f}"
        )

    # ========================================================
    # TOP SPREADS
    # ========================================================

    print(
        "\n"
        + "=" * 78
    )

    print(
        "TOP 5 ANSWER-ACCURACY SPREAD"
    )

    print(
        "=" * 78
    )

    for row in (
        top_accuracy_spread[
            :5
        ]
    ):

        print(
            f"{row['query_id']} "
            f"spread="
            f"{row['accuracy_spread']:.3f} "
            f"| BM25="
            f"{row['bm25_accuracy']:.3f} "
            f"E5="
            f"{row['e5_accuracy']:.3f} "
            f"Hybrid="
            f"{row['hybrid_accuracy']:.3f}"
        )

    print(
        "\n"
        + "=" * 78
    )

    print(
        "TOP 5 ANSWER-RELEVANCY SPREAD"
    )

    print(
        "=" * 78
    )

    for row in (
        top_relevancy_spread[
            :5
        ]
    ):

        print(
            f"{row['query_id']} "
            f"spread="
            f"{row['relevancy_spread']:.3f} "
            f"| BM25="
            f"{row['bm25_relevancy']:.3f} "
            f"E5="
            f"{row['e5_relevancy']:.3f} "
            f"Hybrid="
            f"{row['hybrid_relevancy']:.3f}"
        )

    print(
        "\n"
        + "=" * 78
    )

    print(
        "TOP 5 FAITHFULNESS SPREAD"
    )

    print(
        "=" * 78
    )

    for row in (
        top_faithfulness_spread[
            :5
        ]
    ):

        print(
            f"{row['query_id']} "
            f"spread="
            f"{row['faithfulness_spread']:.3f} "
            f"| BM25="
            f"{row['bm25_faithfulness']:.3f} "
            f"E5="
            f"{row['e5_faithfulness']:.3f} "
            f"Hybrid="
            f"{row['hybrid_faithfulness']:.3f}"
        )

    # ========================================================
    # FINAL
    # ========================================================

    print(
        "\n"
        + "=" * 78
    )

    print(
        "DIAGNOSTIC CASE ANALYSIS FROZEN"
    )

    print(
        "=" * 78
    )

    print(
        "\nOUTPUT HASHES"
    )

    for (
        name,
        value,
    ) in output_hashes.items():

        print(
            f"{name:<25}: "
            f"{value}"
        )

    print(
        "\nMANIFEST SHA256:"
    )

    print(
        freeze_manifest_hash
    )

    print(
        "\nSTATUS: FROZEN"
    )


if __name__ == "__main__":
    main()