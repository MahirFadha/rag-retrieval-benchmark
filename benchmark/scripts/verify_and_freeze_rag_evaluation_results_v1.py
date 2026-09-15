import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


# ============================================================
# 1. PATHS
# ============================================================

DATASET_FILE = Path(
    "benchmark/data/evaluation/"
    "rag_evaluation_dataset_v1.jsonl"
)

CONFIG_FILE = Path(
    "benchmark/config/"
    "rag_evaluator_config_v1.json"
)

RUNTIME_MANIFEST_FILE = Path(
    "benchmark/config/"
    "rag_evaluator_runtime_manifest_v1.json"
)

TASK_PLAN_FILE = Path(
    "benchmark/data/evaluation/"
    "rag_evaluation_task_plan_v1.jsonl"
)

TASK_PLAN_MANIFEST_FILE = Path(
    "benchmark/data/evaluation/"
    "rag_evaluation_task_plan_manifest_v1.json"
)

RESULT_JSONL_FILE = Path(
    "benchmark/results/evaluation/"
    "rag_evaluation_scores_v1.jsonl"
)

RESULT_CSV_FILE = Path(
    "benchmark/results/evaluation/"
    "rag_evaluation_scores_v1.csv"
)

SUMMARY_FILE = Path(
    "benchmark/results/evaluation/"
    "rag_evaluation_scores_summary_v1.json"
)

CHECKPOINT_FILE = Path(
    "benchmark/results/evaluation/"
    "rag_evaluation_scores_v1.checkpoint.json"
)

FREEZE_MANIFEST_FILE = Path(
    "benchmark/results/evaluation/"
    "rag_evaluation_scores_manifest_v1.json"
)


# ============================================================
# 2. EXPECTED FROZEN SOURCE HASHES
# ============================================================

EXPECTED_DATASET_SHA256 = (
    "fe1582a06378e9aa087551a2ee01b4c5"
    "76fb14a38cdb51cc7057439a4e1b27fd"
)

EXPECTED_CONFIG_SHA256 = (
    "60b4cd3d282f347834bd13b2c2dfc722"
    "e5a18ad8939bd3042c41b6c04bc35535"
)

EXPECTED_RUNTIME_SHA256 = (
    "2ebcd5c1ad0770ea6d1b2b1bcaed311"
    "ca21239070fe9902db964afa308cb02ea"
)

EXPECTED_TASK_PLAN_SHA256 = (
    "a689773c5bfedf95c7e3695ad1da77add"
    "5d2b6c7c77371fec35d7f5e76e20240"
)

EXPECTED_TASK_PLAN_MANIFEST_SHA256 = (
    "08ad5cb1c6e7e13ceef9f29776757bcf"
    "da983583d3b4bab9a0d03e5391ef512f"
)


# ============================================================
# 3. EXPECTED FINAL RESULT HASHES
#
# Dari final evaluation output.
# ============================================================

EXPECTED_RESULT_JSONL_SHA256 = (
    "0d66e699ec1f419c132850b13d935b9e"
    "1498a4626235671a6c11667e37ea400b"
)

EXPECTED_RESULT_CSV_SHA256 = (
    "bee69cbba195292c2ef7d82aca7bed530"
    "2ea97a75d6e893fd11db939629dc8b3"
)


# ============================================================
# 4. EXPECTED COUNTS
# ============================================================

EXPECTED_TASK_COUNT = 360
EXPECTED_QUERY_COUNT = 40
EXPECTED_METHOD_COUNT = 3
EXPECTED_METRIC_COUNT = 3

EXPECTED_TECHNICAL_ERROR_ATTEMPTS = 75


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


# ============================================================
# 5. EXPECTED MEANS
#
# Digunakan sebagai additional audit.
#
# Toleransi digunakan karena floating point.
# ============================================================

EXPECTED_MEANS = {
    "bm25": {
        "faithfulness":
            0.937973,

        "answer_relevancy":
            0.841768,

        "answer_accuracy":
            0.781250,
    },

    "e5": {
        "faithfulness":
            0.966675,

        "answer_relevancy":
            0.844610,

        "answer_accuracy":
            0.793750,
    },

    "hybrid_rrf": {
        "faithfulness":
            0.964701,

        "answer_relevancy":
            0.845065,

        "answer_accuracy":
            0.793750,
    },
}

MEAN_TOLERANCE = 0.0000005


# ============================================================
# 6. EXPECTED RESULT SCHEMA
# ============================================================

REQUIRED_RESULT_FIELDS = {
    "task_id",
    "sequence",
    "query_id",
    "method",
    "metric",
    "sample_sha256",
    "value",
    "reason",
    "raw_result",
    "key_label",
    "attempt_count",
    "latency_seconds",
    "completed_at_utc",
}


# ============================================================
# 7. HELPERS
# ============================================================

def sha256_file(
    path: Path,
) -> str:

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


def sample_sha256(
    record,
):

    serialized = json.dumps(
        record,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )

    return hashlib.sha256(
        serialized.encode(
            "utf-8"
        )
    ).hexdigest()


def values_equal(
    value_a,
    value_b,
    tolerance=1e-12,
):

    try:

        a = float(
            value_a
        )

        b = float(
            value_b
        )

        return math.isclose(
            a,
            b,
            rel_tol=0.0,
            abs_tol=tolerance,
        )

    except (
        TypeError,
        ValueError,
    ):

        return (
            value_a
            ==
            value_b
        )


# ============================================================
# 8. MAIN
# ============================================================

def main():

    print(
        "=" * 72
    )

    print(
        "VERIFY & FREEZE "
        "RAG EVALUATION RESULTS V1"
    )

    print(
        "=" * 72
    )

    # ========================================================
    # REQUIRED FILES
    # ========================================================

    required_files = [
        DATASET_FILE,
        CONFIG_FILE,
        RUNTIME_MANIFEST_FILE,
        TASK_PLAN_FILE,
        TASK_PLAN_MANIFEST_FILE,
        RESULT_JSONL_FILE,
        RESULT_CSV_FILE,
        SUMMARY_FILE,
        CHECKPOINT_FILE,
    ]

    for path in required_files:

        if not path.exists():

            raise FileNotFoundError(
                f"File tidak ditemukan: "
                f"{path}"
            )

    # ========================================================
    # SOURCE HASH AUDIT
    # ========================================================

    hashes = {
        "evaluation_dataset":
            sha256_file(
                DATASET_FILE
            ),

        "evaluator_config":
            sha256_file(
                CONFIG_FILE
            ),

        "evaluator_runtime":
            sha256_file(
                RUNTIME_MANIFEST_FILE
            ),

        "task_plan":
            sha256_file(
                TASK_PLAN_FILE
            ),

        "task_plan_manifest":
            sha256_file(
                TASK_PLAN_MANIFEST_FILE
            ),

        "result_jsonl":
            sha256_file(
                RESULT_JSONL_FILE
            ),

        "result_csv":
            sha256_file(
                RESULT_CSV_FILE
            ),

        "summary":
            sha256_file(
                SUMMARY_FILE
            ),

        "checkpoint":
            sha256_file(
                CHECKPOINT_FILE
            ),
    }

    expected_hashes = {
        "evaluation_dataset":
            EXPECTED_DATASET_SHA256,

        "evaluator_config":
            EXPECTED_CONFIG_SHA256,

        "evaluator_runtime":
            EXPECTED_RUNTIME_SHA256,

        "task_plan":
            EXPECTED_TASK_PLAN_SHA256,

        "task_plan_manifest":
            EXPECTED_TASK_PLAN_MANIFEST_SHA256,

        "result_jsonl":
            EXPECTED_RESULT_JSONL_SHA256,

        "result_csv":
            EXPECTED_RESULT_CSV_SHA256,
    }

    print(
        "\nHASH AUDIT"
    )

    for name in [
        "evaluation_dataset",
        "evaluator_config",
        "evaluator_runtime",
        "task_plan",
        "task_plan_manifest",
        "result_jsonl",
        "result_csv",
        "summary",
        "checkpoint",
    ]:

        print(
            f"{name:<24}: "
            f"{hashes[name]}"
        )

        if (
            name
            in expected_hashes
            and
            hashes[name]
            != expected_hashes[name]
        ):

            raise RuntimeError(
                f"{name} hash berubah."
            )

    print(
        "\n✅ Semua expected hash valid."
    )

    # ========================================================
    # LOAD ARTIFACTS
    # ========================================================

    dataset = load_jsonl(
        DATASET_FILE
    )

    tasks = load_jsonl(
        TASK_PLAN_FILE
    )

    results = load_jsonl(
        RESULT_JSONL_FILE
    )

    summary = load_json(
        SUMMARY_FILE
    )

    checkpoint = load_json(
        CHECKPOINT_FILE
    )

    # ========================================================
    # BASIC COUNTS
    # ========================================================

    if (
        len(dataset)
        != 120
    ):

        raise RuntimeError(
            "Evaluation dataset "
            "bukan 120 rows."
        )

    if (
        len(tasks)
        !=
        EXPECTED_TASK_COUNT
    ):

        raise RuntimeError(
            "Task plan bukan "
            "360 tasks."
        )

    if (
        len(results)
        !=
        EXPECTED_TASK_COUNT
    ):

        raise RuntimeError(
            "Final result JSONL "
            "bukan 360 rows."
        )

    # ========================================================
    # DATASET INDEX
    # ========================================================

    dataset_by_pair = {}

    for record in dataset:

        pair = (
            record[
                "query_id"
            ],
            record[
                "method"
            ],
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
        ] = record

    if (
        len(dataset_by_pair)
        != 120
    ):

        raise RuntimeError(
            "Unique query-method "
            "dataset rows bukan 120."
        )

    # ========================================================
    # TASK INDEX
    # ========================================================

    task_by_id = {}

    for task in tasks:

        task_id = (
            task[
                "task_id"
            ]
        )

        if task_id in task_by_id:

            raise RuntimeError(
                f"Duplicate task ID "
                f"di task plan: "
                f"{task_id}"
            )

        task_by_id[
            task_id
        ] = task

    # ========================================================
    # RESULT AUDIT
    # ========================================================

    seen_task_ids = set()
    seen_sequences = set()
    seen_triples = set()

    method_counts = Counter()
    metric_counts = Counter()
    method_metric_counts = Counter()
    query_counts = Counter()

    invalid_scores = []
    task_mismatches = []
    sample_hash_mismatches = []
    sequence_order_errors = []

    grouped_scores = defaultdict(
        list
    )

    for (
        index,
        result,
    ) in enumerate(
        results,
        start=1,
    ):

        # ----------------------------------------------------
        # SCHEMA
        # ----------------------------------------------------

        missing_fields = (
            REQUIRED_RESULT_FIELDS
            -
            set(
                result.keys()
            )
        )

        if missing_fields:

            raise RuntimeError(
                f"Result row {index} "
                f"missing fields: "
                f"{sorted(missing_fields)}"
            )

        task_id = (
            result[
                "task_id"
            ]
        )

        sequence = (
            result[
                "sequence"
            ]
        )

        query_id = (
            result[
                "query_id"
            ]
        )

        method = (
            result[
                "method"
            ]
        )

        metric = (
            result[
                "metric"
            ]
        )

        triple = (
            query_id,
            method,
            metric,
        )

        # ----------------------------------------------------
        # UNIQUE TASK
        # ----------------------------------------------------

        if task_id in seen_task_ids:

            raise RuntimeError(
                f"Duplicate result task_id: "
                f"{task_id}"
            )

        seen_task_ids.add(
            task_id
        )

        # ----------------------------------------------------
        # UNIQUE SEQUENCE
        # ----------------------------------------------------

        if sequence in seen_sequences:

            raise RuntimeError(
                f"Duplicate result sequence: "
                f"{sequence}"
            )

        seen_sequences.add(
            sequence
        )

        # ----------------------------------------------------
        # UNIQUE QUERY-METHOD-METRIC
        # ----------------------------------------------------

        if triple in seen_triples:

            raise RuntimeError(
                f"Duplicate result triple: "
                f"{triple}"
            )

        seen_triples.add(
            triple
        )

        # ----------------------------------------------------
        # TASK MUST EXIST
        # ----------------------------------------------------

        if (
            task_id
            not in task_by_id
        ):

            raise RuntimeError(
                f"Unknown result task: "
                f"{task_id}"
            )

        expected_task = (
            task_by_id[
                task_id
            ]
        )

        # ----------------------------------------------------
        # RESULT MUST MATCH FROZEN TASK PLAN
        # ----------------------------------------------------

        for field in [
            "task_id",
            "sequence",
            "query_id",
            "method",
            "metric",
            "sample_sha256",
        ]:

            if (
                result[
                    field
                ]
                !=
                expected_task[
                    field
                ]
            ):

                task_mismatches.append(
                    {
                        "task_id":
                            task_id,

                        "field":
                            field,

                        "result":
                            result[
                                field
                            ],

                        "expected":
                            expected_task[
                                field
                            ],
                    }
                )

        # ----------------------------------------------------
        # JSONL ORDER MUST FOLLOW TASK PLAN
        # ----------------------------------------------------

        expected_task_at_position = (
            tasks[
                index - 1
            ]
        )

        if (
            task_id
            !=
            expected_task_at_position[
                "task_id"
            ]
        ):

            sequence_order_errors.append(
                {
                    "position":
                        index,

                    "actual":
                        task_id,

                    "expected":
                        expected_task_at_position[
                            "task_id"
                        ],
                }
            )

        # ----------------------------------------------------
        # VERIFY SAMPLE SHA AGAINST FROZEN DATASET
        # ----------------------------------------------------

        pair = (
            query_id,
            method,
        )

        if pair not in dataset_by_pair:

            raise RuntimeError(
                f"Dataset pair "
                f"tidak ditemukan: "
                f"{pair}"
            )

        expected_sample_hash = (
            sample_sha256(
                dataset_by_pair[
                    pair
                ]
            )
        )

        if (
            result[
                "sample_sha256"
            ]
            !=
            expected_sample_hash
        ):

            sample_hash_mismatches.append(
                task_id
            )

        # ----------------------------------------------------
        # METHOD / METRIC
        # ----------------------------------------------------

        if method not in METHODS:

            raise RuntimeError(
                f"Unknown method: "
                f"{method}"
            )

        if metric not in METRICS:

            raise RuntimeError(
                f"Unknown metric: "
                f"{metric}"
            )

        # ----------------------------------------------------
        # SCORE VALIDATION
        # ----------------------------------------------------

        try:

            value = float(
                result[
                    "value"
                ]
            )

        except (
            TypeError,
            ValueError,
        ):

            invalid_scores.append(
                {
                    "task_id":
                        task_id,

                    "value":
                        result[
                            "value"
                        ],

                    "reason":
                        "non_numeric",
                }
            )

            continue

        if not math.isfinite(
            value
        ):

            invalid_scores.append(
                {
                    "task_id":
                        task_id,

                    "value":
                        value,

                    "reason":
                        "non_finite",
                }
            )

            continue

        if metric in {
            "faithfulness",
            "answer_accuracy",
        }:

            if not (
                0.0
                <=
                value
                <=
                1.0
            ):

                invalid_scores.append(
                    {
                        "task_id":
                            task_id,

                        "value":
                            value,

                        "reason":
                            "out_of_range",
                    }
                )

        elif (
            metric
            ==
            "answer_relevancy"
        ):

            if not (
                -1.0
                <=
                value
                <=
                1.0
            ):

                invalid_scores.append(
                    {
                        "task_id":
                            task_id,

                        "value":
                            value,

                        "reason":
                            "out_of_range",
                    }
                )

        # ----------------------------------------------------
        # COUNTS / GROUPS
        # ----------------------------------------------------

        method_counts[
            method
        ] += 1

        metric_counts[
            metric
        ] += 1

        method_metric_counts[
            (
                method,
                metric,
            )
        ] += 1

        query_counts[
            query_id
        ] += 1

        grouped_scores[
            (
                method,
                metric,
            )
        ].append(
            value
        )

    # ========================================================
    # RESULT INTEGRITY
    # ========================================================

    if task_mismatches:

        raise RuntimeError(
            "Result tidak cocok "
            "dengan frozen task plan.\n"
            f"{task_mismatches[:5]}"
        )

    if sample_hash_mismatches:

        raise RuntimeError(
            "Ada sample hash mismatch:\n"
            f"{sample_hash_mismatches[:10]}"
        )

    if sequence_order_errors:

        raise RuntimeError(
            "JSONL ordering berbeda "
            "dari frozen task plan.\n"
            f"{sequence_order_errors[:5]}"
        )

    if invalid_scores:

        raise RuntimeError(
            "Ada invalid final scores:\n"
            f"{invalid_scores[:10]}"
        )

    # ========================================================
    # COMPLETE TASK SET
    # ========================================================

    expected_task_ids = set(
        task_by_id.keys()
    )

    if (
        seen_task_ids
        !=
        expected_task_ids
    ):

        missing = (
            expected_task_ids
            -
            seen_task_ids
        )

        extra = (
            seen_task_ids
            -
            expected_task_ids
        )

        raise RuntimeError(
            "Final result task set "
            "tidak lengkap.\n"
            f"Missing: "
            f"{sorted(missing)}\n"
            f"Extra: "
            f"{sorted(extra)}"
        )

    # ========================================================
    # DISTRIBUTION
    # ========================================================

    for method in METHODS:

        if (
            method_counts[
                method
            ]
            != 120
        ):

            raise RuntimeError(
                f"{method} count "
                f"bukan 120."
            )

    for metric in METRICS:

        if (
            metric_counts[
                metric
            ]
            != 120
        ):

            raise RuntimeError(
                f"{metric} count "
                f"bukan 120."
            )

    for method in METHODS:

        for metric in METRICS:

            if (
                method_metric_counts[
                    (
                        method,
                        metric,
                    )
                ]
                != 40
            ):

                raise RuntimeError(
                    f"{method} × "
                    f"{metric} "
                    f"bukan 40."
                )

    for number in range(
        1,
        EXPECTED_QUERY_COUNT + 1,
    ):

        query_id = (
            f"Q{number:03d}"
        )

        if (
            query_counts[
                query_id
            ]
            != 9
        ):

            raise RuntimeError(
                f"{query_id} "
                "tidak memiliki "
                "9 final scores."
            )

    # ========================================================
    # COMPUTE AGGREGATES FROM RAW RESULTS
    # ========================================================

    computed_aggregates = {}

    print(
        "\n"
        + "=" * 72
    )

    print(
        "RAW RESULT AGGREGATES"
    )

    print(
        "=" * 72
    )

    for method in METHODS:

        computed_aggregates[
            method
        ] = {}

        print(
            f"\n{method}"
        )

        for metric in METRICS:

            values = (
                grouped_scores[
                    (
                        method,
                        metric,
                    )
                ]
            )

            count = len(
                values
            )

            mean_value = (
                math.fsum(
                    values
                )
                /
                count
            )

            minimum = min(
                values
            )

            maximum = max(
                values
            )

            computed_aggregates[
                method
            ][
                metric
            ] = {
                "count":
                    count,

                "mean":
                    mean_value,

                "min":
                    minimum,

                "max":
                    maximum,
            }

            print(
                f"  "
                f"{metric:<20}"
                f": "
                f"{mean_value:.6f}"
            )

            # ------------------------------------------------
            # Compare with console expected mean
            # ------------------------------------------------

            expected_mean = (
                EXPECTED_MEANS[
                    method
                ][
                    metric
                ]
            )

            if not math.isclose(
                mean_value,
                expected_mean,
                rel_tol=0.0,
                abs_tol=(
                    MEAN_TOLERANCE
                ),
            ):

                raise RuntimeError(
                    f"Mean mismatch: "
                    f"{method} × "
                    f"{metric}\n"
                    f"Computed: "
                    f"{mean_value:.12f}\n"
                    f"Expected console: "
                    f"{expected_mean:.6f}"
                )

    print(
        "\n✅ Raw aggregates "
        "match final console output."
    )

    # ========================================================
    # SUMMARY AUDIT
    # ========================================================

    if (
        summary.get(
            "status"
        )
        !=
        "COMPLETE_NOT_YET_FROZEN"
    ):

        raise RuntimeError(
            "Summary status bukan "
            "COMPLETE_NOT_YET_FROZEN."
        )

    if (
        summary.get(
            "task_count"
        )
        !=
        EXPECTED_TASK_COUNT
    ):

        raise RuntimeError(
            "Summary task_count "
            "bukan 360."
        )

    summary_source_hashes = (
        summary.get(
            "source_hashes",
            {},
        )
    )

    expected_summary_source_hashes = {
        "evaluation_dataset":
            hashes[
                "evaluation_dataset"
            ],

        "evaluator_config":
            hashes[
                "evaluator_config"
            ],

        "evaluator_runtime":
            hashes[
                "evaluator_runtime"
            ],

        "task_plan":
            hashes[
                "task_plan"
            ],

        "task_plan_manifest":
            hashes[
                "task_plan_manifest"
            ],
    }

    if (
        summary_source_hashes
        !=
        expected_summary_source_hashes
    ):

        raise RuntimeError(
            "Summary source hashes "
            "tidak cocok."
        )

    summary_jsonl_hash = (
        summary
        .get(
            "outputs",
            {},
        )
        .get(
            "jsonl",
            {},
        )
        .get(
            "sha256"
        )
    )

    summary_csv_hash = (
        summary
        .get(
            "outputs",
            {},
        )
        .get(
            "csv",
            {},
        )
        .get(
            "sha256"
        )
    )

    if (
        summary_jsonl_hash
        !=
        hashes[
            "result_jsonl"
        ]
    ):

        raise RuntimeError(
            "Summary JSONL hash mismatch."
        )

    if (
        summary_csv_hash
        !=
        hashes[
            "result_csv"
        ]
    ):

        raise RuntimeError(
            "Summary CSV hash mismatch."
        )

    # --------------------------------------------------------
    # Summary aggregates vs recomputed raw aggregates
    # --------------------------------------------------------

    summary_aggregates = (
        summary[
            "aggregates"
        ]
    )

    for method in METHODS:

        for metric in METRICS:

            expected = (
                computed_aggregates[
                    method
                ][
                    metric
                ]
            )

            actual = (
                summary_aggregates[
                    method
                ][
                    metric
                ]
            )

            if (
                actual[
                    "count"
                ]
                !=
                expected[
                    "count"
                ]
            ):

                raise RuntimeError(
                    f"Summary aggregate count "
                    f"mismatch: "
                    f"{method} × {metric}"
                )

            for field in [
                "mean",
                "min",
                "max",
            ]:

                if not values_equal(
                    actual[
                        field
                    ],
                    expected[
                        field
                    ],
                    tolerance=1e-12,
                ):

                    raise RuntimeError(
                        f"Summary aggregate "
                        f"{field} mismatch: "
                        f"{method} × {metric}"
                    )

    # --------------------------------------------------------
    # Technical error count
    # --------------------------------------------------------

    technical_error_count = (
        len(
            checkpoint.get(
                "attempt_errors",
                [],
            )
        )
    )

    if (
        technical_error_count
        !=
        EXPECTED_TECHNICAL_ERROR_ATTEMPTS
    ):

        raise RuntimeError(
            "Technical error attempt count "
            "berbeda.\n"
            f"Expected: "
            f"{EXPECTED_TECHNICAL_ERROR_ATTEMPTS}\n"
            f"Actual  : "
            f"{technical_error_count}"
        )

    summary_error_count = (
        summary.get(
            "technical_error_attempts"
        )
    )

    if (
        summary_error_count
        !=
        technical_error_count
    ):

        raise RuntimeError(
            "Summary technical error "
            "count tidak cocok "
            "dengan checkpoint."
        )

    print(
        "✅ Summary valid."
    )

    # ========================================================
    # CHECKPOINT AUDIT
    # ========================================================

    if (
        checkpoint.get(
            "status"
        )
        !=
        "COMPLETE"
    ):

        raise RuntimeError(
            "Checkpoint status "
            "bukan COMPLETE."
        )

    if (
        checkpoint.get(
            "total_tasks"
        )
        !=
        EXPECTED_TASK_COUNT
    ):

        raise RuntimeError(
            "Checkpoint total_tasks "
            "bukan 360."
        )

    checkpoint_completed = (
        checkpoint.get(
            "completed",
            {}
        )
    )

    if (
        len(
            checkpoint_completed
        )
        !=
        EXPECTED_TASK_COUNT
    ):

        raise RuntimeError(
            "Checkpoint completed "
            "bukan 360."
        )

    if (
        set(
            checkpoint_completed.keys()
        )
        !=
        expected_task_ids
    ):

        raise RuntimeError(
            "Checkpoint completed task IDs "
            "tidak sama dengan task plan."
        )

    # --------------------------------------------------------
    # Checkpoint successful result objects must exactly equal
    # final JSONL result objects.
    # --------------------------------------------------------

    for result in results:

        task_id = (
            result[
                "task_id"
            ]
        )

        checkpoint_result = (
            checkpoint_completed[
                task_id
            ]
        )

        if (
            checkpoint_result
            !=
            result
        ):

            raise RuntimeError(
                f"Checkpoint result "
                f"tidak identik "
                f"dengan JSONL: "
                f"{task_id}"
            )

    print(
        "✅ Checkpoint 360/360 "
        "matches final JSONL."
    )

    # ========================================================
    # CSV AUDIT
    # ========================================================

    with RESULT_CSV_FILE.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:

        csv_rows = list(
            csv.DictReader(
                file
            )
        )

    if (
        len(csv_rows)
        !=
        EXPECTED_TASK_COUNT
    ):

        raise RuntimeError(
            "CSV bukan 360 rows."
        )

    csv_fields_to_check = [
        "task_id",
        "sequence",
        "query_id",
        "method",
        "metric",
        "value",
        "key_label",
        "attempt_count",
        "latency_seconds",
        "completed_at_utc",
    ]

    for (
        index,
        (
            csv_row,
            json_result,
        ),
    ) in enumerate(
        zip(
            csv_rows,
            results,
        ),
        start=1,
    ):

        # ----------------------------------------------------
        # String fields
        # ----------------------------------------------------

        for field in [
            "task_id",
            "query_id",
            "method",
            "metric",
            "key_label",
            "completed_at_utc",
        ]:

            expected_value = (
                json_result.get(
                    field
                )
            )

            actual_value = (
                csv_row.get(
                    field
                )
            )

            if (
                str(
                    expected_value
                )
                !=
                str(
                    actual_value
                )
            ):

                raise RuntimeError(
                    f"CSV mismatch "
                    f"row {index}, "
                    f"field {field}."
                )

        # ----------------------------------------------------
        # Integer fields
        # ----------------------------------------------------

        for field in [
            "sequence",
            "attempt_count",
        ]:

            if (
                int(
                    csv_row[
                        field
                    ]
                )
                !=
                int(
                    json_result[
                        field
                    ]
                )
            ):

                raise RuntimeError(
                    f"CSV mismatch "
                    f"row {index}, "
                    f"field {field}."
                )

        # ----------------------------------------------------
        # Float fields
        # ----------------------------------------------------

        for field in [
            "value",
            "latency_seconds",
        ]:

            if not math.isclose(
                float(
                    csv_row[
                        field
                    ]
                ),
                float(
                    json_result[
                        field
                    ]
                ),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):

                raise RuntimeError(
                    f"CSV mismatch "
                    f"row {index}, "
                    f"field {field}."
                )

    print(
        "✅ CSV 360/360 "
        "matches final JSONL."
    )

    # ========================================================
    # KEY USAGE AUDIT
    #
    # Informational only.
    # Key/project is transport, not experimental method.
    # ========================================================

    key_success_counts = Counter(
        result.get(
            "key_label",
            "UNKNOWN",
        )
        for result in results
    )

    error_type_counts = Counter(
        error.get(
            "error_type",
            "UNKNOWN",
        )
        for error in checkpoint.get(
            "attempt_errors",
            [],
        )
    )

    print(
        "\n"
        + "=" * 72
    )

    print(
        "EXECUTION PROVENANCE"
    )

    print(
        "=" * 72
    )

    print(
        f"Successful scoring tasks : "
        f"{len(results)}"
    )

    print(
        f"Technical error attempts : "
        f"{technical_error_count}"
    )

    print(
        "\nSUCCESSFUL TASKS BY KEY LABEL"
    )

    for (
        key_label,
        count,
    ) in sorted(
        key_success_counts.items()
    ):

        print(
            f"{key_label:<25}: "
            f"{count}"
        )

    print(
        "\nTECHNICAL ERRORS BY TYPE"
    )

    for (
        error_type,
        count,
    ) in sorted(
        error_type_counts.items()
    ):

        print(
            f"{error_type:<30}: "
            f"{count}"
        )

    # ========================================================
    # FINAL AUDIT SUMMARY
    # ========================================================

    print(
        "\n"
        + "=" * 72
    )

    print(
        "FINAL RESULT AUDIT"
    )

    print(
        "=" * 72
    )

    print(
        f"Result rows             : "
        f"{len(results)}"
    )

    print(
        f"Unique task IDs         : "
        f"{len(seen_task_ids)}"
    )

    print(
        f"Unique triples          : "
        f"{len(seen_triples)}"
    )

    print(
        f"Unique sequences        : "
        f"{len(seen_sequences)}"
    )

    print(
        f"Invalid scores          : "
        f"{len(invalid_scores)}"
    )

    print(
        f"Task mismatches         : "
        f"{len(task_mismatches)}"
    )

    print(
        f"Sample hash mismatches  : "
        f"{len(sample_hash_mismatches)}"
    )

    print(
        f"Ordering errors         : "
        f"{len(sequence_order_errors)}"
    )

    print(
        f"Technical error attempts: "
        f"{technical_error_count}"
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
        "\nMETRIC COUNTS"
    )

    for metric in METRICS:

        print(
            f"{metric:<20}: "
            f"{metric_counts[metric]}"
        )

    print(
        "\nMETHOD × METRIC"
    )

    for method in METHODS:

        for metric in METRICS:

            print(
                f"{method:<12} × "
                f"{metric:<18}: "
                f"{method_metric_counts[(method, metric)]}"
            )

    # ========================================================
    # CREATE FREEZE MANIFEST
    # ========================================================

    freeze_manifest = {
        "artifact":
            (
                "Aire Optima RAG "
                "Evaluation Scores v1"
            ),

        "status":
            "FROZEN",

        "primary_result": {
            "file":
                str(
                    RESULT_JSONL_FILE
                ),

            "sha256":
                hashes[
                    "result_jsonl"
                ],

            "records":
                len(
                    results
                ),
        },

        "tabular_export": {
            "file":
                str(
                    RESULT_CSV_FILE
                ),

            "sha256":
                hashes[
                    "result_csv"
                ],

            "records":
                len(
                    csv_rows
                ),
        },

        "summary": {
            "file":
                str(
                    SUMMARY_FILE
                ),

            "sha256":
                hashes[
                    "summary"
                ],
        },

        "execution_checkpoint": {
            "file":
                str(
                    CHECKPOINT_FILE
                ),

            "sha256":
                hashes[
                    "checkpoint"
                ],

            "status":
                checkpoint[
                    "status"
                ],

            "successful_tasks":
                len(
                    checkpoint_completed
                ),

            "technical_error_attempts":
                technical_error_count,
        },

        "source_artifacts": {
            "evaluation_dataset": {
                "file":
                    str(
                        DATASET_FILE
                    ),

                "sha256":
                    hashes[
                        "evaluation_dataset"
                    ],
            },

            "evaluator_config": {
                "file":
                    str(
                        CONFIG_FILE
                    ),

                "sha256":
                    hashes[
                        "evaluator_config"
                    ],
            },

            "evaluator_runtime": {
                "file":
                    str(
                        RUNTIME_MANIFEST_FILE
                    ),

                "sha256":
                    hashes[
                        "evaluator_runtime"
                    ],
            },

            "task_plan": {
                "file":
                    str(
                        TASK_PLAN_FILE
                    ),

                "sha256":
                    hashes[
                        "task_plan"
                    ],
            },

            "task_plan_manifest": {
                "file":
                    str(
                        TASK_PLAN_MANIFEST_FILE
                    ),

                "sha256":
                    hashes[
                        "task_plan_manifest"
                    ],
            },
        },

        "validation": {
            "result_rows":
                len(
                    results
                ),

            "unique_task_ids":
                len(
                    seen_task_ids
                ),

            "unique_query_method_metric":
                len(
                    seen_triples
                ),

            "unique_sequences":
                len(
                    seen_sequences
                ),

            "invalid_scores":
                len(
                    invalid_scores
                ),

            "task_mismatches":
                len(
                    task_mismatches
                ),

            "sample_hash_mismatches":
                len(
                    sample_hash_mismatches
                ),

            "ordering_errors":
                len(
                    sequence_order_errors
                ),

            "checkpoint_jsonl_match":
                True,

            "csv_jsonl_match":
                True,

            "method_counts":
                dict(
                    method_counts
                ),

            "metric_counts":
                dict(
                    metric_counts
                ),

            "technical_error_attempts":
                technical_error_count,
        },

        "aggregates": {
            method: {
                metric: {
                    "count":
                        computed_aggregates[
                            method
                        ][
                            metric
                        ][
                            "count"
                        ],

                    "mean":
                        computed_aggregates[
                            method
                        ][
                            metric
                        ][
                            "mean"
                        ],

                    "min":
                        computed_aggregates[
                            method
                        ][
                            metric
                        ][
                            "min"
                        ],

                    "max":
                        computed_aggregates[
                            method
                        ][
                            metric
                        ][
                            "max"
                        ],
                }

                for metric in METRICS
            }

            for method in METHODS
        },

        "interpretation_policy":
            (
                "The 75 technical error attempts are execution "
                "failures that produced no accepted final score. "
                "Every frozen task has exactly one valid accepted "
                "score in the primary result artifact. Technical "
                "retry history is execution provenance and is not "
                "counted as evaluation observations."
            ),

        "freeze_rule":
            (
                "Do not regenerate, replace, retry, remove, "
                "or modify any accepted task score after this "
                "manifest is created. Subsequent statistical "
                "analysis must use the frozen primary JSONL."
            ),
    }

    # ========================================================
    # SAFE MANIFEST WRITE
    # ========================================================

    if FREEZE_MANIFEST_FILE.exists():

        existing_manifest = (
            load_json(
                FREEZE_MANIFEST_FILE
            )
        )

        if (
            existing_manifest
            !=
            freeze_manifest
        ):

            raise RuntimeError(
                "Freeze manifest sudah ada "
                "tetapi berbeda dari hasil "
                "audit saat ini."
            )

        print(
            "\nFreeze manifest "
            "sudah ada dan tetap valid."
        )

    else:

        with FREEZE_MANIFEST_FILE.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                freeze_manifest,
                file,
                ensure_ascii=False,
                indent=2,
            )

    freeze_manifest_hash = (
        sha256_file(
            FREEZE_MANIFEST_FILE
        )
    )

    # ========================================================
    # FINAL
    # ========================================================

    print(
        "\n"
        + "=" * 72
    )

    print(
        "FREEZE COMPLETE"
    )

    print(
        "=" * 72
    )

    print(
        "\nRESULT JSONL SHA256:"
    )

    print(
        hashes[
            "result_jsonl"
        ]
    )

    print(
        "\nRESULT CSV SHA256:"
    )

    print(
        hashes[
            "result_csv"
        ]
    )

    print(
        "\nSUMMARY SHA256:"
    )

    print(
        hashes[
            "summary"
        ]
    )

    print(
        "\nCHECKPOINT SHA256:"
    )

    print(
        hashes[
            "checkpoint"
        ]
    )

    print(
        "\nFREEZE MANIFEST SHA256:"
    )

    print(
        freeze_manifest_hash
    )

    print(
        "\nManifest:"
    )

    print(
        FREEZE_MANIFEST_FILE
    )

    print(
        "\nSTATUS: FROZEN"
    )


if __name__ == "__main__":
    main()