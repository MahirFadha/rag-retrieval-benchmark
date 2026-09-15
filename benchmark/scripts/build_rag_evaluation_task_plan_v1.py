import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


# ============================================================
# PATHS
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

OUTPUT_DIR = Path(
    "benchmark/data/evaluation"
)

TASK_PLAN_JSONL = (
    OUTPUT_DIR
    / "rag_evaluation_task_plan_v1.jsonl"
)

TASK_PLAN_CSV = (
    OUTPUT_DIR
    / "rag_evaluation_task_plan_v1.csv"
)

METADATA_FILE = (
    OUTPUT_DIR
    / "rag_evaluation_task_plan_metadata_v1.json"
)


# ============================================================
# EXPECTED FROZEN HASHES
# ============================================================

EXPECTED_DATASET_SHA256 = (
    "fe1582a06378e9aa087551a2ee01b4c5"
    "76fb14a38cdb51cc7057439a4e1b27fd"
)

EXPECTED_CONFIG_SHA256 = (
    "60b4cd3d282f347834bd13b2c2dfc722"
    "e5a18ad8939bd3042c41b6c04bc35535"
)

EXPECTED_RUNTIME_MANIFEST_SHA256 = (
    "2ebcd5c1ad0770ea6d1b2b1bcaed311"
    "ca21239070fe9902db964afa308cb02ea"
)


# ============================================================
# EXPECTED COUNTS
# ============================================================

EXPECTED_DATASET_ROWS = 120
EXPECTED_QUERIES = 40
EXPECTED_TASKS = 360


BASE_METHODS = [
    "bm25",
    "e5",
    "hybrid_rrf",
]

BASE_METRICS = [
    "faithfulness",
    "answer_relevancy",
    "answer_accuracy",
]


# ============================================================
# SHA256
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


# ============================================================
# LOAD JSONL
# ============================================================

def load_jsonl(
    path: Path,
):

    records = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line_number, line in enumerate(
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


# ============================================================
# ROTATE LIST
# ============================================================

def rotate(
    values,
    offset,
):

    offset = (
        offset
        % len(values)
    )

    return (
        values[offset:]
        +
        values[:offset]
    )


# ============================================================
# SAMPLE HASH
#
# Fingerprint masing-masing frozen query-method row.
#
# Bukan hash file keseluruhan.
# Digunakan runner nanti untuk memastikan task menunjuk
# sample yang benar.
# ============================================================

def sample_sha256(
    record,
):

    serialized = json.dumps(
        record,
        ensure_ascii=False,
        sort_keys=True,
        separators=(
            ",",
            ":",
        ),
    )

    return hashlib.sha256(
        serialized.encode(
            "utf-8"
        )
    ).hexdigest()


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 70
    )

    print(
        "BUILD RAG EVALUATION TASK PLAN V1"
    )

    print(
        "=" * 70
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # FILE CHECK
    # ========================================================

    required_files = [
        DATASET_FILE,
        CONFIG_FILE,
        RUNTIME_MANIFEST_FILE,
    ]

    for path in required_files:

        if not path.exists():

            raise FileNotFoundError(
                f"File tidak ditemukan: "
                f"{path}"
            )

    # ========================================================
    # HASH AUDIT
    # ========================================================

    dataset_hash = (
        sha256_file(
            DATASET_FILE
        )
    )

    config_hash = (
        sha256_file(
            CONFIG_FILE
        )
    )

    runtime_hash = (
        sha256_file(
            RUNTIME_MANIFEST_FILE
        )
    )

    print(
        "\nFROZEN ARTIFACT AUDIT"
    )

    print(
        f"Evaluation dataset : "
        f"{dataset_hash}"
    )

    print(
        f"Evaluator config   : "
        f"{config_hash}"
    )

    print(
        f"Runtime manifest   : "
        f"{runtime_hash}"
    )

    if (
        dataset_hash
        != EXPECTED_DATASET_SHA256
    ):

        raise RuntimeError(
            "Evaluation dataset berubah."
        )

    if (
        config_hash
        != EXPECTED_CONFIG_SHA256
    ):

        raise RuntimeError(
            "Evaluator config berubah."
        )

    if (
        runtime_hash
        != EXPECTED_RUNTIME_MANIFEST_SHA256
    ):

        raise RuntimeError(
            "Evaluator runtime manifest berubah."
        )

    print(
        "\n✅ Semua frozen artifact valid."
    )

    # ========================================================
    # LOAD DATASET
    # ========================================================

    dataset = (
        load_jsonl(
            DATASET_FILE
        )
    )

    if (
        len(dataset)
        != EXPECTED_DATASET_ROWS
    ):

        raise RuntimeError(
            "Evaluation dataset bukan "
            "120 records."
        )

    # ========================================================
    # INDEX DATASET BY QUERY + METHOD
    # ========================================================

    by_pair = {}

    for record in dataset:

        query_id = (
            record[
                "query_id"
            ]
        )

        method = (
            record[
                "method"
            ]
        )

        pair = (
            query_id,
            method,
        )

        if pair in by_pair:

            raise RuntimeError(
                f"Duplicate pair: "
                f"{pair}"
            )

        by_pair[
            pair
        ] = record

    if (
        len(by_pair)
        != 120
    ):

        raise RuntimeError(
            "Unique query-method pairs "
            "bukan 120."
        )

    # ========================================================
    # QUERY IDS
    # ========================================================

    expected_query_ids = [
        f"Q{number:03d}"
        for number in range(
            1,
            41,
        )
    ]

    actual_query_ids = sorted(
        {
            record[
                "query_id"
            ]
            for record
            in dataset
        }
    )

    if (
        actual_query_ids
        != expected_query_ids
    ):

        raise RuntimeError(
            "Query IDs bukan Q001-Q040."
        )

    # ========================================================
    # VERIFY ALL THREE METHODS EXIST FOR EVERY QUERY
    # ========================================================

    for query_id in expected_query_ids:

        for method in BASE_METHODS:

            pair = (
                query_id,
                method,
            )

            if pair not in by_pair:

                raise RuntimeError(
                    "Pair tidak ditemukan: "
                    f"{pair}"
                )

    # ========================================================
    # BUILD DETERMINISTIC TASK PLAN
    #
    # Query selalu Q001 -> Q040.
    #
    # Method starting position dirotasi:
    #
    # Q001: BM25 -> E5 -> Hybrid
    # Q002: E5 -> Hybrid -> BM25
    # Q003: Hybrid -> BM25 -> E5
    # ...
    #
    # Metric starting position juga dirotasi agar satu jenis
    # metric tidak selalu berada pada posisi temporal yang sama.
    #
    # Setiap method tetap mendapat semua 3 metrics.
    # ========================================================

    tasks = []

    sequence = 0

    for query_index, query_id in enumerate(
        expected_query_ids
    ):

        method_order = rotate(
            BASE_METHODS,
            query_index % 3,
        )

        for method_position, method in enumerate(
            method_order
        ):

            sample = (
                by_pair[
                    (
                        query_id,
                        method,
                    )
                ]
            )

            metric_order = rotate(
                BASE_METRICS,
                (
                    query_index
                    +
                    method_position
                )
                % 3,
            )

            for metric in metric_order:

                sequence += 1

                task_id = (
                    f"T{sequence:04d}"
                )

                tasks.append(
                    {
                        "task_id":
                            task_id,

                        "sequence":
                            sequence,

                        "query_id":
                            query_id,

                        "method":
                            method,

                        "metric":
                            metric,

                        "sample_sha256":
                            sample_sha256(
                                sample
                            ),
                    }
                )

    # ========================================================
    # COUNT VALIDATION
    # ========================================================

    if (
        len(tasks)
        != EXPECTED_TASKS
    ):

        raise RuntimeError(
            "Task count bukan 360."
        )

    if (
        len(
            {
                task[
                    "task_id"
                ]
                for task
                in tasks
            }
        )
        != EXPECTED_TASKS
    ):

        raise RuntimeError(
            "Task IDs tidak unik."
        )

    task_triples = {
        (
            task[
                "query_id"
            ],
            task[
                "method"
            ],
            task[
                "metric"
            ],
        )
        for task
        in tasks
    }

    if (
        len(task_triples)
        != EXPECTED_TASKS
    ):

        raise RuntimeError(
            "Query-method-metric "
            "triples tidak unik."
        )

    # ========================================================
    # DISTRIBUTION
    # ========================================================

    metric_counts = Counter(
        task[
            "metric"
        ]
        for task
        in tasks
    )

    method_counts = Counter(
        task[
            "method"
        ]
        for task
        in tasks
    )

    method_metric_counts = Counter(
        (
            task[
                "method"
            ],
            task[
                "metric"
            ],
        )
        for task
        in tasks
    )

    expected_metric_counts = {
        "faithfulness":
            120,

        "answer_relevancy":
            120,

        "answer_accuracy":
            120,
    }

    expected_method_counts = {
        "bm25":
            120,

        "e5":
            120,

        "hybrid_rrf":
            120,
    }

    if (
        dict(metric_counts)
        != expected_metric_counts
    ):

        raise RuntimeError(
            "Metric distribution salah.\n"
            f"{dict(metric_counts)}"
        )

    if (
        dict(method_counts)
        != expected_method_counts
    ):

        raise RuntimeError(
            "Method distribution salah.\n"
            f"{dict(method_counts)}"
        )

    for method in BASE_METHODS:

        for metric in BASE_METRICS:

            count = (
                method_metric_counts[
                    (
                        method,
                        metric,
                    )
                ]
            )

            if count != 40:

                raise RuntimeError(
                    f"{method} × {metric} "
                    f"harus 40, actual {count}."
                )

    # ========================================================
    # EACH QUERY MUST HAVE EXACTLY 9 TASKS
    # ========================================================

    query_task_counts = Counter(
        task[
            "query_id"
        ]
        for task
        in tasks
    )

    for query_id in expected_query_ids:

        if (
            query_task_counts[
                query_id
            ]
            != 9
        ):

            raise RuntimeError(
                f"{query_id} "
                "tidak memiliki 9 tasks."
            )

    # ========================================================
    # WRITE JSONL
    # ========================================================

    with TASK_PLAN_JSONL.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as file:

        for task in tasks:

            file.write(
                json.dumps(
                    task,
                    ensure_ascii=False,
                    separators=(
                        ",",
                        ":",
                    ),
                )
                + "\n"
            )

    # ========================================================
    # WRITE CSV
    # ========================================================

    with TASK_PLAN_CSV.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "task_id",
                "sequence",
                "query_id",
                "method",
                "metric",
                "sample_sha256",
            ],
        )

        writer.writeheader()

        writer.writerows(
            tasks
        )

    # ========================================================
    # PLAN HASH
    # ========================================================

    task_plan_hash = (
        sha256_file(
            TASK_PLAN_JSONL
        )
    )

    csv_hash = (
        sha256_file(
            TASK_PLAN_CSV
        )
    )

    # ========================================================
    # METADATA
    # ========================================================

    metadata = {
        "artifact":
            (
                "Aire Optima RAG "
                "Evaluation Task Plan v1"
            ),

        "status":
            "BUILT_NOT_YET_FROZEN",

        "task_count":
            len(tasks),

        "query_count":
            EXPECTED_QUERIES,

        "query_method_rows":
            EXPECTED_DATASET_ROWS,

        "source_artifacts": {
            "evaluation_dataset": {
                "file":
                    str(
                        DATASET_FILE
                    ),

                "sha256":
                    dataset_hash,
            },

            "evaluator_config": {
                "file":
                    str(
                        CONFIG_FILE
                    ),

                "sha256":
                    config_hash,
            },

            "evaluator_runtime": {
                "file":
                    str(
                        RUNTIME_MANIFEST_FILE
                    ),

                "sha256":
                    runtime_hash,
            },
        },

        "task_distribution": {
            "metrics":
                dict(
                    metric_counts
                ),

            "methods":
                dict(
                    method_counts
                ),

            "tasks_per_query":
                9,

            "tasks_per_method_metric":
                40,
        },

        "ordering_policy": {
            "query_order":
                "Q001 through Q040",

            "method_order":
                (
                    "Three-position deterministic "
                    "rotation by query index."
                ),

            "metric_order":
                (
                    "Three-position deterministic "
                    "rotation by query index and "
                    "method position."
                ),

            "random_shuffle":
                False,
        },

        "task_schema": {
            "task_id":
                "Unique fixed task identifier",

            "sequence":
                "Deterministic execution order",

            "query_id":
                "Frozen benchmark query ID",

            "method":
                (
                    "bm25, e5, or hybrid_rrf"
                ),

            "metric":
                (
                    "faithfulness, "
                    "answer_relevancy, "
                    "or answer_accuracy"
                ),

            "sample_sha256":
                (
                    "Canonical SHA256 fingerprint "
                    "of source evaluation row"
                ),
        },

        "outputs": {
            "jsonl": {
                "file":
                    str(
                        TASK_PLAN_JSONL
                    ),

                "sha256":
                    task_plan_hash,
            },

            "csv": {
                "file":
                    str(
                        TASK_PLAN_CSV
                    ),

                "sha256":
                    csv_hash,
            },
        },

        "freeze_rule":
            (
                "Do not alter task identity, "
                "task ordering, query-method-metric "
                "mapping, or source sample after "
                "final scoring begins."
            ),
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
    # SUMMARY
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "TASK PLAN SUMMARY"
    )

    print(
        "=" * 70
    )

    print(
        f"Queries          : "
        f"{EXPECTED_QUERIES}"
    )

    print(
        f"Query-method rows: "
        f"{EXPECTED_DATASET_ROWS}"
    )

    print(
        f"Total tasks      : "
        f"{len(tasks)}"
    )

    print(
        "\nMETRIC COUNTS"
    )

    for metric in BASE_METRICS:

        print(
            f"{metric:<20}: "
            f"{metric_counts[metric]}"
        )

    print(
        "\nMETHOD COUNTS"
    )

    for method in BASE_METHODS:

        print(
            f"{method:<20}: "
            f"{method_counts[method]}"
        )

    print(
        "\nMETHOD × METRIC"
    )

    for method in BASE_METHODS:

        for metric in BASE_METRICS:

            print(
                f"{method:<12} × "
                f"{metric:<18}: "
                f"{method_metric_counts[(method, metric)]}"
            )

    print(
        "\nFIRST 12 TASKS"
    )

    for task in tasks[:12]:

        print(
            f"{task['task_id']} | "
            f"{task['query_id']} | "
            f"{task['method']:<10} | "
            f"{task['metric']}"
        )

    print(
        "\nTASK PLAN SHA256:"
    )

    print(
        task_plan_hash
    )

    print(
        "\nJSONL:"
    )

    print(
        TASK_PLAN_JSONL
    )

    print(
        "\nCSV:"
    )

    print(
        TASK_PLAN_CSV
    )

    print(
        "\nMetadata:"
    )

    print(
        METADATA_FILE
    )

    print(
        "\nSTATUS:"
    )

    print(
        "BUILT - BELUM FROZEN"
    )


if __name__ == "__main__":
    main()