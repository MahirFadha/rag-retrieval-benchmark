import hashlib
import json
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

TASK_PLAN_METADATA_FILE = Path(
    "benchmark/data/evaluation/"
    "rag_evaluation_task_plan_metadata_v1.json"
)

TASK_PLAN_MANIFEST_FILE = Path(
    "benchmark/data/evaluation/"
    "rag_evaluation_task_plan_manifest_v1.json"
)


# ============================================================
# 2. EXPECTED FROZEN HASHES
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


# ============================================================
# 3. EXPECTED VALUES
# ============================================================

EXPECTED_DATASET_ROWS = 120
EXPECTED_QUERY_COUNT = 40
EXPECTED_TASK_COUNT = 360

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

EXPECTED_TASK_KEYS = {
    "task_id",
    "sequence",
    "query_id",
    "method",
    "metric",
    "sample_sha256",
}


# ============================================================
# 4. HELPERS
# ============================================================

def sha256_file(path: Path) -> str:

    hasher = hashlib.sha256()

    with path.open("rb") as file:

        while True:

            chunk = file.read(
                1024 * 1024
            )

            if not chunk:
                break

            hasher.update(chunk)

    return hasher.hexdigest()


def load_jsonl(path: Path):

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
                    json.loads(line)
                )

            except json.JSONDecodeError as exc:

                raise RuntimeError(
                    f"Invalid JSON pada "
                    f"{path}, "
                    f"line {line_number}."
                ) from exc

    return records


def sample_sha256(record) -> str:

    serialized = json.dumps(
        record,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )

    return hashlib.sha256(
        serialized.encode("utf-8")
    ).hexdigest()


def rotate(values, offset):

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
# 5. BUILD EXPECTED TASK SEQUENCE
#
# Menghasilkan kembali task plan secara deterministik
# dari frozen dataset.
#
# Jika satu task saja berubah posisi/metric/method,
# comparison akan gagal.
# ============================================================

def build_expected_tasks(
    dataset_by_pair,
):

    expected_query_ids = [
        f"Q{number:03d}"
        for number in range(
            1,
            41,
        )
    ]

    tasks = []

    sequence = 0

    for query_index, query_id in enumerate(
        expected_query_ids
    ):

        method_order = rotate(
            METHODS,
            query_index % 3,
        )

        for method_position, method in enumerate(
            method_order
        ):

            pair = (
                query_id,
                method,
            )

            if pair not in dataset_by_pair:

                raise RuntimeError(
                    f"Dataset pair tidak ditemukan: "
                    f"{pair}"
                )

            sample = (
                dataset_by_pair[
                    pair
                ]
            )

            metric_order = rotate(
                METRICS,
                (
                    query_index
                    +
                    method_position
                )
                % 3,
            )

            for metric in metric_order:

                sequence += 1

                tasks.append(
                    {
                        "task_id":
                            f"T{sequence:04d}",

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

    return tasks


# ============================================================
# 6. MAIN
# ============================================================

def main():

    print(
        "=" * 70
    )

    print(
        "VERIFY & FREEZE "
        "RAG EVALUATION TASK PLAN V1"
    )

    print(
        "=" * 70
    )

    # ========================================================
    # FILE EXISTENCE
    # ========================================================

    required_files = [
        DATASET_FILE,
        CONFIG_FILE,
        RUNTIME_MANIFEST_FILE,
        TASK_PLAN_FILE,
        TASK_PLAN_METADATA_FILE,
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

    task_plan_hash = (
        sha256_file(
            TASK_PLAN_FILE
        )
    )

    metadata_hash = (
        sha256_file(
            TASK_PLAN_METADATA_FILE
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

    print(
        f"Task plan          : "
        f"{task_plan_hash}"
    )

    print(
        f"Task metadata      : "
        f"{metadata_hash}"
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
        != EXPECTED_RUNTIME_SHA256
    ):

        raise RuntimeError(
            "Evaluator runtime berubah."
        )

    if (
        task_plan_hash
        != EXPECTED_TASK_PLAN_SHA256
    ):

        raise RuntimeError(
            "Task plan hash berubah."
        )

    print(
        "\n✅ Semua frozen/source hash valid."
    )

    # ========================================================
    # LOAD DATA
    # ========================================================

    dataset = (
        load_jsonl(
            DATASET_FILE
        )
    )

    task_plan = (
        load_jsonl(
            TASK_PLAN_FILE
        )
    )

    if (
        len(dataset)
        != EXPECTED_DATASET_ROWS
    ):

        raise RuntimeError(
            "Evaluation dataset "
            "bukan 120 rows."
        )

    if (
        len(task_plan)
        != EXPECTED_TASK_COUNT
    ):

        raise RuntimeError(
            "Task plan bukan "
            "360 tasks."
        )

    # ========================================================
    # INDEX DATASET
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

        if pair in dataset_by_pair:

            raise RuntimeError(
                f"Duplicate dataset pair: "
                f"{pair}"
            )

        dataset_by_pair[
            pair
        ] = record

    if (
        len(dataset_by_pair)
        != EXPECTED_DATASET_ROWS
    ):

        raise RuntimeError(
            "Unique dataset pair "
            "bukan 120."
        )

    # ========================================================
    # TASK SCHEMA + BASIC AUDIT
    # ========================================================

    seen_task_ids = set()
    seen_sequences = set()
    seen_triples = set()

    method_counts = Counter()
    metric_counts = Counter()
    method_metric_counts = Counter()
    query_counts = Counter()

    query_methods = defaultdict(set)
    query_metrics = defaultdict(set)

    invalid_sample_hashes = []

    for index, task in enumerate(
        task_plan,
        start=1,
    ):

        actual_keys = set(
            task.keys()
        )

        if (
            actual_keys
            != EXPECTED_TASK_KEYS
        ):

            raise RuntimeError(
                f"Task {index} schema berbeda.\n"
                f"Expected: "
                f"{sorted(EXPECTED_TASK_KEYS)}\n"
                f"Actual: "
                f"{sorted(actual_keys)}"
            )

        task_id = (
            task[
                "task_id"
            ]
        )

        sequence = (
            task[
                "sequence"
            ]
        )

        query_id = (
            task[
                "query_id"
            ]
        )

        method = (
            task[
                "method"
            ]
        )

        metric = (
            task[
                "metric"
            ]
        )

        triple = (
            query_id,
            method,
            metric,
        )

        # ----------------------------------------------------
        # UNIQUE TASK ID
        # ----------------------------------------------------

        if task_id in seen_task_ids:

            raise RuntimeError(
                f"Duplicate task_id: "
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
                f"Duplicate sequence: "
                f"{sequence}"
            )

        seen_sequences.add(
            sequence
        )

        # ----------------------------------------------------
        # UNIQUE TRIPLE
        # ----------------------------------------------------

        if triple in seen_triples:

            raise RuntimeError(
                f"Duplicate query-method-metric: "
                f"{triple}"
            )

        seen_triples.add(
            triple
        )

        # ----------------------------------------------------
        # VALID METHOD / METRIC
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
        # VALID QUERY ID
        # ----------------------------------------------------

        if query_id not in {
            f"Q{x:03d}"
            for x in range(
                1,
                41,
            )
        }:

            raise RuntimeError(
                f"Unknown query_id: "
                f"{query_id}"
            )

        # ----------------------------------------------------
        # SAMPLE MUST EXIST
        # ----------------------------------------------------

        pair = (
            query_id,
            method,
        )

        if pair not in dataset_by_pair:

            raise RuntimeError(
                f"Task menunjuk "
                f"dataset pair tidak ada: "
                f"{pair}"
            )

        # ----------------------------------------------------
        # SAMPLE HASH MUST MATCH
        # ----------------------------------------------------

        expected_sample_hash = (
            sample_sha256(
                dataset_by_pair[
                    pair
                ]
            )
        )

        if (
            task[
                "sample_sha256"
            ]
            != expected_sample_hash
        ):

            invalid_sample_hashes.append(
                task_id
            )

        # ----------------------------------------------------
        # COUNTS
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

        query_methods[
            query_id
        ].add(
            method
        )

        query_metrics[
            query_id
        ].add(
            metric
        )

    # ========================================================
    # SAMPLE HASH VALIDATION
    # ========================================================

    if invalid_sample_hashes:

        raise RuntimeError(
            "Ada task dengan "
            "sample_sha256 tidak valid:\n"
            f"{invalid_sample_hashes}"
        )

    # ========================================================
    # EXPECTED SEQUENCE NUMBERS
    # ========================================================

    expected_sequences = set(
        range(
            1,
            EXPECTED_TASK_COUNT + 1,
        )
    )

    if (
        seen_sequences
        != expected_sequences
    ):

        missing = (
            expected_sequences
            - seen_sequences
        )

        extra = (
            seen_sequences
            - expected_sequences
        )

        raise RuntimeError(
            "Sequence tidak lengkap.\n"
            f"Missing: "
            f"{sorted(missing)}\n"
            f"Extra: "
            f"{sorted(extra)}"
        )

    # ========================================================
    # EXPECTED TASK IDS
    # ========================================================

    expected_task_ids = {
        f"T{x:04d}"
        for x in range(
            1,
            EXPECTED_TASK_COUNT + 1,
        )
    }

    if (
        seen_task_ids
        != expected_task_ids
    ):

        raise RuntimeError(
            "Task IDs bukan "
            "T0001-T0360."
        )

    # ========================================================
    # DISTRIBUTION AUDIT
    # ========================================================

    expected_method_counts = {
        "bm25": 120,
        "e5": 120,
        "hybrid_rrf": 120,
    }

    expected_metric_counts = {
        "faithfulness": 120,
        "answer_relevancy": 120,
        "answer_accuracy": 120,
    }

    if (
        dict(method_counts)
        != expected_method_counts
    ):

        raise RuntimeError(
            "Method distribution berubah.\n"
            f"{dict(method_counts)}"
        )

    if (
        dict(metric_counts)
        != expected_metric_counts
    ):

        raise RuntimeError(
            "Metric distribution berubah.\n"
            f"{dict(metric_counts)}"
        )

    for method in METHODS:

        for metric in METRICS:

            actual = (
                method_metric_counts[
                    (
                        method,
                        metric,
                    )
                ]
            )

            if actual != 40:

                raise RuntimeError(
                    f"{method} × {metric} "
                    f"harus 40, "
                    f"actual {actual}."
                )

    # ========================================================
    # QUERY AUDIT
    # ========================================================

    for number in range(
        1,
        41,
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
                f"{query_id} tidak "
                "memiliki 9 tasks."
            )

        if (
            query_methods[
                query_id
            ]
            != set(METHODS)
        ):

            raise RuntimeError(
                f"{query_id}: "
                "3 methods tidak lengkap."
            )

        if (
            query_metrics[
                query_id
            ]
            != set(METRICS)
        ):

            raise RuntimeError(
                f"{query_id}: "
                "3 metrics tidak lengkap."
            )

    # ========================================================
    # DETERMINISTIC ORDER RECONSTRUCTION
    #
    # Ini audit terkuat.
    #
    # Kita rebuild seluruh 360 task dari frozen dataset.
    # Hasilnya harus identik record-per-record.
    # ========================================================

    expected_tasks = (
        build_expected_tasks(
            dataset_by_pair
        )
    )

    if (
        len(expected_tasks)
        != len(task_plan)
    ):

        raise RuntimeError(
            "Expected task count berbeda."
        )

    ordering_errors = []

    for index, (
        actual,
        expected,
    ) in enumerate(
        zip(
            task_plan,
            expected_tasks,
        ),
        start=1,
    ):

        if actual != expected:

            ordering_errors.append(
                {
                    "index":
                        index,

                    "actual":
                        actual,

                    "expected":
                        expected,
                }
            )

            # Tidak perlu memenuhi output
            # dengan ratusan mismatch.
            if len(
                ordering_errors
            ) >= 5:

                break

    if ordering_errors:

        raise RuntimeError(
            "Task ordering/content "
            "tidak sesuai deterministic "
            "generation policy.\n"
            f"{ordering_errors}"
        )

    # ========================================================
    # METADATA AUDIT
    # ========================================================

    with TASK_PLAN_METADATA_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:

        metadata = json.load(
            file
        )

    if (
        metadata.get(
            "task_count"
        )
        != EXPECTED_TASK_COUNT
    ):

        raise RuntimeError(
            "Metadata task_count "
            "bukan 360."
        )

    if (
        metadata.get(
            "query_count"
        )
        != EXPECTED_QUERY_COUNT
    ):

        raise RuntimeError(
            "Metadata query_count "
            "bukan 40."
        )

    metadata_output_hash = (
        metadata
        .get(
            "outputs",
            {}
        )
        .get(
            "jsonl",
            {}
        )
        .get(
            "sha256"
        )
    )

    if (
        metadata_output_hash
        != task_plan_hash
    ):

        raise RuntimeError(
            "Metadata task plan hash "
            "tidak sama dengan file."
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "FINAL TASK PLAN AUDIT"
    )

    print(
        "=" * 70
    )

    print(
        f"Tasks               : "
        f"{len(task_plan)}"
    )

    print(
        f"Unique task IDs     : "
        f"{len(seen_task_ids)}"
    )

    print(
        f"Unique triples      : "
        f"{len(seen_triples)}"
    )

    print(
        f"Sequences           : "
        f"{len(seen_sequences)}"
    )

    print(
        f"Invalid sample hash : "
        f"{len(invalid_sample_hashes)}"
    )

    print(
        f"Ordering errors     : "
        f"{len(ordering_errors)}"
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

    manifest = {
        "artifact":
            (
                "Aire Optima RAG "
                "Evaluation Task Plan v1"
            ),

        "status":
            "FROZEN",

        "task_plan": {
            "file":
                str(
                    TASK_PLAN_FILE
                ),

            "sha256":
                task_plan_hash,

            "task_count":
                len(
                    task_plan
                ),

            "unique_task_ids":
                len(
                    seen_task_ids
                ),

            "unique_query_method_metric":
                len(
                    seen_triples
                ),
        },

        "task_plan_metadata": {
            "file":
                str(
                    TASK_PLAN_METADATA_FILE
                ),

            "sha256":
                metadata_hash,
        },

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

        "validation": {
            "task_count":
                len(
                    task_plan
                ),

            "unique_tasks":
                len(
                    seen_task_ids
                ),

            "unique_triples":
                len(
                    seen_triples
                ),

            "invalid_sample_hashes":
                len(
                    invalid_sample_hashes
                ),

            "ordering_errors":
                len(
                    ordering_errors
                ),

            "method_counts":
                dict(
                    method_counts
                ),

            "metric_counts":
                dict(
                    metric_counts
                ),

            "tasks_per_query":
                9,

            "tasks_per_method_metric":
                40,
        },

        "ordering_policy": {
            "query_order":
                "Q001-Q040",

            "method_rotation":
                (
                    "Deterministic "
                    "three-position rotation "
                    "by query index."
                ),

            "metric_rotation":
                (
                    "Deterministic "
                    "three-position rotation "
                    "by query index and "
                    "method position."
                ),

            "randomized":
                False,
        },

        "execution_rule":
            (
                "Final evaluator must execute "
                "only tasks defined in this "
                "frozen task plan. Completed "
                "successful tasks must not be "
                "re-scored during resume."
            ),

        "freeze_rule":
            (
                "Do not alter task IDs, "
                "sequence, query ID, method, "
                "metric, sample fingerprint, "
                "or ordering after final "
                "evaluation begins."
            ),
    }

    # ========================================================
    # SAFE WRITE / VERIFY EXISTING MANIFEST
    # ========================================================

    if TASK_PLAN_MANIFEST_FILE.exists():

        with TASK_PLAN_MANIFEST_FILE.open(
            "r",
            encoding="utf-8",
        ) as file:

            existing = json.load(
                file
            )

        if existing != manifest:

            raise RuntimeError(
                "Task plan manifest "
                "sudah ada tetapi isinya "
                "berbeda dari audit saat ini."
            )

        print(
            "\nTask plan manifest "
            "sudah ada dan tetap valid."
        )

    else:

        with TASK_PLAN_MANIFEST_FILE.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                manifest,
                file,
                ensure_ascii=False,
                indent=2,
            )

    manifest_hash = (
        sha256_file(
            TASK_PLAN_MANIFEST_FILE
        )
    )

    # ========================================================
    # FINAL
    # ========================================================

    print(
        "\nTASK PLAN SHA256:"
    )

    print(
        task_plan_hash
    )

    print(
        "\nTASK PLAN MANIFEST SHA256:"
    )

    print(
        manifest_hash
    )

    print(
        "\nManifest:"
    )

    print(
        TASK_PLAN_MANIFEST_FILE
    )

    print(
        "\nSTATUS: FROZEN"
    )


if __name__ == "__main__":
    main()