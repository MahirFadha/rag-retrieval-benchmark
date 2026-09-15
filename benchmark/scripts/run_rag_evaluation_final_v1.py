import argparse
import asyncio
import csv
import hashlib
import json
import math
import os
import platform
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import instructor
from dotenv import load_dotenv
from google import genai

from ragas.embeddings import GoogleEmbeddings
from ragas.llms import InstructorLLM
from ragas.metrics.collections import (
    AnswerAccuracy,
    AnswerRelevancy,
    Faithfulness,
)


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

OUTPUT_DIR = Path(
    "benchmark/results/evaluation"
)

CHECKPOINT_FILE = (
    OUTPUT_DIR
    / "rag_evaluation_scores_v1.checkpoint.json"
)

FINAL_JSONL = (
    OUTPUT_DIR
    / "rag_evaluation_scores_v1.jsonl"
)

FINAL_CSV = (
    OUTPUT_DIR
    / "rag_evaluation_scores_v1.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "rag_evaluation_scores_summary_v1.json"
)


# ============================================================
# 2. FROZEN HASHES
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
# 3. CONSTANTS
# ============================================================

EXPECTED_TASK_COUNT = 360
EXPECTED_SAMPLE_COUNT = 120

PAID_API_KEY_ENV = (
    "GEMINI_API_KEY_TIER1"
)

PAID_KEY_LABEL = (
    "GEMINI_API_KEY_TIER1"
)

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
# RETRY POLICY
#
# Hanya technical retry.
#
# Valid score TIDAK PERNAH diulang karena kualitas.
# ============================================================

QUOTA_WAIT_SECONDS = 70.0

TRANSIENT_WAIT_SECONDS = 20.0

STRUCTURED_OUTPUT_WAIT_SECONDS = 5.0

INVALID_METRIC_WAIT_SECONDS = 5.0


# ============================================================
# 4. CUSTOM TECHNICAL ERRORS
# ============================================================

class InvalidMetricResultError(RuntimeError):
    """
    Evaluator selesai tetapi tidak menghasilkan numerical
    score yang valid, misalnya None, NaN, atau Infinity.

    Ini bukan score penelitian.
    """

    pass


# ============================================================
# 5. TIME
# ============================================================

def utc_now():

    return datetime.now(
        timezone.utc
    ).isoformat()


# ============================================================
# 6. SHA256
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
# 7. JSONL
# ============================================================

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

            line = (
                line.strip()
            )

            if not line:
                continue

            try:

                records.append(
                    json.loads(
                        line
                    )
                )

            except (
                json.JSONDecodeError
            ) as exc:

                raise RuntimeError(
                    f"Invalid JSON pada "
                    f"{path}, "
                    f"line {line_number}."
                ) from exc

    return records


# ============================================================
# 8. ATOMIC JSON WRITE
# ============================================================

def atomic_write_json(
    path: Path,
    data,
):

    temporary_path = (
        path.with_suffix(
            path.suffix
            +
            ".tmp"
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

        os.fsync(
            file.fileno()
        )

    os.replace(
        temporary_path,
        path,
    )


# ============================================================
# 9. SAMPLE SHA256
# ============================================================

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


# ============================================================
# 10. LOAD SINGLE PAID API KEY
#
# Runner sengaja HANYA membaca:
#
# GEMINI_API_KEY_TIER1
#
# Key-key lama:
#
# GEMINI_API_KEY
# GEMINI_API_KEY_2
# ...
#
# DIABAIKAN.
# ============================================================

def load_paid_api_key():

    value = os.getenv(
        PAID_API_KEY_ENV
    )

    if value is None:

        raise RuntimeError(
            f"{PAID_API_KEY_ENV} "
            "tidak ditemukan di .env."
        )

    value = (
        value.strip()
    )

    if not value:

        raise RuntimeError(
            f"{PAID_API_KEY_ENV} "
            "kosong."
        )

    return {
        "label":
            PAID_KEY_LABEL,

        "value":
            value,
    }


# ============================================================
# 11. NORMALIZE METRIC RESULT
# ============================================================

def normalize_metric_result(
    result,
):

    if hasattr(
        result,
        "model_dump",
    ):

        try:

            return result.model_dump(
                mode="json"
            )

        except Exception:
            pass

    value = getattr(
        result,
        "value",
        None,
    )

    reason = getattr(
        result,
        "reason",
        None,
    )

    return {
        "value":
            value,

        "reason":
            (
                None
                if reason is None
                else str(reason)
            ),
    }


# ============================================================
# 12. SCORE VALIDATION
# ============================================================

def validate_score(
    metric,
    value,
):

    # ========================================================
    # NONE
    # ========================================================

    if value is None:

        raise InvalidMetricResultError(
            f"{metric} menghasilkan "
            "score None."
        )

    # ========================================================
    # NUMERIC
    # ========================================================

    try:

        value = float(
            value
        )

    except (
        TypeError,
        ValueError,
    ) as exc:

        raise InvalidMetricResultError(
            f"{metric} menghasilkan "
            f"score non-numeric: "
            f"{value}"
        ) from exc

    # ========================================================
    # NaN / Inf
    # ========================================================

    if not math.isfinite(
        value
    ):

        raise InvalidMetricResultError(
            f"{metric} menghasilkan "
            f"non-finite score: "
            f"{value}"
        )

    # ========================================================
    # VALID RANGE
    # ========================================================

    ranges = {
        "faithfulness":
            (
                0.0,
                1.0,
            ),

        "answer_relevancy":
            (
                -1.0,
                1.0,
            ),

        "answer_accuracy":
            (
                0.0,
                1.0,
            ),
    }

    minimum, maximum = (
        ranges[
            metric
        ]
    )

    if not (
        minimum
        <=
        value
        <=
        maximum
    ):

        raise InvalidMetricResultError(
            f"{metric} menghasilkan "
            f"score di luar range "
            f"{minimum} sampai "
            f"{maximum}: "
            f"{value}"
        )

    return value


# ============================================================
# 13. ERROR STATUS EXTRACTION
# ============================================================

def get_error_code(
    exc,
):

    for attribute in [
        "status_code",
        "code",
    ]:

        value = getattr(
            exc,
            attribute,
            None,
        )

        if isinstance(
            value,
            int,
        ):

            return value

    return None


# ============================================================
# 14. ERROR CLASSIFICATION
#
# Instructor dapat membungkus error asli dalam
# InstructorRetryException.
#
# Karena itu selain status_code kita juga membaca text
# nested exception.
# ============================================================

def classify_error(
    exc,
):

    # ========================================================
    # INVALID METRIC RESULT
    # ========================================================

    if isinstance(
        exc,
        InvalidMetricResultError,
    ):

        return (
            "INVALID_METRIC_RESULT",
            INVALID_METRIC_WAIT_SECONDS,
        )

    code = get_error_code(
        exc
    )

    text = (
        f"{type(exc).__name__}: "
        f"{exc}"
    ).lower()

    # ========================================================
    # AUTH / KEY FATAL
    # ========================================================

    key_fatal_terms = [
        "401 unauthenticated",
        "unauthenticated",
        "api key not valid",
        "api_key_invalid",
        "account_state_invalid",
        "service account is deleted",
        "service account is disabled",
        "service account bound to the api key must be active",
        "permission_denied",
    ]

    if (
        code in {
            401,
            403,
        }
        or
        any(
            term in text
            for term
            in key_fatal_terms
        )
    ):

        return (
            "KEY_FATAL",
            0.0,
        )

    # ========================================================
    # QUOTA / RATE LIMIT
    # ========================================================

    quota_terms = [
        "429 resource_exhausted",
        "resource_exhausted",
        "rate_limit_exceeded",
        "rate limit",
        "quota exceeded",
        "too many requests",
    ]

    if (
        code == 429
        or
        any(
            term in text
            for term
            in quota_terms
        )
    ):

        return (
            "QUOTA",
            QUOTA_WAIT_SECONDS,
        )

    # ========================================================
    # TEMPORARY PROVIDER / NETWORK
    # ========================================================

    transient_terms = [
        "503 unavailable",
        "high demand",
        "temporarily unavailable",
        "service unavailable",
        "internal server error",
        "deadline exceeded",
        "timeout",
        "timed out",
        "connection reset",
        "connection aborted",
        "connection error",
        "bad gateway",
        "502",
        "503",
        "504",
    ]

    if (
        (
            code is not None
            and
            500 <= code <= 599
        )
        or
        any(
            term in text
            for term
            in transient_terms
        )
    ):

        return (
            "TRANSIENT",
            TRANSIENT_WAIT_SECONDS,
        )

    # ========================================================
    # STRUCTURED OUTPUT / PARSE FAILURE
    # ========================================================

    structured_terms = [
        "validation error",
        "json decode",
        "jsondecodeerror",
        "failed to parse",
        "parse error",
        "response model",
        "could not parse",
    ]

    if any(
        term in text
        for term
        in structured_terms
    ):

        return (
            "STRUCTURED_OUTPUT",
            STRUCTURED_OUTPUT_WAIT_SECONDS,
        )

    # ========================================================
    # PROGRAMMING / CONFIG ERROR
    # ========================================================

    return (
        "GLOBAL_FATAL",
        0.0,
    )


# ============================================================
# 15. SINGLE PAID KEY RESOURCE
# ============================================================

class PaidKeyResource:

    def __init__(
        self,
        key_record,
        judge_model,
        embedding_model,
        strictness,
    ):

        self.label = (
            key_record[
                "label"
            ]
        )

        self.base_client = (
            genai.Client(
                api_key=(
                    key_record[
                        "value"
                    ]
                )
            )
        )

        self.instructor_client = (
            instructor.from_genai(
                self.base_client,
                use_async=True,
            )
        )

        self.llm = InstructorLLM(
            client=(
                self.instructor_client
            ),
            model=judge_model,
            provider="google",
        )

        if not self.llm.is_async:

            raise RuntimeError(
                "InstructorLLM "
                "tidak async."
            )

        self.embeddings = (
            GoogleEmbeddings(
                client=(
                    self.base_client
                ),
                model=embedding_model,
            )
        )

        self.faithfulness = (
            Faithfulness(
                llm=self.llm
            )
        )

        self.answer_relevancy = (
            AnswerRelevancy(
                llm=self.llm,
                embeddings=(
                    self.embeddings
                ),
                strictness=strictness,
            )
        )

        self.answer_accuracy = (
            AnswerAccuracy(
                llm=self.llm
            )
        )

    async def close(
        self,
    ):

        try:

            await (
                self.base_client
                .aio
                .aclose()
            )

        except Exception:
            pass

        try:

            self.base_client.close()

        except Exception:
            pass


# ============================================================
# 16. EXECUTE ONE METRIC
# ============================================================

async def execute_metric(
    resource,
    metric,
    sample,
):

    # ========================================================
    # FAITHFULNESS
    # ========================================================

    if metric == "faithfulness":

        return await (
            resource
            .faithfulness
            .ascore(
                user_input=(
                    sample[
                        "user_input"
                    ]
                ),

                response=(
                    sample[
                        "response"
                    ]
                ),

                retrieved_contexts=(
                    sample[
                        "retrieved_contexts"
                    ]
                ),
            )
        )

    # ========================================================
    # ANSWER RELEVANCY
    # ========================================================

    if metric == "answer_relevancy":

        return await (
            resource
            .answer_relevancy
            .ascore(
                user_input=(
                    sample[
                        "user_input"
                    ]
                ),

                response=(
                    sample[
                        "response"
                    ]
                ),
            )
        )

    # ========================================================
    # ANSWER ACCURACY
    # ========================================================

    if metric == "answer_accuracy":

        return await (
            resource
            .answer_accuracy
            .ascore(
                user_input=(
                    sample[
                        "user_input"
                    ]
                ),

                response=(
                    sample[
                        "response"
                    ]
                ),

                reference=(
                    sample[
                        "reference"
                    ]
                ),
            )
        )

    raise RuntimeError(
        f"Unknown metric: "
        f"{metric}"
    )


# ============================================================
# 17. VERIFY FROZEN RUNTIME
# ============================================================

def verify_runtime():

    with (
        RUNTIME_MANIFEST_FILE
        .open(
            "r",
            encoding="utf-8",
        )
    ) as file:

        manifest = json.load(
            file
        )

    # ========================================================
    # PYTHON
    # ========================================================

    frozen_python = (
        manifest[
            "python"
        ][
            "version"
        ]
    )

    current_python = (
        platform.python_version()
    )

    if (
        current_python
        !=
        frozen_python
    ):

        raise RuntimeError(
            "Python version berubah.\n"
            f"Frozen : {frozen_python}\n"
            f"Current: {current_python}"
        )

    # ========================================================
    # PACKAGES
    # ========================================================

    frozen_packages = (
        manifest[
            "runtime_packages"
        ]
    )

    for (
        package_name,
        frozen_version,
    ) in frozen_packages.items():

        current_version = version(
            package_name
        )

        if (
            current_version
            !=
            frozen_version
        ):

            raise RuntimeError(
                f"Package berubah: "
                f"{package_name}\n"
                f"Frozen : "
                f"{frozen_version}\n"
                f"Current: "
                f"{current_version}"
            )


# ============================================================
# 18. CHECKPOINT VALIDATION
# ============================================================

def validate_checkpoint(
    checkpoint,
    source_hashes,
):

    if (
        checkpoint.get(
            "source_hashes"
        )
        !=
        source_hashes
    ):

        raise RuntimeError(
            "Checkpoint berasal dari "
            "source artifact berbeda."
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

    completed = (
        checkpoint.get(
            "completed",
            {},
        )
    )

    if not isinstance(
        completed,
        dict,
    ):

        raise RuntimeError(
            "Checkpoint completed "
            "bukan dictionary."
        )

    if (
        len(completed)
        >
        EXPECTED_TASK_COUNT
    ):

        raise RuntimeError(
            "Completed task "
            "lebih dari 360."
        )


# ============================================================
# 19. CREATE CHECKPOINT
# ============================================================

def new_checkpoint(
    source_hashes,
):

    return {
        "artifact":
            (
                "Aire Optima RAG "
                "Evaluation Final v1 "
                "Checkpoint"
            ),

        "status":
            "IN_PROGRESS",

        "started_at_utc":
            utc_now(),

        "updated_at_utc":
            utc_now(),

        "source_hashes":
            source_hashes,

        "total_tasks":
            EXPECTED_TASK_COUNT,

        "completed":
            {},

        "attempt_errors":
            [],

        "disabled_keys":
            [],
    }


# ============================================================
# 20. LOAD CHECKPOINT
# ============================================================

def load_or_create_checkpoint(
    source_hashes,
):

    if not (
        CHECKPOINT_FILE.exists()
    ):

        checkpoint = (
            new_checkpoint(
                source_hashes
            )
        )

        atomic_write_json(
            CHECKPOINT_FILE,
            checkpoint,
        )

        return checkpoint

    with CHECKPOINT_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:

        checkpoint = json.load(
            file
        )

    validate_checkpoint(
        checkpoint,
        source_hashes,
    )

    return checkpoint


# ============================================================
# 21. PREFLIGHT CHECKPOINT READ
# ============================================================

def read_checkpoint_if_exists(
    source_hashes,
):

    if not (
        CHECKPOINT_FILE.exists()
    ):

        return None

    with CHECKPOINT_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:

        checkpoint = json.load(
            file
        )

    validate_checkpoint(
        checkpoint,
        source_hashes,
    )

    return checkpoint


# ============================================================
# 22. WRITE FINAL OUTPUTS
# ============================================================

def write_final_outputs(
    tasks,
    checkpoint,
    source_hashes,
):

    completed = (
        checkpoint[
            "completed"
        ]
    )

    if (
        len(completed)
        !=
        EXPECTED_TASK_COUNT
    ):

        raise RuntimeError(
            "Final outputs hanya boleh "
            "dibuat setelah 360 task selesai."
        )

    # ========================================================
    # ORDER EXACTLY LIKE FROZEN TASK PLAN
    # ========================================================

    ordered_results = []

    for task in tasks:

        task_id = (
            task[
                "task_id"
            ]
        )

        if (
            task_id
            not in completed
        ):

            raise RuntimeError(
                f"Missing result: "
                f"{task_id}"
            )

        ordered_results.append(
            completed[
                task_id
            ]
        )

    # ========================================================
    # JSONL
    # ========================================================

    with FINAL_JSONL.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as file:

        for record in (
            ordered_results
        ):

            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    default=str,
                )
                +
                "\n"
            )

    # ========================================================
    # CSV
    # ========================================================

    csv_fields = [
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

    with FINAL_CSV.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=csv_fields,
        )

        writer.writeheader()

        for record in (
            ordered_results
        ):

            writer.writerow(
                {
                    field:
                        record.get(
                            field
                        )

                    for field
                    in csv_fields
                }
            )

    # ========================================================
    # AGGREGATES
    # ========================================================

    grouped = defaultdict(
        list
    )

    for record in (
        ordered_results
    ):

        grouped[
            (
                record[
                    "method"
                ],
                record[
                    "metric"
                ],
            )
        ].append(
            float(
                record[
                    "value"
                ]
            )
        )

    aggregates = {}

    for method in METHODS:

        aggregates[
            method
        ] = {}

        for metric in METRICS:

            values = (
                grouped[
                    (
                        method,
                        metric,
                    )
                ]
            )

            if (
                len(values)
                !=
                40
            ):

                raise RuntimeError(
                    f"{method} × "
                    f"{metric} "
                    f"bukan 40."
                )

            aggregates[
                method
            ][
                metric
            ] = {
                "count":
                    len(
                        values
                    ),

                "mean":
                    (
                        sum(
                            values
                        )
                        /
                        len(
                            values
                        )
                    ),

                "min":
                    min(
                        values
                    ),

                "max":
                    max(
                        values
                    ),
            }

    # ========================================================
    # KEY USAGE
    # ========================================================

    key_success = Counter(
        record.get(
            "key_label",
            "UNKNOWN",
        )
        for record
        in ordered_results
    )

    key_errors = Counter(
        error.get(
            "key_label",
            "UNKNOWN",
        )
        for error
        in checkpoint.get(
            "attempt_errors",
            [],
        )
    )

    # ========================================================
    # HASHES
    # ========================================================

    jsonl_hash = (
        sha256_file(
            FINAL_JSONL
        )
    )

    csv_hash = (
        sha256_file(
            FINAL_CSV
        )
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    summary = {
        "artifact":
            (
                "Aire Optima RAG "
                "Evaluation Scores v1"
            ),

        "status":
            "COMPLETE_NOT_YET_FROZEN",

        "task_count":
            len(
                ordered_results
            ),

        "source_hashes":
            source_hashes,

        "aggregates":
            aggregates,

        "technical_error_attempts":
            len(
                checkpoint.get(
                    "attempt_errors",
                    [],
                )
            ),

        "key_usage_success":
            dict(
                key_success
            ),

        "key_usage_error":
            dict(
                key_errors
            ),

        "outputs": {
            "jsonl": {
                "file":
                    str(
                        FINAL_JSONL
                    ),

                "sha256":
                    jsonl_hash,
            },

            "csv": {
                "file":
                    str(
                        FINAL_CSV
                    ),

                "sha256":
                    csv_hash,
            },
        },

        "completed_at_utc":
            utc_now(),
    }

    atomic_write_json(
        SUMMARY_FILE,
        summary,
    )

    return (
        ordered_results,
        summary,
    )


# ============================================================
# 23. ASYNC MAIN
# ============================================================

async def async_main(
    preflight=False,
    max_new_tasks=None,
    max_attempts_per_task=8,
):

    print(
        "=" * 70
    )

    print(
        "AIRE OPTIMA RAG "
        "FINAL EVALUATION V1 "
        "- SINGLE PAID PROJECT"
    )

    print(
        "=" * 70
    )

    load_dotenv()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
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
    ]

    for path in (
        required_files
    ):

        if not (
            path.exists()
        ):

            raise (
                FileNotFoundError(
                    f"File tidak ditemukan: "
                    f"{path}"
                )
            )

    # ========================================================
    # FROZEN HASHES
    # ========================================================

    source_hashes = {
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
    }

    print(
        "\nFROZEN HASH AUDIT"
    )

    for (
        name,
        actual_hash,
    ) in source_hashes.items():

        print(
            f"{name:<22}: "
            f"{actual_hash}"
        )

        if (
            actual_hash
            !=
            expected_hashes[
                name
            ]
        ):

            raise RuntimeError(
                f"{name} hash berubah."
            )

    print(
        "\n✅ Semua frozen hash valid."
    )

    # ========================================================
    # RUNTIME
    # ========================================================

    verify_runtime()

    print(
        "✅ Frozen runtime valid."
    )

    # ========================================================
    # CONFIG
    # ========================================================

    with CONFIG_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:

        config = json.load(
            file
        )

    judge_model = (
        config[
            "judge"
        ][
            "model"
        ]
    )

    embedding_model = (
        config[
            "embedding"
        ][
            "model"
        ]
    )

    strictness = (
        config[
            "metrics"
        ][
            "answer_relevancy"
        ][
            "strictness"
        ]
    )

    print(
        "\nEVALUATOR"
    )

    print(
        f"Judge model       : "
        f"{judge_model}"
    )

    print(
        f"Embedding model   : "
        f"{embedding_model}"
    )

    print(
        f"Relevancy strict. : "
        f"{strictness}"
    )

    # ========================================================
    # DATASET + TASK PLAN
    # ========================================================

    dataset = load_jsonl(
        DATASET_FILE
    )

    tasks = load_jsonl(
        TASK_PLAN_FILE
    )

    if (
        len(dataset)
        !=
        EXPECTED_SAMPLE_COUNT
    ):

        raise RuntimeError(
            "Evaluation dataset "
            "bukan 120 samples."
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

    dataset_by_pair = {}

    for row in dataset:

        pair = (
            row[
                "query_id"
            ],
            row[
                "method"
            ],
        )

        if pair in (
            dataset_by_pair
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
        EXPECTED_SAMPLE_COUNT
    ):

        raise RuntimeError(
            "Unique dataset pair "
            "bukan 120."
        )

    # ========================================================
    # TASK VALIDATION
    # ========================================================

    seen_task_ids = set()

    for task in tasks:

        task_id = (
            task[
                "task_id"
            ]
        )

        if (
            task_id
            in
            seen_task_ids
        ):

            raise RuntimeError(
                f"Duplicate task ID: "
                f"{task_id}"
            )

        seen_task_ids.add(
            task_id
        )

        pair = (
            task[
                "query_id"
            ],
            task[
                "method"
            ],
        )

        if (
            pair
            not in
            dataset_by_pair
        ):

            raise RuntimeError(
                f"Dataset pair "
                f"tidak ada: "
                f"{pair}"
            )

        actual_sample_hash = (
            sample_sha256(
                dataset_by_pair[
                    pair
                ]
            )
        )

        if (
            actual_sample_hash
            !=
            task[
                "sample_sha256"
            ]
        ):

            raise RuntimeError(
                f"{task_id}: "
                "sample hash mismatch."
            )

    print(
        "✅ 360 frozen tasks valid."
    )

    # ========================================================
    # SINGLE PAID KEY
    # ========================================================

    paid_key = (
        load_paid_api_key()
    )

    print(
        "\nEXECUTION TRANSPORT"
    )

    print(
        "Mode              : "
        "SINGLE PAID PROJECT"
    )

    print(
        f"API key env       : "
        f"{PAID_KEY_LABEL}"
    )

    print(
        "Old key pool      : "
        "IGNORED"
    )

    print(
        f"Max attempts/task : "
        f"{max_attempts_per_task}"
    )

    if (
        max_new_tasks
        is None
    ):

        print(
            "Max new tasks     : "
            "NO LIMIT"
        )

    else:

        print(
            f"Max new tasks     : "
            f"{max_new_tasks}"
        )

    # ========================================================
    # PREFLIGHT
    # ========================================================

    if preflight:

        existing_checkpoint = (
            read_checkpoint_if_exists(
                source_hashes
            )
        )

        if (
            existing_checkpoint
            is None
        ):

            completed_count = 0

            print(
                "\nCheckpoint : "
                "belum ada"
            )

        else:

            completed_count = (
                len(
                    existing_checkpoint[
                        "completed"
                    ]
                )
            )

            print(
                "\nCheckpoint : "
                "ditemukan"
            )

            print(
                f"Status     : "
                f"{existing_checkpoint.get('status')}"
            )

        print(
            f"Completed  : "
            f"{completed_count}/360"
        )

        print(
            f"Pending    : "
            f"{360 - completed_count}"
        )

        print(
            "\n"
            + "=" * 70
        )

        print(
            "PREFLIGHT ONLY"
        )

        print(
            "=" * 70
        )

        print(
            "Tidak ada metric scoring "
            "yang dijalankan."
        )

        print(
            "\nSTATUS: PASS"
        )

        return

    # ========================================================
    # CHECKPOINT
    # ========================================================

    checkpoint = (
        load_or_create_checkpoint(
            source_hashes
        )
    )

    completed = (
        checkpoint[
            "completed"
        ]
    )

    initial_completed_count = (
        len(
            completed
        )
    )

    print(
        "\nCHECKPOINT"
    )

    print(
        f"Previous status : "
        f"{checkpoint.get('status')}"
    )

    print(
        f"Completed       : "
        f"{initial_completed_count}/360"
    )

    print(
        f"Pending         : "
        f"{360 - initial_completed_count}"
    )

    # ========================================================
    # CREATE ONE PAID RESOURCE
    # ========================================================

    resource = None

    newly_completed = 0

    try:

        resource = (
            PaidKeyResource(
                key_record=(
                    paid_key
                ),
                judge_model=(
                    judge_model
                ),
                embedding_model=(
                    embedding_model
                ),
                strictness=(
                    strictness
                ),
            )
        )

        # ====================================================
        # MAIN LOOP
        # ====================================================

        for task in tasks:

            task_id = (
                task[
                    "task_id"
                ]
            )

            # =================================================
            # SUCCESSFUL TASKS ARE NEVER RE-SCORED
            # =================================================

            if (
                task_id
                in
                completed
            ):

                continue

            # =================================================
            # PILOT LIMIT
            # =================================================

            if (
                max_new_tasks
                is not None
                and
                newly_completed
                >=
                max_new_tasks
            ):

                checkpoint[
                    "status"
                ] = (
                    "PAUSED_BY_MAX_NEW_TASKS"
                )

                checkpoint[
                    "updated_at_utc"
                ] = utc_now()

                atomic_write_json(
                    CHECKPOINT_FILE,
                    checkpoint,
                )

                print(
                    "\n"
                    + "=" * 70
                )

                print(
                    "RUN PAUSED BY "
                    "--max-new-tasks"
                )

                print(
                    "=" * 70
                )

                print(
                    f"New successful tasks : "
                    f"{newly_completed}"
                )

                print(
                    f"Completed total      : "
                    f"{len(completed)}/360"
                )

                print(
                    f"Pending              : "
                    f"{360 - len(completed)}"
                )

                print(
                    "\nCheckpoint aman."
                )

                return

            sample = (
                dataset_by_pair[
                    (
                        task[
                            "query_id"
                        ],
                        task[
                            "method"
                        ],
                    )
                ]
            )

            print(
                "\n"
                + "-" * 70
            )

            print(
                f"{task_id} / T0360"
            )

            print(
                f"Query  : "
                f"{task['query_id']}"
            )

            print(
                f"Method : "
                f"{task['method']}"
            )

            print(
                f"Metric : "
                f"{task['metric']}"
            )

            # =================================================
            # RETRY SAME PAID PROJECT
            # =================================================

            task_success = False

            for attempt_number in range(
                1,
                max_attempts_per_task + 1,
            ):

                print(
                    f"Attempt "
                    f"{attempt_number}/"
                    f"{max_attempts_per_task} "
                    f"→ "
                    f"{PAID_KEY_LABEL}"
                )

                start_time = (
                    time.perf_counter()
                )

                try:

                    metric_result = (
                        await execute_metric(
                            resource=resource,
                            metric=(
                                task[
                                    "metric"
                                ]
                            ),
                            sample=sample,
                        )
                    )

                    latency = (
                        time.perf_counter()
                        -
                        start_time
                    )

                    value = (
                        validate_score(
                            metric=(
                                task[
                                    "metric"
                                ]
                            ),
                            value=(
                                metric_result.value
                            ),
                        )
                    )

                    # =========================================
                    # VALID SCORE:
                    #
                    # ACCEPT IMMEDIATELY.
                    #
                    # Tidak ada quality retry.
                    # =========================================

                    reason_value = getattr(
                        metric_result,
                        "reason",
                        None,
                    )

                    result_record = {
                        "task_id":
                            task_id,

                        "sequence":
                            task[
                                "sequence"
                            ],

                        "query_id":
                            task[
                                "query_id"
                            ],

                        "method":
                            task[
                                "method"
                            ],

                        "metric":
                            task[
                                "metric"
                            ],

                        "sample_sha256":
                            task[
                                "sample_sha256"
                            ],

                        "value":
                            value,

                        "reason":
                            (
                                None
                                if (
                                    reason_value
                                    is None
                                )
                                else
                                str(
                                    reason_value
                                )
                            ),

                        "raw_result":
                            normalize_metric_result(
                                metric_result
                            ),

                        "key_label":
                            PAID_KEY_LABEL,

                        "attempt_count":
                            attempt_number,

                        "latency_seconds":
                            round(
                                latency,
                                4,
                            ),

                        "completed_at_utc":
                            utc_now(),
                    }

                    # =========================================
                    # CHECKPOINT AFTER EVERY VALID SCORE
                    # =========================================

                    completed[
                        task_id
                    ] = (
                        result_record
                    )

                    newly_completed += 1

                    checkpoint[
                        "status"
                    ] = "IN_PROGRESS"

                    checkpoint[
                        "updated_at_utc"
                    ] = utc_now()

                    checkpoint[
                        "execution_transport"
                    ] = {
                        "mode":
                            "single_paid_project",

                        "key_label":
                            PAID_KEY_LABEL,
                    }

                    atomic_write_json(
                        CHECKPOINT_FILE,
                        checkpoint,
                    )

                    print(
                        f"✅ Score      : "
                        f"{value:.6f}"
                    )

                    print(
                        f"Latency      : "
                        f"{latency:.2f}s"
                    )

                    print(
                        f"New this run : "
                        f"{newly_completed}"
                    )

                    print(
                        f"Progress     : "
                        f"{len(completed)}/360"
                    )

                    task_success = True

                    break

                except Exception as exc:

                    latency = (
                        time.perf_counter()
                        -
                        start_time
                    )

                    (
                        error_type,
                        wait_seconds,
                    ) = classify_error(
                        exc
                    )

                    error_record = {
                        "task_id":
                            task_id,

                        "sequence":
                            task[
                                "sequence"
                            ],

                        "query_id":
                            task[
                                "query_id"
                            ],

                        "method":
                            task[
                                "method"
                            ],

                        "metric":
                            task[
                                "metric"
                            ],

                        "key_label":
                            PAID_KEY_LABEL,

                        "attempt_number":
                            attempt_number,

                        "error_type":
                            error_type,

                        "exception_class":
                            type(
                                exc
                            ).__name__,

                        "message":
                            str(
                                exc
                            ),

                        "latency_seconds":
                            round(
                                latency,
                                4,
                            ),

                        "timestamp_utc":
                            utc_now(),
                    }

                    checkpoint.setdefault(
                        "attempt_errors",
                        [],
                    ).append(
                        error_record
                    )

                    checkpoint[
                        "updated_at_utc"
                    ] = utc_now()

                    atomic_write_json(
                        CHECKPOINT_FILE,
                        checkpoint,
                    )

                    print(
                        f"⚠ {error_type}"
                    )

                    print(
                        f"{type(exc).__name__}: "
                        f"{exc}"
                    )

                    # =========================================
                    # API KEY / PROJECT INVALID
                    # =========================================

                    if (
                        error_type
                        ==
                        "KEY_FATAL"
                    ):

                        checkpoint[
                            "status"
                        ] = (
                            "ABORTED_KEY_FATAL"
                        )

                        checkpoint[
                            "updated_at_utc"
                        ] = utc_now()

                        atomic_write_json(
                            CHECKPOINT_FILE,
                            checkpoint,
                        )

                        raise RuntimeError(
                            "Tier-1 API key/project "
                            "menghasilkan authentication "
                            "atau permission failure. "
                            "Runner dihentikan."
                        ) from exc

                    # =========================================
                    # GLOBAL CODE / CONFIG ERROR
                    # =========================================

                    if (
                        error_type
                        ==
                        "GLOBAL_FATAL"
                    ):

                        checkpoint[
                            "status"
                        ] = (
                            "ABORTED_GLOBAL_FATAL"
                        )

                        checkpoint[
                            "updated_at_utc"
                        ] = utc_now()

                        atomic_write_json(
                            CHECKPOINT_FILE,
                            checkpoint,
                        )

                        raise

                    # =========================================
                    # TECHNICAL RETRY
                    #
                    # SAME paid project.
                    # =========================================

                    if (
                        attempt_number
                        >=
                        max_attempts_per_task
                    ):

                        break

                    print(
                        f"→ Technical retry "
                        f"after "
                        f"{wait_seconds:.0f}s"
                    )

                    await asyncio.sleep(
                        wait_seconds
                    )

            # =================================================
            # MAX TECHNICAL ATTEMPTS EXHAUSTED
            # =================================================

            if not (
                task_success
            ):

                checkpoint[
                    "status"
                ] = (
                    "PAUSED_TECHNICAL_RETRY_EXHAUSTED"
                )

                checkpoint[
                    "updated_at_utc"
                ] = utc_now()

                atomic_write_json(
                    CHECKPOINT_FILE,
                    checkpoint,
                )

                raise RuntimeError(
                    f"{task_id} gagal "
                    f"menghasilkan valid score "
                    f"setelah "
                    f"{max_attempts_per_task} "
                    f"technical attempts.\n"
                    "Checkpoint aman. "
                    "Tidak ada score invalid "
                    "yang disimpan."
                )

        # ====================================================
        # ALL 360 COMPLETE
        # ====================================================

        if (
            len(completed)
            !=
            EXPECTED_TASK_COUNT
        ):

            raise RuntimeError(
                "Loop selesai tetapi "
                f"completed = "
                f"{len(completed)}, "
                "bukan 360."
            )

        checkpoint[
            "status"
        ] = "COMPLETE"

        checkpoint[
            "completed_at_utc"
        ] = utc_now()

        checkpoint[
            "updated_at_utc"
        ] = utc_now()

        checkpoint[
            "execution_transport"
        ] = {
            "mode":
                "single_paid_project",

            "key_label":
                PAID_KEY_LABEL,
        }

        atomic_write_json(
            CHECKPOINT_FILE,
            checkpoint,
        )

        # ====================================================
        # FINAL FILES
        # ====================================================

        (
            ordered_results,
            summary,
        ) = write_final_outputs(
            tasks=tasks,
            checkpoint=checkpoint,
            source_hashes=source_hashes,
        )

        # ====================================================
        # FINAL SUMMARY
        # ====================================================

        print(
            "\n"
            + "=" * 70
        )

        print(
            "FINAL EVALUATION COMPLETE"
        )

        print(
            "=" * 70
        )

        print(
            f"Successful tasks : "
            f"{len(ordered_results)}"
        )

        print(
            f"Technical errors : "
            f"{len(checkpoint.get('attempt_errors', []))}"
        )

        print(
            "\nMEAN SCORES"
        )

        for method in METHODS:

            print(
                f"\n{method}"
            )

            for metric in METRICS:

                mean_value = (
                    summary[
                        "aggregates"
                    ][
                        method
                    ][
                        metric
                    ][
                        "mean"
                    ]
                )

                print(
                    f"  "
                    f"{metric:<20}: "
                    f"{mean_value:.6f}"
                )

        print(
            "\nRESULT JSONL SHA256:"
        )

        print(
            summary[
                "outputs"
            ][
                "jsonl"
            ][
                "sha256"
            ]
        )

        print(
            "\nRESULT CSV SHA256:"
        )

        print(
            summary[
                "outputs"
            ][
                "csv"
            ][
                "sha256"
            ]
        )

        print(
            "\nSTATUS:"
        )

        print(
            "COMPLETE - BELUM FROZEN"
        )

    finally:

        if (
            resource
            is not None
        ):

            try:

                await resource.close()

            except Exception:
                pass


# ============================================================
# 24. ENTRY POINT
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Run frozen Aire Optima "
            "RAG evaluation v1 using "
            "one paid Gemini Tier-1 project."
        )
    )

    parser.add_argument(
        "--preflight",
        action="store_true",
        help=(
            "Validate frozen artifacts, "
            "runtime, task plan, paid key, "
            "and checkpoint without scoring."
        ),
    )

    parser.add_argument(
        "--max-new-tasks",
        type=int,
        default=None,
        help=(
            "Stop automatically after N "
            "new successful tasks. "
            "Useful for paid-cost pilot."
        ),
    )

    parser.add_argument(
        "--max-attempts-per-task",
        type=int,
        default=8,
        help=(
            "Maximum technical attempts "
            "for one task using the same "
            "paid project. Default: 8."
        ),
    )

    args = parser.parse_args()

    if (
        args.max_new_tasks
        is not None
        and
        args.max_new_tasks
        <= 0
    ):

        parser.error(
            "--max-new-tasks "
            "harus > 0."
        )

    if (
        args.max_attempts_per_task
        <= 0
    ):

        parser.error(
            "--max-attempts-per-task "
            "harus > 0."
        )

    asyncio.run(
        async_main(
            preflight=(
                args.preflight
            ),
            max_new_tasks=(
                args.max_new_tasks
            ),
            max_attempts_per_task=(
                args.max_attempts_per_task
            ),
        )
    )


if __name__ == "__main__":
    main()