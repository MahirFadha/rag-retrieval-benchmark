import hashlib
import json
from importlib.metadata import (
    PackageNotFoundError,
    version,
)
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

CONFIG_FILE = Path(
    "benchmark/config/"
    "rag_evaluator_config_v1.json"
)

EVALUATION_DATASET_FILE = Path(
    "benchmark/data/evaluation/"
    "rag_evaluation_dataset_v1.jsonl"
)

MANIFEST_FILE = Path(
    "benchmark/config/"
    "rag_evaluator_manifest_v1.json"
)


# ============================================================
# EXPECTED VALUES
# ============================================================

EXPECTED_DATASET_SHA256 = (
    "fe1582a06378e9aa087551a2ee01b4c5"
    "76fb14a38cdb51cc7057439a4e1b27fd"
)

EXPECTED_RAGAS_VERSION = "0.4.3"

EXPECTED_JUDGE_MODEL = (
    "gemini-3.7-flash"
)

EXPECTED_EMBEDDING_MODEL = (
    "gemini-embedding-001"
)


# ============================================================
# HELPERS
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


def package_version(
    package_name: str,
):

    try:

        return version(
            package_name
        )

    except PackageNotFoundError:

        return None


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 70
    )

    print(
        "VERIFY & FREEZE "
        "RAG EVALUATOR CONFIG V1"
    )

    print(
        "=" * 70
    )

    # ========================================================
    # FILE EXISTENCE
    # ========================================================

    for path in [
        CONFIG_FILE,
        EVALUATION_DATASET_FILE,
    ]:

        if not path.exists():

            raise FileNotFoundError(
                f"File tidak ditemukan: "
                f"{path}"
            )

    # ========================================================
    # HASH DATASET
    # ========================================================

    dataset_hash = (
        sha256_file(
            EVALUATION_DATASET_FILE
        )
    )

    if (
        dataset_hash
        != EXPECTED_DATASET_SHA256
    ):

        raise RuntimeError(
            "Evaluation dataset SHA256 "
            "berubah."
        )

    # ========================================================
    # LOAD CONFIG
    # ========================================================

    with CONFIG_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:

        config = json.load(
            file
        )

    # ========================================================
    # BASIC CONFIG
    # ========================================================

    if (
        config.get(
            "framework"
        )
        != "ragas"
    ):

        raise RuntimeError(
            "Framework harus ragas."
        )

    if (
        config.get(
            "framework_version"
        )
        != EXPECTED_RAGAS_VERSION
    ):

        raise RuntimeError(
            "framework_version harus "
            f"{EXPECTED_RAGAS_VERSION}."
        )

    if (
        config.get(
            "evaluation_type"
        )
        != "single_turn_rag"
    ):

        raise RuntimeError(
            "evaluation_type harus "
            "single_turn_rag."
        )

    # ========================================================
    # DATASET CONFIG
    # ========================================================

    dataset_cfg = (
        config.get(
            "evaluation_dataset",
            {},
        )
    )

    if (
        dataset_cfg.get(
            "sha256"
        )
        != EXPECTED_DATASET_SHA256
    ):

        raise RuntimeError(
            "Dataset SHA256 pada config "
            "tidak sesuai."
        )

    if (
        dataset_cfg.get(
            "records"
        )
        != 120
    ):

        raise RuntimeError(
            "Dataset records harus 120."
        )

    if (
        dataset_cfg.get(
            "queries"
        )
        != 40
    ):

        raise RuntimeError(
            "Dataset queries harus 40."
        )

    expected_methods = {
        "bm25",
        "e5",
        "hybrid_rrf",
    }

    actual_methods = set(
        dataset_cfg.get(
            "methods",
            [],
        )
    )

    if (
        actual_methods
        != expected_methods
    ):

        raise RuntimeError(
            "Methods tidak sesuai."
        )

    # ========================================================
    # JUDGE
    # ========================================================

    judge = (
        config.get(
            "judge",
            {}
        )
    )

    if (
        judge.get(
            "provider"
        )
        != "google"
    ):

        raise RuntimeError(
            "Judge provider harus google."
        )

    if (
        judge.get(
            "model"
        )
        != EXPECTED_JUDGE_MODEL
    ):

        raise RuntimeError(
            "Judge model harus "
            f"{EXPECTED_JUDGE_MODEL}."
        )

    if (
        judge.get(
            "model_alias_latest"
        )
        is not False
    ):

        raise RuntimeError(
            "model_alias_latest "
            "harus false."
        )

    if (
        judge.get(
            "thinking_control"
        )
        != "provider_default"
    ):

        raise RuntimeError(
            "thinking_control harus "
            "provider_default."
        )

    if (
        judge.get(
            "documented_default_thinking_level"
        )
        != "medium"
    ):

        raise RuntimeError(
            "Documented thinking level "
            "harus medium."
        )

    # ========================================================
    # EMBEDDING
    # ========================================================

    embedding = (
        config.get(
            "embedding",
            {}
        )
    )

    if (
        embedding.get(
            "provider"
        )
        != "google"
    ):

        raise RuntimeError(
            "Embedding provider harus "
            "google."
        )

    if (
        embedding.get(
            "model"
        )
        != EXPECTED_EMBEDDING_MODEL
    ):

        raise RuntimeError(
            "Embedding model harus "
            f"{EXPECTED_EMBEDDING_MODEL}."
        )

    if (
        embedding.get(
            "purpose"
        )
        != "answer_relevancy_only"
    ):

        raise RuntimeError(
            "Embedding purpose harus "
            "answer_relevancy_only."
        )

    # ========================================================
    # METRICS
    # ========================================================

    metrics = (
        config.get(
            "metrics",
            {}
        )
    )

    required_primary = [
        "faithfulness",
        "answer_relevancy",
        "answer_accuracy",
    ]

    for metric_name in required_primary:

        metric = (
            metrics.get(
                metric_name,
                {}
            )
        )

        if (
            metric.get(
                "enabled"
            )
            is not True
        ):

            raise RuntimeError(
                f"{metric_name} "
                "harus enabled."
            )

        if (
            metric.get(
                "primary"
            )
            is not True
        ):

            raise RuntimeError(
                f"{metric_name} "
                "harus primary."
            )

    if (
        metrics[
            "answer_relevancy"
        ].get(
            "strictness"
        )
        != 3
    ):

        raise RuntimeError(
            "Answer relevancy "
            "strictness harus 3."
        )

    # Context metrics harus OFF.
    for metric_name in [
        "context_precision",
        "context_recall",
    ]:

        if (
            metrics.get(
                metric_name,
                {}
            ).get(
                "enabled"
            )
            is not False
        ):

            raise RuntimeError(
                f"{metric_name} "
                "harus disabled."
            )

    # ========================================================
    # EVALUATION POLICY
    # ========================================================

    policy = (
        config.get(
            "evaluation_policy",
            {}
        )
    )

    required_false = [
        "generated_answers_regenerated",
        "references_modified",
        "retrieved_contexts_modified",
        "evaluation_dataset_modified",
        "quality_retry",
    ]

    for field in required_false:

        if (
            policy.get(
                field
            )
            is not False
        ):

            raise RuntimeError(
                f"{field} harus false."
            )

    if (
        policy.get(
            "technical_retry_only"
        )
        is not True
    ):

        raise RuntimeError(
            "technical_retry_only "
            "harus true."
        )

    # ========================================================
    # PACKAGE VERSIONS
    # ========================================================

    runtime_versions = {
        "ragas":
            package_version(
                "ragas"
            ),

        "google-genai":
            package_version(
                "google-genai"
            ),

        "litellm":
            package_version(
                "litellm"
            ),

        "pydantic":
            package_version(
                "pydantic"
            ),
    }

    if (
        runtime_versions[
            "ragas"
        ]
        != EXPECTED_RAGAS_VERSION
    ):

        raise RuntimeError(
            "Installed Ragas version "
            "tidak sesuai.\n"
            f"Expected: "
            f"{EXPECTED_RAGAS_VERSION}\n"
            f"Actual: "
            f"{runtime_versions['ragas']}"
        )

    if (
        runtime_versions[
            "google-genai"
        ]
        is None
    ):

        raise RuntimeError(
            "google-genai "
            "belum terinstall."
        )

    # ========================================================
    # CONFIG HASH
    # ========================================================

    config_hash = (
        sha256_file(
            CONFIG_FILE
        )
    )

    # ========================================================
    # PRINT
    # ========================================================

    print(
        "\nSHA256 AUDIT"
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
        "\nRUNTIME"
    )

    for name, package_ver in (
        runtime_versions.items()
    ):

        print(
            f"{name:<15}: "
            f"{package_ver}"
        )

    print(
        "\nEVALUATOR"
    )

    print(
        f"Judge model      : "
        f"{EXPECTED_JUDGE_MODEL}"
    )

    print(
        "Thinking policy  : "
        "provider default "
        "(documented medium)"
    )

    print(
        f"Embedding model  : "
        f"{EXPECTED_EMBEDDING_MODEL}"
    )

    print(
        "Metrics          : "
        "Faithfulness, "
        "Answer Relevancy, "
        "Answer Accuracy"
    )

    # ========================================================
    # MANIFEST
    # ========================================================

    manifest = {
        "artifact":
            "Aire Optima RAG "
            "Evaluator Config v1",

        "status":
            "FROZEN",

        "config": {
            "file":
                str(
                    CONFIG_FILE
                ),

            "sha256":
                config_hash,
        },

        "evaluation_dataset": {
            "file":
                str(
                    EVALUATION_DATASET_FILE
                ),

            "sha256":
                dataset_hash,
        },

        "framework": {
            "name":
                "ragas",

            "version":
                runtime_versions[
                    "ragas"
                ],
        },

        "runtime_packages":
            runtime_versions,

        "judge": {
            "provider":
                "google",

            "model":
                EXPECTED_JUDGE_MODEL,

            "thinking_control":
                "provider_default",

            "documented_default_thinking_level":
                "medium",
        },

        "embedding": {
            "provider":
                "google",

            "model":
                EXPECTED_EMBEDDING_MODEL,

            "purpose":
                "answer_relevancy_only",
        },

        "primary_metrics": [
            "faithfulness",
            "answer_relevancy",
            "answer_accuracy",
        ],

        "disabled_rag_context_metrics": [
            "context_precision",
            "context_recall",
        ],

        "freeze_rule":
            (
                "Do not change evaluator model, "
                "embedding model, metric definitions, "
                "metric parameters, evaluation dataset, "
                "or evaluation policy after scoring begins."
            ),
    }

    with MANIFEST_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            manifest,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(
        "\nEVALUATOR CONFIG SHA256:"
    )

    print(
        config_hash
    )

    print(
        "\nManifest:"
    )

    print(
        MANIFEST_FILE
    )

    print(
        "\nSTATUS: FROZEN"
    )


if __name__ == "__main__":
    main()