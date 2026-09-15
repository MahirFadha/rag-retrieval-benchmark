import hashlib
import json
import platform
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

DATASET_FILE = Path(
    "benchmark/data/evaluation/"
    "rag_evaluation_dataset_v1.jsonl"
)

SMOKE_FILE = Path(
    "benchmark/results/evaluation/"
    "rag_evaluator_smoke_v1.json"
)

RUNTIME_MANIFEST_FILE = Path(
    "benchmark/config/"
    "rag_evaluator_runtime_manifest_v1.json"
)


# ============================================================
# FROZEN HASHES
# ============================================================

EXPECTED_DATASET_SHA256 = (
    "fe1582a06378e9aa087551a2ee01b4c5"
    "76fb14a38cdb51cc7057439a4e1b27fd"
)

EXPECTED_CONFIG_SHA256 = (
    "60b4cd3d282f347834bd13b2c2dfc722"
    "e5a18ad8939bd3042c41b6c04bc35535"
)


# ============================================================
# EXPECTED CORE RUNTIME
# ============================================================

EXPECTED_RAGAS_VERSION = "0.4.3"

EXPECTED_GOOGLE_GENAI_VERSION = "2.21.0"


# ============================================================
# HELPERS
# ============================================================

def sha256_file(path: Path) -> str:

    hasher = hashlib.sha256()

    with path.open("rb") as file:

        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            hasher.update(chunk)

    return hasher.hexdigest()


def get_package_version(
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
        "RAG EVALUATOR RUNTIME V1"
    )

    print(
        "=" * 70
    )

    # ========================================================
    # FILE CHECK
    # ========================================================

    for path in [
        CONFIG_FILE,
        DATASET_FILE,
        SMOKE_FILE,
    ]:

        if not path.exists():

            raise FileNotFoundError(
                f"File tidak ditemukan: "
                f"{path}"
            )

    # ========================================================
    # FROZEN HASH AUDIT
    # ========================================================

    config_hash = (
        sha256_file(
            CONFIG_FILE
        )
    )

    dataset_hash = (
        sha256_file(
            DATASET_FILE
        )
    )

    smoke_hash = (
        sha256_file(
            SMOKE_FILE
        )
    )

    print(
        "\nARTIFACT HASH AUDIT"
    )

    print(
        f"Evaluator config : "
        f"{config_hash}"
    )

    print(
        f"Evaluation data  : "
        f"{dataset_hash}"
    )

    print(
        f"Smoke artifact   : "
        f"{smoke_hash}"
    )

    if (
        config_hash
        != EXPECTED_CONFIG_SHA256
    ):

        raise RuntimeError(
            "Evaluator config berubah."
        )

    if (
        dataset_hash
        != EXPECTED_DATASET_SHA256
    ):

        raise RuntimeError(
            "Evaluation dataset berubah."
        )

    print(
        "\n✅ Frozen config dan dataset valid."
    )

    # ========================================================
    # SMOKE AUDIT
    # ========================================================

    with SMOKE_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:

        smoke = json.load(
            file
        )

    if (
        smoke.get(
            "status"
        )
        != "SMOKE_TEST_ONLY"
    ):

        raise RuntimeError(
            "Smoke artifact status "
            "tidak sesuai."
        )

    sample = (
        smoke.get(
            "sample",
            {}
        )
    )

    if (
        sample.get(
            "query_id"
        )
        != "Q001"
        or
        sample.get(
            "method"
        )
        != "bm25"
    ):

        raise RuntimeError(
            "Smoke sample harus "
            "Q001 + bm25."
        )

    evaluator = (
        smoke.get(
            "evaluator",
            {}
        )
    )

    if (
        evaluator.get(
            "judge_model"
        )
        != "gemini-3.7-flash"
    ):

        raise RuntimeError(
            "Smoke judge model "
            "tidak sesuai."
        )

    if (
        evaluator.get(
            "embedding_model"
        )
        != "gemini-embedding-001"
    ):

        raise RuntimeError(
            "Smoke embedding model "
            "tidak sesuai."
        )

    # ========================================================
    # SMOKE SCORE AUDIT
    # ========================================================

    scores = (
        smoke.get(
            "scores",
            []
        )
    )

    if len(scores) != 3:

        raise RuntimeError(
            "Smoke artifact harus "
            "memiliki 3 metric scores."
        )

    score_by_metric = {
        record[
            "metric"
        ]:
            record[
                "value"
            ]

        for record
        in scores
    }

    required_metrics = {
        "faithfulness",
        "answer_relevancy",
        "answer_accuracy",
    }

    if (
        set(
            score_by_metric
        )
        != required_metrics
    ):

        raise RuntimeError(
            "Metric smoke tidak lengkap."
        )

    metric_ranges = {
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

    for metric, value in (
        score_by_metric.items()
    ):

        try:

            value = float(
                value
            )

        except (
            TypeError,
            ValueError,
        ) as exc:

            raise RuntimeError(
                f"{metric}: "
                "score non-numeric."
            ) from exc

        minimum, maximum = (
            metric_ranges[
                metric
            ]
        )

        if not (
            minimum
            <= value
            <= maximum
        ):

            raise RuntimeError(
                f"{metric}: "
                f"score di luar range."
            )

    print(
        "\n✅ Smoke test artifact valid."
    )

    # ========================================================
    # RUNTIME VERSIONS
    # ========================================================

    runtime_packages = {
        "ragas":
            get_package_version(
                "ragas"
            ),

        "google-genai":
            get_package_version(
                "google-genai"
            ),

        "instructor":
            get_package_version(
                "instructor"
            ),

        "jsonref":
            get_package_version(
                "jsonref"
            ),

        "pydantic":
            get_package_version(
                "pydantic"
            ),

        "python-dotenv":
            get_package_version(
                "python-dotenv"
            ),
    }

    # ========================================================
    # REQUIRED PACKAGES
    # ========================================================

    for name, package_version in (
        runtime_packages.items()
    ):

        if package_version is None:

            raise RuntimeError(
                f"Package runtime "
                f"tidak ditemukan: "
                f"{name}"
            )

    if (
        runtime_packages[
            "ragas"
        ]
        != EXPECTED_RAGAS_VERSION
    ):

        raise RuntimeError(
            "Ragas version berubah.\n"
            f"Expected: "
            f"{EXPECTED_RAGAS_VERSION}\n"
            f"Actual  : "
            f"{runtime_packages['ragas']}"
        )

    if (
        runtime_packages[
            "google-genai"
        ]
        != EXPECTED_GOOGLE_GENAI_VERSION
    ):

        raise RuntimeError(
            "google-genai version berubah.\n"
            f"Expected: "
            f"{EXPECTED_GOOGLE_GENAI_VERSION}\n"
            f"Actual  : "
            f"{runtime_packages['google-genai']}"
        )

    # ========================================================
    # PRINT RUNTIME
    # ========================================================

    print(
        "\nRUNTIME ENVIRONMENT"
    )

    print(
        f"Python         : "
        f"{platform.python_version()}"
    )

    for name, package_version in (
        runtime_packages.items()
    ):

        print(
            f"{name:<15}: "
            f"{package_version}"
        )

    # ========================================================
    # CREATE MANIFEST
    # ========================================================

    manifest = {
        "artifact":
            (
                "Aire Optima RAG "
                "Evaluator Runtime v1"
            ),

        "status":
            "FROZEN",

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

        "smoke_test": {
            "file":
                str(
                    SMOKE_FILE
                ),

            "sha256":
                smoke_hash,

            "sample":
                {
                    "query_id":
                        "Q001",

                    "method":
                        "bm25",
                },

            "scores":
                score_by_metric,

            "included_in_final_results":
                False,
        },

        "python": {
            "version":
                platform.python_version(),

            "implementation":
                platform.python_implementation(),
        },

        "runtime_packages":
            runtime_packages,

        "execution_architecture": {
            "ragas_api":
                (
                    "metrics.collections"
                ),

            "metric_execution":
                "ascore",

            "judge_transport":
                (
                    "Instructor "
                    "AsyncInstructor"
                ),

            "judge_model":
                "gemini-3.7-flash",

            "embedding_model":
                "gemini-embedding-001",

            "embedding_transport":
                (
                    "GoogleEmbeddings with "
                    "google.genai.Client"
                ),
        },

        "freeze_rule":
            (
                "Do not upgrade or downgrade "
                "evaluator runtime packages "
                "after final evaluation scoring "
                "has begun."
            ),
    }

    # ========================================================
    # IF ALREADY FROZEN, VERIFY
    # ========================================================

    if (
        RUNTIME_MANIFEST_FILE.exists()
    ):

        with RUNTIME_MANIFEST_FILE.open(
            "r",
            encoding="utf-8",
        ) as file:

            existing = json.load(
                file
            )

        if (
            existing
            != manifest
        ):

            raise RuntimeError(
                "Runtime manifest sudah ada "
                "tetapi runtime saat ini "
                "berbeda dari versi frozen."
            )

        print(
            "\nRuntime manifest "
            "sudah ada dan tetap valid."
        )

    else:

        with RUNTIME_MANIFEST_FILE.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                manifest,
                file,
                ensure_ascii=False,
                indent=2,
            )

    # ========================================================
    # MANIFEST HASH
    # ========================================================

    manifest_hash = (
        sha256_file(
            RUNTIME_MANIFEST_FILE
        )
    )

    # ========================================================
    # FINAL
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "EVALUATOR RUNTIME FREEZE"
    )

    print(
        "=" * 70
    )

    print(
        f"Ragas          : "
        f"{runtime_packages['ragas']}"
    )

    print(
        f"google-genai   : "
        f"{runtime_packages['google-genai']}"
    )

    print(
        f"instructor     : "
        f"{runtime_packages['instructor']}"
    )

    print(
        f"jsonref        : "
        f"{runtime_packages['jsonref']}"
    )

    print(
        f"pydantic       : "
        f"{runtime_packages['pydantic']}"
    )

    print(
        "\nRUNTIME MANIFEST SHA256:"
    )

    print(
        manifest_hash
    )

    print(
        "\nManifest:"
    )

    print(
        RUNTIME_MANIFEST_FILE
    )

    print(
        "\nSTATUS: FROZEN"
    )


if __name__ == "__main__":
    main()