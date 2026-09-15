import hashlib
import json
from collections import Counter
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

RAG_INPUT_FILE = Path(
    "benchmark/data/rag/"
    "rag_generation_inputs_v1.jsonl"
)

ANSWER_FILE = Path(
    "benchmark/results/rag/"
    "rag_generated_answers_v2.jsonl"
)

GENERATOR_CONFIG_FILE = Path(
    "benchmark/config/"
    "rag_generator_config_v2.json"
)

SYSTEM_PROMPT_FILE = Path(
    "benchmark/config/"
    "rag_system_prompt_v1.txt"
)

MANIFEST_FILE = Path(
    "benchmark/results/rag/"
    "rag_generated_answers_manifest_v2.json"
)


# ============================================================
# EXPECTED VALUES
# ============================================================

EXPECTED_RAG_INPUT_SHA256 = (
    "742937ac9af494eb41fb0662dcdce02a2"
    "cf30f06c1eb02a57c98d37bc7067a93"
)

EXPECTED_ANSWERS_SHA256 = (
    "4fec3abd2320c48ecf0d654e830ec6b2"
    "effba83a84808fb92970b85dbbdb0af8"
)

EXPECTED_COUNT = 120

EXPECTED_METHOD_COUNTS = {
    "bm25": 40,
    "e5": 40,
    "hybrid_rrf": 40,
}

EXPECTED_MODEL = (
    "gemini-3.6-flash"
)


# ============================================================
# SHA256
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


# ============================================================
# LOAD JSONL
# ============================================================

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
                    "Invalid JSON pada "
                    f"line {line_number}."
                ) from exc

    return records


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 70
    )

    print(
        "VERIFY & FREEZE "
        "RAG GENERATED ANSWERS V2"
    )

    print(
        "=" * 70
    )

    # ========================================================
    # FILE EXISTENCE
    # ========================================================

    required_files = [
        RAG_INPUT_FILE,
        ANSWER_FILE,
        GENERATOR_CONFIG_FILE,
        SYSTEM_PROMPT_FILE,
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

    rag_input_hash = (
        sha256_file(
            RAG_INPUT_FILE
        )
    )

    answers_hash = (
        sha256_file(
            ANSWER_FILE
        )
    )

    config_hash = (
        sha256_file(
            GENERATOR_CONFIG_FILE
        )
    )

    system_prompt_hash = (
        sha256_file(
            SYSTEM_PROMPT_FILE
        )
    )

    print(
        "\nSHA256 AUDIT"
    )

    print(
        f"RAG input     : "
        f"{rag_input_hash}"
    )

    print(
        f"Generated ans.: "
        f"{answers_hash}"
    )

    print(
        f"Generator cfg : "
        f"{config_hash}"
    )

    print(
        f"System prompt : "
        f"{system_prompt_hash}"
    )

    if (
        rag_input_hash
        != EXPECTED_RAG_INPUT_SHA256
    ):

        raise RuntimeError(
            "RAG input SHA256 berubah."
        )

    if (
        answers_hash
        != EXPECTED_ANSWERS_SHA256
    ):

        raise RuntimeError(
            "Generated answers SHA256 "
            "berubah."
        )

    # ========================================================
    # LOAD ANSWERS
    # ========================================================

    records = (
        load_jsonl(
            ANSWER_FILE
        )
    )

    if (
        len(records)
        != EXPECTED_COUNT
    ):

        raise RuntimeError(
            "Jumlah generated answers "
            f"bukan {EXPECTED_COUNT}."
        )

    # ========================================================
    # AUDIT
    # ========================================================

    pairs = set()

    response_ids = set()

    method_counts = Counter()

    finish_counts = Counter()

    model_counts = Counter()

    empty_answers = []

    duplicate_response_ids = []

    prompt_tokens = 0
    output_tokens = 0
    thoughts_tokens = 0
    total_tokens = 0

    latencies = []

    for record in records:

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

        # ----------------------------------------------------
        # UNIQUE QUERY-METHOD
        # ----------------------------------------------------

        if pair in pairs:

            raise RuntimeError(
                "Duplicate query-method: "
                f"{pair}"
            )

        pairs.add(pair)

        method_counts[
            method
        ] += 1

        # ----------------------------------------------------
        # ANSWER
        # ----------------------------------------------------

        answer = (
            record.get(
                "answer"
            )
            or ""
        ).strip()

        if not answer:

            empty_answers.append(
                pair
            )

        # ----------------------------------------------------
        # FINISH REASON
        # ----------------------------------------------------

        finish_reason = (
            record.get(
                "finish_reason"
            )
        )

        finish_counts[
            finish_reason
        ] += 1

        # ----------------------------------------------------
        # MODEL VERSION
        # ----------------------------------------------------

        model_version = (
            record.get(
                "model_version"
            )
        )

        model_counts[
            model_version
        ] += 1

        # ----------------------------------------------------
        # RESPONSE ID
        # ----------------------------------------------------

        response_id = (
            record.get(
                "response_id"
            )
        )

        if not response_id:

            raise RuntimeError(
                f"{pair}: response_id kosong."
            )

        if (
            response_id
            in response_ids
        ):

            duplicate_response_ids.append(
                response_id
            )

        response_ids.add(
            response_id
        )

        # ----------------------------------------------------
        # TOKEN USAGE
        # ----------------------------------------------------

        prompt_tokens += (
            record.get(
                "prompt_tokens"
            )
            or 0
        )

        output_tokens += (
            record.get(
                "output_tokens"
            )
            or 0
        )

        thoughts_tokens += (
            record.get(
                "thoughts_tokens"
            )
            or 0
        )

        total_tokens += (
            record.get(
                "total_tokens"
            )
            or 0
        )

        # ----------------------------------------------------
        # LATENCY
        # ----------------------------------------------------

        latency = (
            record.get(
                "latency_seconds"
            )
        )

        if latency is not None:

            latencies.append(
                float(
                    latency
                )
            )

    # ========================================================
    # STRICT VALIDATION
    # ========================================================

    if (
        len(pairs)
        != EXPECTED_COUNT
    ):

        raise RuntimeError(
            "Unique query-method "
            "bukan 120."
        )

    if (
        dict(method_counts)
        != EXPECTED_METHOD_COUNTS
    ):

        raise RuntimeError(
            "Method counts tidak sesuai.\n"
            f"{dict(method_counts)}"
        )

    if empty_answers:

        raise RuntimeError(
            "Ditemukan jawaban kosong:\n"
            f"{empty_answers}"
        )

    if duplicate_response_ids:

        raise RuntimeError(
            "Duplicate response_id:\n"
            f"{duplicate_response_ids}"
        )

    if (
        finish_counts.get(
            "STOP",
            0,
        )
        != EXPECTED_COUNT
    ):

        raise RuntimeError(
            "Tidak semua generation "
            "berakhir STOP.\n"
            f"{dict(finish_counts)}"
        )

    if (
        len(finish_counts)
        != 1
    ):

        raise RuntimeError(
            "Ada finish reason selain STOP.\n"
            f"{dict(finish_counts)}"
        )

    if (
        model_counts.get(
            EXPECTED_MODEL,
            0,
        )
        != EXPECTED_COUNT
    ):

        raise RuntimeError(
            "Tidak semua response memakai "
            f"{EXPECTED_MODEL}.\n"
            f"{dict(model_counts)}"
        )

    # ========================================================
    # CONFIG VALIDATION
    # ========================================================

    with GENERATOR_CONFIG_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:

        config = (
            json.load(file)
        )

    expected_config = {
        "model":
            "gemini-3.6-flash",

        "thinking_level":
            "low",

        "max_output_tokens":
            2048,

        "candidate_count":
            1,

        "rag_top_k":
            5,

        "expected_inputs":
            120,
    }

    for key, expected_value in (
        expected_config.items()
    ):

        actual = (
            config.get(
                key
            )
        )

        if (
            actual
            != expected_value
        ):

            raise RuntimeError(
                f"Config {key} berubah.\n"
                f"Expected: "
                f"{expected_value}\n"
                f"Actual: {actual}"
            )

    for key in [
        "temperature",
        "top_p",
        "top_k",
    ]:

        if (
            config.get(
                key
            )
            is not None
        ):

            raise RuntimeError(
                f"{key} harus null."
            )

    # ========================================================
    # SUMMARY
    # ========================================================

    mean_latency = None

    if latencies:

        mean_latency = (
            sum(latencies)
            / len(latencies)
        )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "FINAL GENERATION V2 AUDIT"
    )

    print(
        "=" * 70
    )

    print(
        f"Answers          : "
        f"{len(records)}"
    )

    print(
        f"Unique pairs     : "
        f"{len(pairs)}"
    )

    print(
        f"BM25             : "
        f"{method_counts['bm25']}"
    )

    print(
        f"E5               : "
        f"{method_counts['e5']}"
    )

    print(
        f"Hybrid RRF       : "
        f"{method_counts['hybrid_rrf']}"
    )

    print(
        f"STOP             : "
        f"{finish_counts['STOP']}"
    )

    print(
        f"Empty answers    : "
        f"{len(empty_answers)}"
    )

    print(
        f"Duplicate resp ID: "
        f"{len(duplicate_response_ids)}"
    )

    print(
        f"Model version    : "
        f"{dict(model_counts)}"
    )

    print(
        f"\nPrompt tokens    : "
        f"{prompt_tokens}"
    )

    print(
        f"Output tokens    : "
        f"{output_tokens}"
    )

    print(
        f"Thoughts tokens  : "
        f"{thoughts_tokens}"
    )

    print(
        f"Total tokens     : "
        f"{total_tokens}"
    )

    if (
        mean_latency
        is not None
    ):

        print(
            f"Mean latency     : "
            f"{mean_latency:.2f}s"
        )

    # ========================================================
    # MANIFEST
    # ========================================================

    manifest = {
        "artifact":
            "Aire Optima RAG "
            "Generated Answers v2",

        "status":
            "FROZEN",

        "generated_answers": {
            "file":
                str(
                    ANSWER_FILE
                ),

            "sha256":
                answers_hash,

            "count":
                len(records),
        },

        "source_artifacts": {
            "rag_input": {
                "file":
                    str(
                        RAG_INPUT_FILE
                    ),

                "sha256":
                    rag_input_hash,
            },

            "system_prompt": {
                "file":
                    str(
                        SYSTEM_PROMPT_FILE
                    ),

                "sha256":
                    system_prompt_hash,
            },

            "generator_config": {
                "file":
                    str(
                        GENERATOR_CONFIG_FILE
                    ),

                "sha256":
                    config_hash,
            },
        },

        "generation_validation": {
            "unique_query_method_pairs":
                len(pairs),

            "method_counts":
                dict(
                    method_counts
                ),

            "finish_reason_counts":
                dict(
                    finish_counts
                ),

            "model_version_counts":
                dict(
                    model_counts
                ),

            "empty_answers":
                len(
                    empty_answers
                ),

            "duplicate_response_ids":
                len(
                    duplicate_response_ids
                ),
        },

        "token_usage": {
            "prompt_tokens":
                prompt_tokens,

            "output_tokens":
                output_tokens,

            "thoughts_tokens":
                thoughts_tokens,

            "total_tokens":
                total_tokens,
        },

        "mean_latency_seconds":
            mean_latency,

        "note": (
            "Generation v1 was excluded "
            "from final RAG evaluation "
            "because 10 responses ended "
            "with MAX_TOKENS. Generation "
            "v2 uses max_output_tokens=2048 "
            "and is the final generation "
            "artifact."
        ),

        "freeze_rule": (
            "Do not modify or regenerate "
            "individual answers after "
            "evaluation begins."
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
        "\nGENERATED ANSWERS SHA256:"
    )

    print(
        answers_hash
    )

    print(
        "\nSTATUS: FROZEN"
    )

    print(
        f"\nManifest:\n"
        f"{MANIFEST_FILE}"
    )


if __name__ == "__main__":
    main()