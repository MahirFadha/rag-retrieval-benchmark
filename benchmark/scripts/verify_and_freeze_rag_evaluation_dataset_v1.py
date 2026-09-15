import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


# ============================================================
# 1. PATHS
# ============================================================

RAG_INPUT_FILE = Path(
    "benchmark/data/rag/"
    "rag_generation_inputs_v1.jsonl"
)

GENERATED_ANSWERS_FILE = Path(
    "benchmark/results/rag/"
    "rag_generated_answers_v2.jsonl"
)

REFERENCE_FILE = Path(
    "benchmark/data/evaluation/"
    "rag_evaluation_reference_v1.jsonl"
)

EVALUATION_DATASET_FILE = Path(
    "benchmark/data/evaluation/"
    "rag_evaluation_dataset_v1.jsonl"
)

MANIFEST_FILE = Path(
    "benchmark/data/evaluation/"
    "rag_evaluation_dataset_manifest_v1.json"
)


# ============================================================
# 2. EXPECTED HASHES
# ============================================================

EXPECTED_RAG_INPUT_SHA256 = (
    "742937ac9af494eb41fb0662dcdce02a2"
    "cf30f06c1eb02a57c98d37bc7067a93"
)

EXPECTED_GENERATED_ANSWERS_SHA256 = (
    "4fec3abd2320c48ecf0d654e830ec6b2"
    "effba83a84808fb92970b85dbbdb0af8"
)

EXPECTED_REFERENCE_SHA256 = (
    "5a2fa623cba071ca2fa70e1ed1a3930e"
    "9fc084969a300f05d6148269673bb0ba"
)

EXPECTED_EVALUATION_DATASET_SHA256 = (
    "fe1582a06378e9aa087551a2ee01b4c5"
    "76fb14a38cdb51cc7057439a4e1b27fd"
)


# ============================================================
# 3. EXPECTED COUNTS
# ============================================================

EXPECTED_COUNT = 120

EXPECTED_QUERY_COUNT = 40

EXPECTED_METHODS = {
    "bm25",
    "e5",
    "hybrid_rrf",
}

EXPECTED_METHOD_COUNTS = {
    "bm25": 40,
    "e5": 40,
    "hybrid_rrf": 40,
}

EXPECTED_CATEGORY_COUNTS = {
    "exact": 30,
    "semantic": 30,
    "multi_constraint": 30,
    "fine_grained": 30,
}

EXPECTED_DOMAIN_COUNTS = {
    "product_ac": 36,
    "service_ac": 48,
    "cctv": 36,
}

EXPECTED_TOP5_COUNTS = {
    "bm25": 37,
    "e5": 38,
    "hybrid_rrf": 37,
}


# ============================================================
# 4. EXPECTED SCHEMA
# ============================================================

EXPECTED_KEYS = {
    "query_id",
    "method",
    "category",
    "domain",
    "question",
    "retrieved_ids",
    "relevant_id",
    "target_in_top5",
    "target_rank_top5",
    "reference_context",
    "user_input",
    "response",
    "retrieved_contexts",
    "reference",
}


# ============================================================
# 5. SHA256
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
# 6. LOAD JSONL
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

                record = json.loads(
                    line
                )

            except json.JSONDecodeError as exc:

                raise RuntimeError(
                    f"Invalid JSON pada "
                    f"{path}, "
                    f"line {line_number}."
                ) from exc

            records.append(
                record
            )

    return records


# ============================================================
# 7. MAIN
# ============================================================

def main():

    print(
        "=" * 70
    )

    print(
        "VERIFY & FREEZE "
        "RAG EVALUATION DATASET V1"
    )

    print(
        "=" * 70
    )

    # ========================================================
    # FILE EXISTENCE
    # ========================================================

    required_files = [
        RAG_INPUT_FILE,
        GENERATED_ANSWERS_FILE,
        REFERENCE_FILE,
        EVALUATION_DATASET_FILE,
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

    rag_input_hash = (
        sha256_file(
            RAG_INPUT_FILE
        )
    )

    answers_hash = (
        sha256_file(
            GENERATED_ANSWERS_FILE
        )
    )

    reference_hash = (
        sha256_file(
            REFERENCE_FILE
        )
    )

    evaluation_hash = (
        sha256_file(
            EVALUATION_DATASET_FILE
        )
    )

    print(
        "\nSHA256 AUDIT"
    )

    print(
        f"RAG input  : "
        f"{rag_input_hash}"
    )

    print(
        f"Answers    : "
        f"{answers_hash}"
    )

    print(
        f"Reference  : "
        f"{reference_hash}"
    )

    print(
        f"Evaluation : "
        f"{evaluation_hash}"
    )

    if (
        rag_input_hash
        != EXPECTED_RAG_INPUT_SHA256
    ):

        raise RuntimeError(
            "RAG input hash berubah."
        )

    if (
        answers_hash
        != EXPECTED_GENERATED_ANSWERS_SHA256
    ):

        raise RuntimeError(
            "Generated answers hash berubah."
        )

    if (
        reference_hash
        != EXPECTED_REFERENCE_SHA256
    ):

        raise RuntimeError(
            "Reference hash berubah."
        )

    if (
        evaluation_hash
        != EXPECTED_EVALUATION_DATASET_SHA256
    ):

        raise RuntimeError(
            "Evaluation dataset hash berubah."
        )

    print(
        "\n✅ Semua source + evaluation "
        "artifact hash valid."
    )

    # ========================================================
    # LOAD DATASET
    # ========================================================

    records = (
        load_jsonl(
            EVALUATION_DATASET_FILE
        )
    )

    if (
        len(records)
        != EXPECTED_COUNT
    ):

        raise RuntimeError(
            f"Evaluation dataset bukan "
            f"{EXPECTED_COUNT} records."
        )

    # ========================================================
    # AUDIT CONTAINERS
    # ========================================================

    seen_pairs = set()

    query_methods = defaultdict(
        set
    )

    query_shared_values = defaultdict(
        dict
    )

    method_counts = Counter()

    category_counts = Counter()

    domain_counts = Counter()

    top5_counts = Counter()

    empty_responses = []

    empty_references = []

    empty_contexts = []

    schema_errors = []

    target_consistency_errors = []

    question_alias_errors = []

    per_query_consistency_errors = []


    # ========================================================
    # RECORD-BY-RECORD AUDIT
    # ========================================================

    for index, record in enumerate(
        records,
        start=1,
    ):

        # ----------------------------------------------------
        # EXACT SCHEMA
        # ----------------------------------------------------

        actual_keys = set(
            record.keys()
        )

        if (
            actual_keys
            != EXPECTED_KEYS
        ):

            schema_errors.append(
                {
                    "index":
                        index,

                    "missing":
                        sorted(
                            EXPECTED_KEYS
                            - actual_keys
                        ),

                    "extra":
                        sorted(
                            actual_keys
                            - EXPECTED_KEYS
                        ),
                }
            )

            continue

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
        # UNIQUE PAIR
        # ----------------------------------------------------

        if pair in seen_pairs:

            raise RuntimeError(
                "Duplicate query-method pair: "
                f"{pair}"
            )

        seen_pairs.add(
            pair
        )

        query_methods[
            query_id
        ].add(
            method
        )

        # ----------------------------------------------------
        # METHOD
        # ----------------------------------------------------

        if (
            method
            not in EXPECTED_METHODS
        ):

            raise RuntimeError(
                f"{pair}: "
                f"unknown method {method}"
            )

        method_counts[
            method
        ] += 1

        # ----------------------------------------------------
        # CATEGORY / DOMAIN
        # ----------------------------------------------------

        category_counts[
            record[
                "category"
            ]
        ] += 1

        domain_counts[
            record[
                "domain"
            ]
        ] += 1

        # ----------------------------------------------------
        # QUESTION ALIAS
        # ----------------------------------------------------

        if (
            record[
                "question"
            ]
            != record[
                "user_input"
            ]
        ):

            question_alias_errors.append(
                pair
            )

        # ----------------------------------------------------
        # RESPONSE
        # ----------------------------------------------------

        response = (
            record.get(
                "response"
            )
            or ""
        ).strip()

        if not response:

            empty_responses.append(
                pair
            )

        # ----------------------------------------------------
        # REFERENCE
        # ----------------------------------------------------

        reference = (
            record.get(
                "reference"
            )
            or ""
        ).strip()

        if not reference:

            empty_references.append(
                pair
            )

        reference_context = (
            record.get(
                "reference_context"
            )
            or ""
        ).strip()

        if not reference_context:

            raise RuntimeError(
                f"{pair}: "
                "reference_context kosong."
            )

        # ----------------------------------------------------
        # RETRIEVED IDS / CONTEXTS
        # ----------------------------------------------------

        retrieved_ids = (
            record[
                "retrieved_ids"
            ]
        )

        retrieved_contexts = (
            record[
                "retrieved_contexts"
            ]
        )

        if (
            not isinstance(
                retrieved_ids,
                list,
            )
            or len(
                retrieved_ids
            ) != 5
        ):

            raise RuntimeError(
                f"{pair}: "
                "retrieved_ids bukan Top-5."
            )

        if (
            not isinstance(
                retrieved_contexts,
                list,
            )
            or len(
                retrieved_contexts
            ) != 5
        ):

            raise RuntimeError(
                f"{pair}: "
                "retrieved_contexts "
                "bukan Top-5."
            )

        for context_index, context in enumerate(
            retrieved_contexts,
            start=1,
        ):

            if (
                not isinstance(
                    context,
                    str,
                )
                or not context.strip()
            ):

                empty_contexts.append(
                    {
                        "pair":
                            pair,

                        "rank":
                            context_index,
                    }
                )

        # ----------------------------------------------------
        # TARGET CONSISTENCY
        # ----------------------------------------------------

        relevant_id = (
            record[
                "relevant_id"
            ]
        )

        actual_in_top5 = (
            relevant_id
            in retrieved_ids
        )

        stored_in_top5 = (
            record[
                "target_in_top5"
            ]
        )

        if (
            actual_in_top5
            != stored_in_top5
        ):

            target_consistency_errors.append(
                {
                    "pair":
                        pair,

                    "reason":
                        "target_in_top5 mismatch",
                }
            )

        actual_rank = None

        if actual_in_top5:

            actual_rank = (
                retrieved_ids.index(
                    relevant_id
                )
                + 1
            )

        stored_rank = (
            record[
                "target_rank_top5"
            ]
        )

        if (
            actual_rank
            != stored_rank
        ):

            target_consistency_errors.append(
                {
                    "pair":
                        pair,

                    "reason":
                        "target_rank_top5 mismatch",

                    "expected":
                        actual_rank,

                    "actual":
                        stored_rank,
                }
            )

        if actual_in_top5:

            top5_counts[
                method
            ] += 1

        # ----------------------------------------------------
        # QUERY-LEVEL SHARED VALUES
        #
        # Ketiga method untuk query yang sama harus berbagi:
        #
        # question
        # category
        # domain
        # relevant_id
        # reference_context
        # reference
        #
        # Yang boleh berbeda:
        #
        # retrieved_ids
        # retrieved_contexts
        # response
        # target rank
        # ----------------------------------------------------

        shared = {
            "question":
                record[
                    "question"
                ],

            "category":
                record[
                    "category"
                ],

            "domain":
                record[
                    "domain"
                ],

            "relevant_id":
                record[
                    "relevant_id"
                ],

            "reference_context":
                record[
                    "reference_context"
                ],

            "reference":
                record[
                    "reference"
                ],
        }

        if (
            query_id
            not in query_shared_values
        ):

            query_shared_values[
                query_id
            ] = shared

        else:

            expected_shared = (
                query_shared_values[
                    query_id
                ]
            )

            for field, value in (
                shared.items()
            ):

                if (
                    value
                    != expected_shared[
                        field
                    ]
                ):

                    per_query_consistency_errors.append(
                        {
                            "query_id":
                                query_id,

                            "method":
                                method,

                            "field":
                                field,
                        }
                    )

    # ========================================================
    # SCHEMA VALIDATION
    # ========================================================

    if schema_errors:

        raise RuntimeError(
            "Schema error ditemukan:\n"
            f"{schema_errors}"
        )

    # ========================================================
    # BASIC VALIDATION
    # ========================================================

    if (
        len(seen_pairs)
        != EXPECTED_COUNT
    ):

        raise RuntimeError(
            "Unique pairs bukan 120."
        )

    if empty_responses:

        raise RuntimeError(
            "Ada response kosong:\n"
            f"{empty_responses}"
        )

    if empty_references:

        raise RuntimeError(
            "Ada reference kosong:\n"
            f"{empty_references}"
        )

    if empty_contexts:

        raise RuntimeError(
            "Ada retrieved context kosong:\n"
            f"{empty_contexts}"
        )

    if question_alias_errors:

        raise RuntimeError(
            "question != user_input:\n"
            f"{question_alias_errors}"
        )

    if target_consistency_errors:

        raise RuntimeError(
            "Target metadata tidak konsisten:\n"
            f"{target_consistency_errors}"
        )

    if per_query_consistency_errors:

        raise RuntimeError(
            "Reference/query metadata berbeda "
            "antar-method:\n"
            f"{per_query_consistency_errors}"
        )

    # ========================================================
    # QUERY COUNTS
    # ========================================================

    if (
        len(query_methods)
        != EXPECTED_QUERY_COUNT
    ):

        raise RuntimeError(
            "Unique query count bukan 40."
        )

    expected_query_ids = {
        f"Q{number:03d}"
        for number in range(
            1,
            41,
        )
    }

    if (
        set(query_methods)
        != expected_query_ids
    ):

        raise RuntimeError(
            "Query IDs bukan Q001-Q040."
        )

    # ========================================================
    # EACH QUERY MUST HAVE ALL 3 METHODS
    # ========================================================

    missing_method_queries = []

    for query_id in sorted(
        query_methods
    ):

        methods = (
            query_methods[
                query_id
            ]
        )

        if (
            methods
            != EXPECTED_METHODS
        ):

            missing_method_queries.append(
                {
                    "query_id":
                        query_id,

                    "methods":
                        sorted(
                            methods
                        ),
                }
            )

    if missing_method_queries:

        raise RuntimeError(
            "Ada query yang tidak memiliki "
            "3 metode lengkap:\n"
            f"{missing_method_queries}"
        )

    # ========================================================
    # DISTRIBUTION VALIDATION
    # ========================================================

    if (
        dict(method_counts)
        != EXPECTED_METHOD_COUNTS
    ):

        raise RuntimeError(
            "Method counts tidak sesuai.\n"
            f"{dict(method_counts)}"
        )

    if (
        dict(category_counts)
        != EXPECTED_CATEGORY_COUNTS
    ):

        raise RuntimeError(
            "Category counts tidak sesuai.\n"
            f"{dict(category_counts)}"
        )

    if (
        dict(domain_counts)
        != EXPECTED_DOMAIN_COUNTS
    ):

        raise RuntimeError(
            "Domain counts tidak sesuai.\n"
            f"{dict(domain_counts)}"
        )

    actual_top5_counts = {
        method:
            top5_counts.get(
                method,
                0,
            )
        for method
        in EXPECTED_METHOD_COUNTS
    }

    if (
        actual_top5_counts
        != EXPECTED_TOP5_COUNTS
    ):

        raise RuntimeError(
            "Target Top-5 counts "
            "tidak sesuai.\n"
            f"Actual: "
            f"{actual_top5_counts}\n"
            f"Expected: "
            f"{EXPECTED_TOP5_COUNTS}"
        )

    # ========================================================
    # LENGTH AUDIT
    # ========================================================

    response_word_counts = [
        len(
            record[
                "response"
            ].split()
        )
        for record
        in records
    ]

    reference_word_counts = [
        len(
            record[
                "reference"
            ].split()
        )
        for record
        in records
    ]

    context_counts = [
        len(
            record[
                "retrieved_contexts"
            ]
        )
        for record
        in records
    ]

    mean_response_words = (
        sum(
            response_word_counts
        )
        / len(
            response_word_counts
        )
    )

    mean_reference_words = (
        sum(
            reference_word_counts
        )
        / len(
            reference_word_counts
        )
    )

    # ========================================================
    # PRINT AUDIT
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "FINAL EVALUATION DATASET AUDIT"
    )

    print(
        "=" * 70
    )

    print(
        f"Records             : "
        f"{len(records)}"
    )

    print(
        f"Unique pairs        : "
        f"{len(seen_pairs)}"
    )

    print(
        f"Unique queries      : "
        f"{len(query_methods)}"
    )

    print(
        f"Empty responses     : "
        f"{len(empty_responses)}"
    )

    print(
        f"Empty references    : "
        f"{len(empty_references)}"
    )

    print(
        f"Empty contexts      : "
        f"{len(empty_contexts)}"
    )

    print(
        f"Target inconsist.   : "
        f"{len(target_consistency_errors)}"
    )

    print(
        f"Cross-method inconsist.: "
        f"{len(per_query_consistency_errors)}"
    )

    print(
        "\nMETHOD COUNTS"
    )

    for method in [
        "bm25",
        "e5",
        "hybrid_rrf",
    ]:

        print(
            f"{method:<15}: "
            f"{method_counts[method]}"
        )

    print(
        "\nTARGET IN TOP-5"
    )

    for method in [
        "bm25",
        "e5",
        "hybrid_rrf",
    ]:

        print(
            f"{method:<15}: "
            f"{top5_counts[method]}/40"
        )

    print(
        "\nTEXT LENGTH"
    )

    print(
        f"Response words "
        f"min/mean/max : "
        f"{min(response_word_counts)} / "
        f"{mean_response_words:.2f} / "
        f"{max(response_word_counts)}"
    )

    print(
        f"Reference words "
        f"min/mean/max: "
        f"{min(reference_word_counts)} / "
        f"{mean_reference_words:.2f} / "
        f"{max(reference_word_counts)}"
    )

    print(
        f"Contexts per row   : "
        f"{min(context_counts)}-"
        f"{max(context_counts)}"
    )

    # ========================================================
    # CREATE FREEZE MANIFEST
    # ========================================================

    manifest = {
        "artifact":
            "Aire Optima RAG "
            "Evaluation Dataset v1",

        "status":
            "FROZEN",

        "evaluation_dataset": {
            "file":
                str(
                    EVALUATION_DATASET_FILE
                ),

            "sha256":
                evaluation_hash,

            "record_count":
                len(
                    records
                ),

            "unique_query_method_pairs":
                len(
                    seen_pairs
                ),

            "unique_queries":
                len(
                    query_methods
                ),
        },

        "source_artifacts": {
            "rag_generation_input": {
                "file":
                    str(
                        RAG_INPUT_FILE
                    ),

                "sha256":
                    rag_input_hash,
            },

            "generated_answers": {
                "file":
                    str(
                        GENERATED_ANSWERS_FILE
                    ),

                "sha256":
                    answers_hash,
            },

            "evaluation_reference": {
                "file":
                    str(
                        REFERENCE_FILE
                    ),

                "sha256":
                    reference_hash,
            },
        },

        "validation": {
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

            "target_in_top5":
                actual_top5_counts,

            "empty_responses":
                len(
                    empty_responses
                ),

            "empty_references":
                len(
                    empty_references
                ),

            "empty_contexts":
                len(
                    empty_contexts
                ),

            "target_consistency_errors":
                len(
                    target_consistency_errors
                ),

            "cross_method_reference_errors":
                len(
                    per_query_consistency_errors
                ),

            "retrieved_contexts_per_row":
                5,
        },

        "text_statistics": {
            "response_word_count": {
                "min":
                    min(
                        response_word_counts
                    ),

                "mean":
                    mean_response_words,

                "max":
                    max(
                        response_word_counts
                    ),
            },

            "reference_word_count": {
                "min":
                    min(
                        reference_word_counts
                    ),

                "mean":
                    mean_reference_words,

                "max":
                    max(
                        reference_word_counts
                    ),
            },
        },

        "evaluation_schema": {
            "user_input":
                "original frozen question",

            "response":
                "frozen Gemini generated answer",

            "retrieved_contexts":
                (
                    "Top-5 frozen context_text "
                    "ordered by retrieval rank"
                ),

            "reference":
                (
                    "frozen curated minimal "
                    "sufficient reference answer"
                ),
        },

        "freeze_rule":
            (
                "Do not modify the evaluation "
                "dataset, generated responses, "
                "retrieved contexts, references, "
                "or source artifacts after "
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

    # ========================================================
    # FINAL
    # ========================================================

    print(
        "\nEVALUATION DATASET SHA256:"
    )

    print(
        evaluation_hash
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