import hashlib
import json
import shutil
from pathlib import Path


# ============================================================
# 1. PATHS
# ============================================================

CANDIDATE_FILE = Path(
    "benchmark/data/evaluation/"
    "rag_evaluation_reference_candidate_v1.jsonl"
)

CURATED_FILE = Path(
    "benchmark/data/evaluation/"
    "rag_evaluation_reference_curated_v1.jsonl"
)

FINAL_REFERENCE_FILE = Path(
    "benchmark/data/evaluation/"
    "rag_evaluation_reference_v1.jsonl"
)

MANIFEST_FILE = Path(
    "benchmark/data/evaluation/"
    "rag_evaluation_reference_manifest_v1.json"
)


# ============================================================
# 2. EXPECTED HASHES
# ============================================================

EXPECTED_CANDIDATE_SHA256 = (
    "16a8fcce9599bc7d57aab37f975c5a322"
    "4f812bc7f939d4c44f778e344ef10cc"
)

EXPECTED_CURATED_SHA256 = (
    "5a2fa623cba071ca2fa70e1ed1a3930e"
    "9fc084969a300f05d6148269673bb0ba"
)

EXPECTED_COUNT = 40


# ============================================================
# 3. EXPECTED SCHEMA
# ============================================================

EXPECTED_KEYS = {
    "query_id",
    "category",
    "domain",
    "question",
    "relevant_id",
    "reference_context",
    "reference_answer",
}


# ============================================================
# 4. SHA256
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
# 5. LOAD JSONL
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
# 6. VALIDATE SCHEMA
# ============================================================

def validate_schema(
    records,
    label,
):

    for index, record in enumerate(
        records,
        start=1,
    ):

        keys = set(
            record.keys()
        )

        if (
            keys
            != EXPECTED_KEYS
        ):

            raise RuntimeError(
                f"{label} record {index} "
                "memiliki schema berbeda.\n"
                f"Expected: "
                f"{sorted(EXPECTED_KEYS)}\n"
                f"Actual: "
                f"{sorted(keys)}"
            )


# ============================================================
# 7. MAIN
# ============================================================

def main():

    print(
        "=" * 70
    )

    print(
        "VERIFY & FREEZE "
        "RAG EVALUATION REFERENCE V1"
    )

    print(
        "=" * 70
    )

    # ========================================================
    # FILE EXISTENCE
    # ========================================================

    for path in [
        CANDIDATE_FILE,
        CURATED_FILE,
    ]:

        if not path.exists():

            raise FileNotFoundError(
                f"File tidak ditemukan: "
                f"{path}"
            )

    # ========================================================
    # HASH AUDIT
    # ========================================================

    candidate_hash = (
        sha256_file(
            CANDIDATE_FILE
        )
    )

    curated_hash = (
        sha256_file(
            CURATED_FILE
        )
    )

    print(
        "\nSHA256 AUDIT"
    )

    print(
        f"Candidate : "
        f"{candidate_hash}"
    )

    print(
        f"Curated   : "
        f"{curated_hash}"
    )

    if (
        candidate_hash
        != EXPECTED_CANDIDATE_SHA256
    ):

        raise RuntimeError(
            "Candidate reference berubah."
        )

    if (
        curated_hash
        != EXPECTED_CURATED_SHA256
    ):

        raise RuntimeError(
            "Curated reference berubah."
        )

    print(
        "\n✅ Candidate dan curated "
        "hash valid."
    )

    # ========================================================
    # LOAD
    # ========================================================

    candidate_records = (
        load_jsonl(
            CANDIDATE_FILE
        )
    )

    curated_records = (
        load_jsonl(
            CURATED_FILE
        )
    )

    if (
        len(candidate_records)
        != EXPECTED_COUNT
    ):

        raise RuntimeError(
            "Candidate count bukan 40."
        )

    if (
        len(curated_records)
        != EXPECTED_COUNT
    ):

        raise RuntimeError(
            "Curated count bukan 40."
        )

    # ========================================================
    # SCHEMA
    # ========================================================

    validate_schema(
        candidate_records,
        "Candidate",
    )

    validate_schema(
        curated_records,
        "Curated",
    )

    # ========================================================
    # INDEX BY QUERY ID
    # ========================================================

    candidate_by_id = {}

    curated_by_id = {}

    for record in candidate_records:

        query_id = (
            record[
                "query_id"
            ]
        )

        if (
            query_id
            in candidate_by_id
        ):

            raise RuntimeError(
                "Duplicate query_id "
                f"pada candidate: "
                f"{query_id}"
            )

        candidate_by_id[
            query_id
        ] = record

    for record in curated_records:

        query_id = (
            record[
                "query_id"
            ]
        )

        if (
            query_id
            in curated_by_id
        ):

            raise RuntimeError(
                "Duplicate query_id "
                f"pada curated: "
                f"{query_id}"
            )

        curated_by_id[
            query_id
        ] = record

    if (
        set(candidate_by_id)
        != set(curated_by_id)
    ):

        raise RuntimeError(
            "Query ID candidate dan "
            "curated tidak sama."
        )

    # ========================================================
    # STRICT FIELD COMPARISON
    #
    # Yang boleh berubah HANYA reference_answer.
    # ========================================================

    immutable_fields = [
        "query_id",
        "category",
        "domain",
        "question",
        "relevant_id",
        "reference_context",
    ]

    changed_fields = []

    empty_reference_answers = []

    candidate_nonempty_answers = []

    for query_id in sorted(
        candidate_by_id
    ):

        candidate = (
            candidate_by_id[
                query_id
            ]
        )

        curated = (
            curated_by_id[
                query_id
            ]
        )

        # ----------------------------------------------------
        # CANDIDATE REFERENCE ANSWER MUST BE EMPTY
        # ----------------------------------------------------

        candidate_answer = (
            candidate.get(
                "reference_answer"
            )
            or ""
        ).strip()

        if candidate_answer:

            candidate_nonempty_answers.append(
                query_id
            )

        # ----------------------------------------------------
        # IMMUTABLE FIELDS MUST MATCH EXACTLY
        # ----------------------------------------------------

        for field in immutable_fields:

            if (
                candidate[
                    field
                ]
                != curated[
                    field
                ]
            ):

                changed_fields.append(
                    {
                        "query_id":
                            query_id,

                        "field":
                            field,
                    }
                )

        # ----------------------------------------------------
        # CURATED ANSWER MUST EXIST
        # ----------------------------------------------------

        curated_answer = (
            curated.get(
                "reference_answer"
            )
            or ""
        ).strip()

        if not curated_answer:

            empty_reference_answers.append(
                query_id
            )

    # ========================================================
    # VALIDATION
    # ========================================================

    if candidate_nonempty_answers:

        raise RuntimeError(
            "Candidate seharusnya memiliki "
            "reference_answer kosong.\n"
            f"{candidate_nonempty_answers}"
        )

    if changed_fields:

        raise RuntimeError(
            "Ditemukan perubahan pada field "
            "selain reference_answer:\n"
            f"{changed_fields}"
        )

    if empty_reference_answers:

        raise RuntimeError(
            "Masih ada reference_answer "
            "kosong:\n"
            f"{empty_reference_answers}"
        )

    # ========================================================
    # QUERY RANGE AUDIT
    # ========================================================

    expected_query_ids = {
        f"Q{number:03d}"
        for number in range(
            1,
            41,
        )
    }

    actual_query_ids = set(
        curated_by_id
    )

    if (
        actual_query_ids
        != expected_query_ids
    ):

        missing = (
            expected_query_ids
            - actual_query_ids
        )

        unexpected = (
            actual_query_ids
            - expected_query_ids
        )

        raise RuntimeError(
            "Query ID tidak sesuai.\n"
            f"Missing: "
            f"{sorted(missing)}\n"
            f"Unexpected: "
            f"{sorted(unexpected)}"
        )

    # ========================================================
    # CATEGORY COUNTS
    # ========================================================

    category_counts = {}

    domain_counts = {}

    for record in curated_records:

        category = (
            record[
                "category"
            ]
        )

        domain = (
            record[
                "domain"
            ]
        )

        category_counts[
            category
        ] = (
            category_counts.get(
                category,
                0,
            )
            + 1
        )

        domain_counts[
            domain
        ] = (
            domain_counts.get(
                domain,
                0,
            )
            + 1
        )

    expected_category_counts = {
        "exact": 10,
        "semantic": 10,
        "multi_constraint": 10,
        "fine_grained": 10,
    }

    expected_domain_counts = {
        "product_ac": 12,
        "service_ac": 16,
        "cctv": 12,
    }

    if (
        category_counts
        != expected_category_counts
    ):

        raise RuntimeError(
            "Category counts berubah.\n"
            f"{category_counts}"
        )

    if (
        domain_counts
        != expected_domain_counts
    ):

        raise RuntimeError(
            "Domain counts berubah.\n"
            f"{domain_counts}"
        )

    # ========================================================
    # CREATE FINAL FROZEN FILE
    #
    # Byte-for-byte copy dari curated.
    # ========================================================

    shutil.copyfile(
        CURATED_FILE,
        FINAL_REFERENCE_FILE,
    )

    final_hash = (
        sha256_file(
            FINAL_REFERENCE_FILE
        )
    )

    if (
        final_hash
        != curated_hash
    ):

        raise RuntimeError(
            "Final reference hash berbeda "
            "dari curated."
        )

    # ========================================================
    # FINAL FILE RE-READ
    # ========================================================

    final_records = (
        load_jsonl(
            FINAL_REFERENCE_FILE
        )
    )

    if (
        len(final_records)
        != EXPECTED_COUNT
    ):

        raise RuntimeError(
            "Final reference count "
            "bukan 40."
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    answer_lengths = [
        len(
            record[
                "reference_answer"
            ].split()
        )
        for record
        in curated_records
    ]

    mean_answer_words = (
        sum(
            answer_lengths
        )
        / len(
            answer_lengths
        )
    )

    min_answer_words = min(
        answer_lengths
    )

    max_answer_words = max(
        answer_lengths
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "REFERENCE DATASET AUDIT"
    )

    print(
        "=" * 70
    )

    print(
        f"Records             : "
        f"{len(curated_records)}"
    )

    print(
        f"Unique query IDs    : "
        f"{len(curated_by_id)}"
    )

    print(
        f"Reference answers   : "
        f"{len(curated_records)}"
    )

    print(
        f"Empty answers       : "
        f"{len(empty_reference_answers)}"
    )

    print(
        "Immutable changes   : "
        f"{len(changed_fields)}"
    )

    print(
        "\nCATEGORY COUNTS"
    )

    for key, value in (
        category_counts.items()
    ):

        print(
            f"{key:<20}: "
            f"{value}"
        )

    print(
        "\nDOMAIN COUNTS"
    )

    for key, value in (
        domain_counts.items()
    ):

        print(
            f"{key:<20}: "
            f"{value}"
        )

    print(
        "\nREFERENCE ANSWER LENGTH"
    )

    print(
        f"Min words  : "
        f"{min_answer_words}"
    )

    print(
        f"Mean words : "
        f"{mean_answer_words:.2f}"
    )

    print(
        f"Max words  : "
        f"{max_answer_words}"
    )

    # ========================================================
    # MANIFEST
    # ========================================================

    manifest = {
        "artifact":
            "Aire Optima RAG "
            "Evaluation Reference v1",

        "status":
            "FROZEN",

        "record_count":
            len(
                curated_records
            ),

        "reference_file": {
            "file":
                str(
                    FINAL_REFERENCE_FILE
                ),

            "sha256":
                final_hash,
        },

        "source_candidate": {
            "file":
                str(
                    CANDIDATE_FILE
                ),

            "sha256":
                candidate_hash,
        },

        "curated_source": {
            "file":
                str(
                    CURATED_FILE
                ),

            "sha256":
                curated_hash,
        },

        "validation": {
            "unique_query_ids":
                len(
                    curated_by_id
                ),

            "reference_answers_nonempty":
                len(
                    curated_records
                )
                - len(
                    empty_reference_answers
                ),

            "immutable_field_changes":
                len(
                    changed_fields
                ),

            "category_counts":
                category_counts,

            "domain_counts":
                domain_counts,

            "reference_answer_word_length": {
                "min":
                    min_answer_words,

                "mean":
                    mean_answer_words,

                "max":
                    max_answer_words,
            },
        },

        "reference_policy": {
            "basis": [
                "question",
                "relevant_id",
                "reference_context",
            ],

            "generated_answers_used":
                False,

            "retrieval_rankings_used":
                False,

            "principle":
                (
                    "Minimal sufficient answer; "
                    "all factual content grounded "
                    "in frozen reference_context."
                ),
        },

        "freeze_rule":
            (
                "Do not modify question, "
                "relevant_id, reference_context, "
                "or reference_answer after "
                "RAG evaluation begins."
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
    # FINAL PRINT
    # ========================================================

    print(
        "\nFINAL REFERENCE SHA256:"
    )

    print(
        final_hash
    )

    print(
        "\nFinal reference:"
    )

    print(
        FINAL_REFERENCE_FILE
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