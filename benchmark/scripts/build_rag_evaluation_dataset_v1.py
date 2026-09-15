import csv
import hashlib
import json
from collections import Counter
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

OUTPUT_DIR = Path(
    "benchmark/data/evaluation"
)

OUTPUT_JSONL = (
    OUTPUT_DIR
    / "rag_evaluation_dataset_v1.jsonl"
)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "rag_evaluation_dataset_v1.csv"
)

AUDIT_CSV = (
    OUTPUT_DIR
    / "rag_evaluation_dataset_audit_v1.csv"
)

METADATA_FILE = (
    OUTPUT_DIR
    / "rag_evaluation_dataset_metadata_v1.json"
)


# ============================================================
# 2. EXPECTED FROZEN HASHES
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


EXPECTED_RAG_INPUT_COUNT = 120
EXPECTED_GENERATED_COUNT = 120
EXPECTED_REFERENCE_COUNT = 40
EXPECTED_EVALUATION_COUNT = 120


EXPECTED_METHOD_COUNTS = {
    "bm25": 40,
    "e5": 40,
    "hybrid_rrf": 40,
}


# ============================================================
# 3. SHA256
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
# 4. LOAD JSONL
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
# 5. MAIN
# ============================================================

def main():

    print(
        "=" * 70
    )

    print(
        "BUILD RAG EVALUATION DATASET V1"
    )

    print(
        "=" * 70
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # FILE EXISTENCE
    # ========================================================

    required_files = [
        RAG_INPUT_FILE,
        GENERATED_ANSWERS_FILE,
        REFERENCE_FILE,
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
            GENERATED_ANSWERS_FILE
        )
    )

    reference_hash = (
        sha256_file(
            REFERENCE_FILE
        )
    )

    print(
        "\nSHA256 AUDIT"
    )

    print(
        f"RAG input : "
        f"{rag_input_hash}"
    )

    print(
        f"Answers   : "
        f"{answers_hash}"
    )

    print(
        f"Reference : "
        f"{reference_hash}"
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
            "Evaluation reference hash berubah."
        )

    print(
        "\n✅ Semua frozen source artifact valid."
    )

    # ========================================================
    # LOAD
    # ========================================================

    rag_inputs = (
        load_jsonl(
            RAG_INPUT_FILE
        )
    )

    generated_answers = (
        load_jsonl(
            GENERATED_ANSWERS_FILE
        )
    )

    references = (
        load_jsonl(
            REFERENCE_FILE
        )
    )

    if (
        len(rag_inputs)
        != EXPECTED_RAG_INPUT_COUNT
    ):

        raise RuntimeError(
            "RAG input count bukan 120."
        )

    if (
        len(generated_answers)
        != EXPECTED_GENERATED_COUNT
    ):

        raise RuntimeError(
            "Generated answer count bukan 120."
        )

    if (
        len(references)
        != EXPECTED_REFERENCE_COUNT
    ):

        raise RuntimeError(
            "Reference count bukan 40."
        )

    # ========================================================
    # INDEX RAG INPUT BY QUERY + METHOD
    # ========================================================

    input_by_pair = {}

    for record in rag_inputs:

        pair = (
            record[
                "query_id"
            ],
            record[
                "method"
            ],
        )

        if pair in input_by_pair:

            raise RuntimeError(
                "Duplicate RAG input pair: "
                f"{pair}"
            )

        input_by_pair[
            pair
        ] = record

    # ========================================================
    # INDEX GENERATED ANSWERS BY QUERY + METHOD
    # ========================================================

    answer_by_pair = {}

    for record in generated_answers:

        pair = (
            record[
                "query_id"
            ],
            record[
                "method"
            ],
        )

        if pair in answer_by_pair:

            raise RuntimeError(
                "Duplicate generated pair: "
                f"{pair}"
            )

        answer_by_pair[
            pair
        ] = record

    # ========================================================
    # INDEX REFERENCE BY QUERY
    # ========================================================

    reference_by_query = {}

    for record in references:

        query_id = (
            record[
                "query_id"
            ]
        )

        if (
            query_id
            in reference_by_query
        ):

            raise RuntimeError(
                "Duplicate reference query: "
                f"{query_id}"
            )

        reference_by_query[
            query_id
        ] = record

    # ========================================================
    # PAIR CONSISTENCY
    # ========================================================

    if (
        set(input_by_pair)
        != set(answer_by_pair)
    ):

        missing_answers = (
            set(input_by_pair)
            - set(answer_by_pair)
        )

        extra_answers = (
            set(answer_by_pair)
            - set(input_by_pair)
        )

        raise RuntimeError(
            "Pair RAG input dan generated "
            "answers tidak identik.\n"
            f"Missing answers: "
            f"{sorted(missing_answers)}\n"
            f"Extra answers: "
            f"{sorted(extra_answers)}"
        )

    # ========================================================
    # BUILD 120 EVALUATION RECORDS
    # ========================================================

    evaluation_records = []

    audit_records = []

    seen_pairs = set()

    # Ikuti exact ordering RAG input.
    for input_record in rag_inputs:

        query_id = (
            input_record[
                "query_id"
            ]
        )

        method = (
            input_record[
                "method"
            ]
        )

        pair = (
            query_id,
            method,
        )

        if pair in seen_pairs:

            raise RuntimeError(
                f"Duplicate pair: {pair}"
            )

        seen_pairs.add(
            pair
        )

        answer_record = (
            answer_by_pair[
                pair
            ]
        )

        if (
            query_id
            not in reference_by_query
        ):

            raise RuntimeError(
                f"{query_id}: "
                "reference tidak ditemukan."
            )

        reference_record = (
            reference_by_query[
                query_id
            ]
        )

        # ====================================================
        # QUESTION MUST MATCH
        # ====================================================

        input_question = (
            input_record[
                "question"
            ]
        )

        reference_question = (
            reference_record[
                "question"
            ]
        )

        if (
            input_question
            != reference_question
        ):

            raise RuntimeError(
                f"{query_id}: question berbeda "
                "antara RAG input dan reference."
            )

        # ====================================================
        # ANSWER MUST BE COMPLETE
        # ====================================================

        generated_answer = (
            answer_record.get(
                "answer"
            )
            or ""
        ).strip()

        if not generated_answer:

            raise RuntimeError(
                f"{pair}: generated answer kosong."
            )

        if (
            answer_record.get(
                "finish_reason"
            )
            != "STOP"
        ):

            raise RuntimeError(
                f"{pair}: finish_reason bukan STOP."
            )

        # ====================================================
        # RETRIEVED CONTEXTS
        # ====================================================

        retrieved_ids = (
            input_record.get(
                "retrieved_ids"
            )
        )

        contexts = (
            input_record.get(
                "contexts"
            )
        )

        if not isinstance(
            retrieved_ids,
            list,
        ):

            raise RuntimeError(
                f"{pair}: "
                "retrieved_ids bukan list."
            )

        if not isinstance(
            contexts,
            list,
        ):

            raise RuntimeError(
                f"{pair}: contexts bukan list."
            )

        if (
            len(retrieved_ids)
            != 5
        ):

            raise RuntimeError(
                f"{pair}: retrieved_ids "
                "bukan Top-5."
            )

        if (
            len(contexts)
            != 5
        ):

            raise RuntimeError(
                f"{pair}: contexts "
                "bukan Top-5."
            )

        if any(
            not isinstance(
                context,
                str,
            )
            or not context.strip()

            for context
            in contexts
        ):

            raise RuntimeError(
                f"{pair}: "
                "ada context kosong."
            )

        # ====================================================
        # REFERENCE
        # ====================================================

        reference_answer = (
            reference_record.get(
                "reference_answer"
            )
            or ""
        ).strip()

        reference_context = (
            reference_record.get(
                "reference_context"
            )
            or ""
        ).strip()

        relevant_id = (
            reference_record[
                "relevant_id"
            ]
        )

        if not reference_answer:

            raise RuntimeError(
                f"{query_id}: "
                "reference answer kosong."
            )

        if not reference_context:

            raise RuntimeError(
                f"{query_id}: "
                "reference context kosong."
            )

        # ====================================================
        # TARGET IN RETRIEVED TOP-5
        # ====================================================

        target_in_top5 = (
            relevant_id
            in retrieved_ids
        )

        target_rank = None

        if target_in_top5:

            target_rank = (
                retrieved_ids.index(
                    relevant_id
                )
                + 1
            )

        # ====================================================
        # FINAL RECORD
        #
        # Ada dua kelompok field:
        #
        # 1. metadata penelitian
        # 2. RAG-evaluator friendly fields
        #
        # user_input
        # response
        # retrieved_contexts
        # reference
        # ====================================================

        evaluation_record = {
            # --------------------------------------------
            # Research metadata
            # --------------------------------------------

            "query_id":
                query_id,

            "method":
                method,

            "category":
                reference_record[
                    "category"
                ],

            "domain":
                reference_record[
                    "domain"
                ],

            "question":
                input_question,

            "retrieved_ids":
                retrieved_ids,

            "relevant_id":
                relevant_id,

            "target_in_top5":
                target_in_top5,

            "target_rank_top5":
                target_rank,

            "reference_context":
                reference_context,

            # --------------------------------------------
            # Evaluation-friendly schema
            # --------------------------------------------

            "user_input":
                input_question,

            "response":
                generated_answer,

            "retrieved_contexts":
                contexts,

            "reference":
                reference_answer,
        }

        evaluation_records.append(
            evaluation_record
        )

        # ====================================================
        # AUDIT ROW
        # ====================================================

        audit_records.append(
            {
                "query_id":
                    query_id,

                "method":
                    method,

                "category":
                    reference_record[
                        "category"
                    ],

                "domain":
                    reference_record[
                        "domain"
                    ],

                "relevant_id":
                    relevant_id,

                "retrieved_rank_1":
                    retrieved_ids[0],

                "retrieved_rank_2":
                    retrieved_ids[1],

                "retrieved_rank_3":
                    retrieved_ids[2],

                "retrieved_rank_4":
                    retrieved_ids[3],

                "retrieved_rank_5":
                    retrieved_ids[4],

                "target_in_top5":
                    target_in_top5,

                "target_rank_top5":
                    (
                        target_rank
                        if target_rank is not None
                        else ""
                    ),

                "finish_reason":
                    answer_record[
                        "finish_reason"
                    ],

                "model_version":
                    answer_record[
                        "model_version"
                    ],
            }
        )

    # ========================================================
    # FINAL COUNT VALIDATION
    # ========================================================

    if (
        len(evaluation_records)
        != EXPECTED_EVALUATION_COUNT
    ):

        raise RuntimeError(
            "Evaluation records bukan 120."
        )

    if (
        len(seen_pairs)
        != EXPECTED_EVALUATION_COUNT
    ):

        raise RuntimeError(
            "Unique query-method "
            "pairs bukan 120."
        )

    # ========================================================
    # METHOD COUNTS
    # ========================================================

    method_counts = Counter(
        record[
            "method"
        ]
        for record
        in evaluation_records
    )

    if (
        dict(method_counts)
        != EXPECTED_METHOD_COUNTS
    ):

        raise RuntimeError(
            "Method counts tidak sesuai.\n"
            f"{dict(method_counts)}"
        )

    # ========================================================
    # CATEGORY / DOMAIN COUNTS
    #
    # Karena tiap query muncul tiga kali,
    # jumlah original dikalikan 3.
    # ========================================================

    category_counts = Counter(
        record[
            "category"
        ]
        for record
        in evaluation_records
    )

    domain_counts = Counter(
        record[
            "domain"
        ]
        for record
        in evaluation_records
    )

    expected_category_counts = {
        "exact": 30,
        "semantic": 30,
        "multi_constraint": 30,
        "fine_grained": 30,
    }

    expected_domain_counts = {
        "product_ac": 36,
        "service_ac": 48,
        "cctv": 36,
    }

    if (
        dict(category_counts)
        != expected_category_counts
    ):

        raise RuntimeError(
            "Evaluation category counts "
            "tidak sesuai.\n"
            f"{dict(category_counts)}"
        )

    if (
        dict(domain_counts)
        != expected_domain_counts
    ):

        raise RuntimeError(
            "Evaluation domain counts "
            "tidak sesuai.\n"
            f"{dict(domain_counts)}"
        )

    # ========================================================
    # TOP-5 TARGET COUNTS
    #
    # Harus konsisten dengan retrieval benchmark:
    #
    # BM25   = 37/40
    # E5     = 38/40
    # Hybrid = 37/40
    # ========================================================

    top5_counts = Counter()

    for record in evaluation_records:

        if (
            record[
                "target_in_top5"
            ]
        ):

            top5_counts[
                record[
                    "method"
                ]
            ] += 1

    expected_top5_counts = {
        "bm25": 37,
        "e5": 38,
        "hybrid_rrf": 37,
    }

    if (
        dict(top5_counts)
        != expected_top5_counts
    ):

        raise RuntimeError(
            "Target Top-5 count tidak "
            "sesuai retrieval benchmark.\n"
            f"Actual: "
            f"{dict(top5_counts)}\n"
            f"Expected: "
            f"{expected_top5_counts}"
        )

    # ========================================================
    # WRITE JSONL
    # ========================================================

    with OUTPUT_JSONL.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as file:

        for record in evaluation_records:

            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    separators=(
                        ",",
                        ":",
                    ),
                )
                + "\n"
            )

    # ========================================================
    # WRITE HUMAN-READABLE CSV
    #
    # List contexts disimpan sebagai JSON string.
    # ========================================================

    csv_fieldnames = [
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
    ]

    with OUTPUT_CSV.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=csv_fieldnames,
        )

        writer.writeheader()

        for record in evaluation_records:

            csv_record = dict(
                record
            )

            csv_record[
                "retrieved_ids"
            ] = json.dumps(
                record[
                    "retrieved_ids"
                ],
                ensure_ascii=False,
            )

            csv_record[
                "retrieved_contexts"
            ] = json.dumps(
                record[
                    "retrieved_contexts"
                ],
                ensure_ascii=False,
            )

            writer.writerow(
                csv_record
            )

    # ========================================================
    # WRITE AUDIT CSV
    # ========================================================

    audit_fieldnames = [
        "query_id",
        "method",
        "category",
        "domain",
        "relevant_id",
        "retrieved_rank_1",
        "retrieved_rank_2",
        "retrieved_rank_3",
        "retrieved_rank_4",
        "retrieved_rank_5",
        "target_in_top5",
        "target_rank_top5",
        "finish_reason",
        "model_version",
    ]

    with AUDIT_CSV.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=audit_fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            audit_records
        )

    # ========================================================
    # OUTPUT HASH
    # ========================================================

    output_hash = (
        sha256_file(
            OUTPUT_JSONL
        )
    )

    audit_hash = (
        sha256_file(
            AUDIT_CSV
        )
    )

    # ========================================================
    # METADATA
    # ========================================================

    metadata = {
        "artifact":
            "Aire Optima RAG "
            "Evaluation Dataset v1",

        "status":
            "BUILT_NOT_YET_FROZEN",

        "record_count":
            len(
                evaluation_records
            ),

        "unique_query_method_pairs":
            len(
                seen_pairs
            ),

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

        "distribution": {
            "methods":
                dict(
                    method_counts
                ),

            "categories":
                dict(
                    category_counts
                ),

            "domains":
                dict(
                    domain_counts
                ),
        },

        "retrieval_target_top5": {
            key:
                top5_counts.get(
                    key,
                    0,
                )

            for key
            in EXPECTED_METHOD_COUNTS
        },

        "evaluation_schema": {
            "user_input":
                "question",

            "response":
                "generated answer",

            "retrieved_contexts":
                "Top-5 context_text in retrieval rank order",

            "reference":
                "frozen curated reference answer",
        },

        "output": {
            "jsonl": {
                "file":
                    str(
                        OUTPUT_JSONL
                    ),

                "sha256":
                    output_hash,
            },

            "csv": {
                "file":
                    str(
                        OUTPUT_CSV
                    ),
            },

            "audit_csv": {
                "file":
                    str(
                        AUDIT_CSV
                    ),

                "sha256":
                    audit_hash,
            },
        },

        "evaluation_rule":
            (
                "Do not modify generated answers, "
                "retrieved contexts, question, or "
                "reference after this dataset is frozen."
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
    # PRINT SUMMARY
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "RAG EVALUATION DATASET SUMMARY"
    )

    print(
        "=" * 70
    )

    print(
        f"Records          : "
        f"{len(evaluation_records)}"
    )

    print(
        f"Unique pairs     : "
        f"{len(seen_pairs)}"
    )

    print(
        "\nMETHOD COUNTS"
    )

    for key in [
        "bm25",
        "e5",
        "hybrid_rrf",
    ]:

        print(
            f"{key:<15}: "
            f"{method_counts[key]}"
        )

    print(
        "\nCATEGORY COUNTS"
    )

    for key in [
        "exact",
        "semantic",
        "multi_constraint",
        "fine_grained",
    ]:

        print(
            f"{key:<20}: "
            f"{category_counts[key]}"
        )

    print(
        "\nDOMAIN COUNTS"
    )

    for key in [
        "product_ac",
        "service_ac",
        "cctv",
    ]:

        print(
            f"{key:<20}: "
            f"{domain_counts[key]}"
        )

    print(
        "\nTARGET IN TOP-5"
    )

    for key in [
        "bm25",
        "e5",
        "hybrid_rrf",
    ]:

        print(
            f"{key:<15}: "
            f"{top5_counts[key]}/40"
        )

    print(
        "\nEVALUATION DATASET SHA256:"
    )

    print(
        output_hash
    )

    print(
        "\nJSONL:"
    )

    print(
        OUTPUT_JSONL
    )

    print(
        "\nCSV:"
    )

    print(
        OUTPUT_CSV
    )

    print(
        "\nAUDIT CSV:"
    )

    print(
        AUDIT_CSV
    )

    print(
        "\nMETADATA:"
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