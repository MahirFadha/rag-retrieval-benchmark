import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


# ============================================================
# 1. CONFIG
# ============================================================

CORPUS_FILE = Path(
    "benchmark/data/canonical/"
    "aire_catalog_final_v1.jsonl"
)

QUERY_FILE = Path(
    "benchmark/data/canonical/"
    "aire_benchmark_queries_v1.jsonl"
)

RANKINGS_FILE = Path(
    "benchmark/results/"
    "retrieval_rankings_v1.csv"
)

OUTPUT_DIR = Path(
    "benchmark/data/rag"
)

RAG_INPUT_FILE = (
    OUTPUT_DIR
    / "rag_generation_inputs_v1.jsonl"
)

RAG_AUDIT_FILE = (
    OUTPUT_DIR
    / "rag_generation_inputs_audit_v1.csv"
)

RAG_METADATA_FILE = (
    OUTPUT_DIR
    / "rag_generation_inputs_metadata_v1.json"
)


# ============================================================
# 2. FROZEN HASHES
# ============================================================

EXPECTED_CORPUS_SHA256 = (
    "7e5c041917e43b32298ecf7dc02b571e"
    "11ffe6cf5c77b52dcf8b531d6b4609be"
)

EXPECTED_QUERY_SHA256 = (
    "dfbf771a3f7dc00d465be4e2605f9cca"
    "e7b0e2147933954d367cfcd0751bfb0d"
)


# ============================================================
# 3. BENCHMARK CONFIG
# ============================================================

EXPECTED_DOCUMENTS = 145
EXPECTED_QUERIES = 40

METHODS = [
    "bm25",
    "e5",
    "hybrid_rrf",
]

TOP_K = 5

EXPECTED_RAG_INPUTS = (
    EXPECTED_QUERIES
    * len(METHODS)
)

EXPECTED_RANKING_ROWS = (
    EXPECTED_RAG_INPUTS
    * TOP_K
)


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
# 5. VERIFY FROZEN INPUT
# ============================================================

def verify_frozen_artifacts():

    print(
        "Memverifikasi frozen corpus "
        "dan query set..."
    )

    corpus_hash = sha256_file(
        CORPUS_FILE
    )

    query_hash = sha256_file(
        QUERY_FILE
    )

    if (
        corpus_hash
        != EXPECTED_CORPUS_SHA256
    ):
        raise RuntimeError(
            "SHA256 corpus berubah.\n"
            f"Expected: "
            f"{EXPECTED_CORPUS_SHA256}\n"
            f"Actual  : "
            f"{corpus_hash}"
        )

    if (
        query_hash
        != EXPECTED_QUERY_SHA256
    ):
        raise RuntimeError(
            "SHA256 query set berubah.\n"
            f"Expected: "
            f"{EXPECTED_QUERY_SHA256}\n"
            f"Actual  : "
            f"{query_hash}"
        )

    print(
        "✅ Corpus SHA256 valid."
    )

    print(
        "✅ Query SHA256 valid."
    )

    return (
        corpus_hash,
        query_hash,
    )


# ============================================================
# 6. LOAD CORPUS
# ============================================================

def load_corpus():

    records = {}

    with CORPUS_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            record = json.loads(
                line
            )

            document_id = (
                record["id"]
            )

            if (
                document_id
                in records
            ):
                raise RuntimeError(
                    "Duplicate corpus ID: "
                    f"{document_id}"
                )

            context_text = (
                record.get(
                    "context_text",
                    ""
                )
                or ""
            ).strip()

            if not context_text:

                raise RuntimeError(
                    "context_text kosong: "
                    f"{document_id}"
                )

            records[
                document_id
            ] = record

    if (
        len(records)
        != EXPECTED_DOCUMENTS
    ):

        raise RuntimeError(
            "Jumlah dokumen corpus "
            "tidak sesuai.\n"
            f"Expected: "
            f"{EXPECTED_DOCUMENTS}\n"
            f"Actual  : "
            f"{len(records)}"
        )

    return records


# ============================================================
# 7. LOAD QUERIES
# ============================================================

def load_queries():

    queries = {}

    with QUERY_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            query = json.loads(
                line
            )

            query_id = (
                query["query_id"]
            )

            if (
                query_id
                in queries
            ):
                raise RuntimeError(
                    "Duplicate query ID: "
                    f"{query_id}"
                )

            queries[
                query_id
            ] = query

    if (
        len(queries)
        != EXPECTED_QUERIES
    ):

        raise RuntimeError(
            "Jumlah query tidak sesuai.\n"
            f"Expected: "
            f"{EXPECTED_QUERIES}\n"
            f"Actual  : "
            f"{len(queries)}"
        )

    return queries


# ============================================================
# 8. LOAD RETRIEVAL RANKINGS
# ============================================================

def load_rankings():

    grouped = defaultdict(
        list
    )

    total_rows = 0

    with RANKINGS_FILE.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:

        reader = csv.DictReader(
            file
        )

        for row in reader:

            total_rows += 1

            query_id = row[
                "query_id"
            ]

            method = row[
                "method"
            ]

            rank = int(
                row[
                    "result_rank"
                ]
            )

            document_id = row[
                "result_id"
            ]

            grouped[
                (
                    query_id,
                    method,
                )
            ].append({
                "rank":
                    rank,

                "document_id":
                    document_id,
            })

    if (
        total_rows
        != EXPECTED_RANKING_ROWS
    ):

        raise RuntimeError(
            "Jumlah ranking rows "
            "tidak sesuai.\n"
            f"Expected: "
            f"{EXPECTED_RANKING_ROWS}\n"
            f"Actual  : "
            f"{total_rows}"
        )

    return grouped


# ============================================================
# 9. VALIDATE ONE RANKING GROUP
# ============================================================

def validate_ranking_group(
    query_id,
    method,
    ranking_rows,
    corpus,
):

    if (
        len(ranking_rows)
        != TOP_K
    ):

        raise RuntimeError(
            f"{query_id} / {method}: "
            f"jumlah context bukan {TOP_K}."
        )

    ranking_rows = sorted(
        ranking_rows,
        key=lambda item:
            item["rank"]
    )

    actual_ranks = [
        item["rank"]
        for item
        in ranking_rows
    ]

    expected_ranks = list(
        range(
            1,
            TOP_K + 1,
        )
    )

    if (
        actual_ranks
        != expected_ranks
    ):

        raise RuntimeError(
            f"{query_id} / {method}: "
            "ranking tidak berurutan.\n"
            f"Expected: "
            f"{expected_ranks}\n"
            f"Actual  : "
            f"{actual_ranks}"
        )

    retrieved_ids = [
        item[
            "document_id"
        ]
        for item
        in ranking_rows
    ]

    if (
        len(retrieved_ids)
        != len(
            set(
                retrieved_ids
            )
        )
    ):

        raise RuntimeError(
            f"{query_id} / {method}: "
            "duplicate document pada Top-5."
        )

    for document_id in retrieved_ids:

        if (
            document_id
            not in corpus
        ):

            raise RuntimeError(
                f"{query_id} / {method}: "
                f"{document_id} "
                "tidak ada di corpus."
            )

    return ranking_rows


# ============================================================
# 10. BUILD CONTEXT BLOCK
# ============================================================

def build_context_block(
    contexts,
):

    blocks = []

    for index, context_text in enumerate(
        contexts,
        start=1,
    ):

        blocks.append(
            f"[CONTEXT {index}]\n"
            f"{context_text}"
        )

    return "\n\n".join(
        blocks
    )


# ============================================================
# 11. BUILD GENERATION INPUTS
# ============================================================

def build_generation_inputs(
    corpus,
    queries,
    rankings,
):

    generation_inputs = []
    audit_rows = []

    seen_pairs = set()

    for query_id in sorted(
        queries.keys()
    ):

        query = queries[
            query_id
        ]

        question = query[
            "question"
        ]

        relevant_id = query[
            "relevant_ids"
        ][0]

        hard_negative_ids = set(
            query.get(
                "hard_negative_ids",
                [],
            )
        )

        for method in METHODS:

            pair = (
                query_id,
                method,
            )

            if pair in seen_pairs:

                raise RuntimeError(
                    "Duplicate query-method pair: "
                    f"{pair}"
                )

            seen_pairs.add(
                pair
            )

            ranking_rows = (
                rankings.get(
                    pair
                )
            )

            if ranking_rows is None:

                raise RuntimeError(
                    "Ranking tidak ditemukan: "
                    f"{query_id} / "
                    f"{method}"
                )

            ranking_rows = (
                validate_ranking_group(
                    query_id=query_id,
                    method=method,
                    ranking_rows=
                        ranking_rows,
                    corpus=corpus,
                )
            )

            retrieved_ids = [
                item[
                    "document_id"
                ]
                for item
                in ranking_rows
            ]

            contexts = [
                corpus[
                    document_id
                ][
                    "context_text"
                ].strip()

                for document_id
                in retrieved_ids
            ]

            context_block = (
                build_context_block(
                    contexts
                )
            )

            # =================================================
            # IMPORTANT:
            #
            # File generation input TIDAK mengandung:
            #
            # - relevant_id
            # - hard_negative_ids
            # - ground_truth_answer
            #
            # =================================================

            generation_inputs.append({
                "query_id":
                    query_id,

                "method":
                    method,

                "question":
                    question,

                "retrieved_ids":
                    retrieved_ids,

                "contexts":
                    contexts,

                "context_block":
                    context_block,
            })

            # -------------------------------------------------
            # Audit file BOLEH menggunakan ground truth karena
            # audit ini tidak dikirim ke generator.
            # -------------------------------------------------

            target_rank = None

            if (
                relevant_id
                in retrieved_ids
            ):

                target_rank = (
                    retrieved_ids.index(
                        relevant_id
                    )
                    + 1
                )

            hard_negatives_in_top5 = [
                document_id
                for document_id
                in retrieved_ids

                if (
                    document_id
                    in hard_negative_ids
                )
            ]

            total_context_chars = sum(
                len(context)
                for context
                in contexts
            )

            total_context_words = sum(
                len(
                    context.split()
                )
                for context
                in contexts
            )

            audit_rows.append({
                "query_id":
                    query_id,

                "category":
                    query[
                        "category"
                    ],

                "domain":
                    query[
                        "domain"
                    ],

                "method":
                    method,

                "relevant_id":
                    relevant_id,

                "target_in_top5":
                    (
                        target_rank
                        is not None
                    ),

                "target_rank":
                    (
                        target_rank
                        if (
                            target_rank
                            is not None
                        )
                        else ""
                    ),

                "retrieved_ids":
                    ";".join(
                        retrieved_ids
                    ),

                "hard_negatives_in_top5":
                    ";".join(
                        hard_negatives_in_top5
                    ),

                "context_count":
                    len(contexts),

                "total_context_chars":
                    total_context_chars,

                "total_context_words":
                    total_context_words,

                "ground_truth_leakage":
                    False,

                "audit_result":
                    "PASS",
            })

    if (
        len(generation_inputs)
        != EXPECTED_RAG_INPUTS
    ):

        raise RuntimeError(
            "Jumlah RAG input "
            "tidak sesuai.\n"
            f"Expected: "
            f"{EXPECTED_RAG_INPUTS}\n"
            f"Actual  : "
            f"{len(generation_inputs)}"
        )

    if (
        len(seen_pairs)
        != EXPECTED_RAG_INPUTS
    ):

        raise RuntimeError(
            "Jumlah unique query-method "
            "pair tidak sesuai."
        )

    return (
        generation_inputs,
        audit_rows,
    )


# ============================================================
# 12. LEAKAGE AUDIT
# ============================================================

def audit_generation_input_leakage(
    generation_inputs,
):

    forbidden_keys = {
        "relevant_id",
        "relevant_ids",
        "hard_negative_id",
        "hard_negative_ids",
        "ground_truth_answer",
        "target_rank",
        "is_relevant",
    }

    for record in generation_inputs:

        present_forbidden = (
            forbidden_keys
            & set(
                record.keys()
            )
        )

        if present_forbidden:

            raise RuntimeError(
                "Ground-truth leakage "
                "ditemukan pada "
                f"{record['query_id']} / "
                f"{record['method']}:\n"
                f"{present_forbidden}"
            )

    print(
        "✅ Tidak ada ground-truth "
        "field pada generation input."
    )


# ============================================================
# 13. SAVE JSONL
# ============================================================

def save_generation_inputs(
    generation_inputs,
):

    with RAG_INPUT_FILE.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as file:

        for record in generation_inputs:

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


# ============================================================
# 14. SAVE AUDIT CSV
# ============================================================

def save_audit(
    audit_rows,
):

    fieldnames = [
        "query_id",
        "category",
        "domain",
        "method",
        "relevant_id",
        "target_in_top5",
        "target_rank",
        "retrieved_ids",
        "hard_negatives_in_top5",
        "context_count",
        "total_context_chars",
        "total_context_words",
        "ground_truth_leakage",
        "audit_result",
    ]

    with RAG_AUDIT_FILE.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            audit_rows
        )


# ============================================================
# 15. MAIN
# ============================================================

def main():

    print(
        "=" * 70
    )

    print(
        "BUILD RAG GENERATION INPUTS V1"
    )

    print(
        "=" * 70
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        corpus_hash,
        query_hash,
    ) = verify_frozen_artifacts()

    print(
        "\nMemuat corpus..."
    )

    corpus = load_corpus()

    print(
        f"✅ Corpus loaded: "
        f"{len(corpus)}"
    )

    print(
        "\nMemuat query set..."
    )

    queries = load_queries()

    print(
        f"✅ Queries loaded: "
        f"{len(queries)}"
    )

    print(
        "\nMemuat retrieval rankings..."
    )

    rankings = load_rankings()

    print(
        f"✅ Query-method groups: "
        f"{len(rankings)}"
    )

    print(
        "\nMembangun RAG inputs..."
    )

    (
        generation_inputs,
        audit_rows,
    ) = build_generation_inputs(
        corpus=corpus,
        queries=queries,
        rankings=rankings,
    )

    audit_generation_input_leakage(
        generation_inputs
    )

    save_generation_inputs(
        generation_inputs
    )

    save_audit(
        audit_rows
    )

    # ========================================================
    # HASH RAG INPUT
    # ========================================================

    rag_input_hash = sha256_file(
        RAG_INPUT_FILE
    )

    # ========================================================
    # SUMMARY STATISTICS
    # ========================================================

    method_counts = {
        method: 0
        for method
        in METHODS
    }

    target_present_counts = {
        method: 0
        for method
        in METHODS
    }

    total_chars = []

    total_words = []

    for row in audit_rows:

        method = row[
            "method"
        ]

        method_counts[
            method
        ] += 1

        if row[
            "target_in_top5"
        ]:

            target_present_counts[
                method
            ] += 1

        total_chars.append(
            row[
                "total_context_chars"
            ]
        )

        total_words.append(
            row[
                "total_context_words"
            ]
        )

    # ========================================================
    # METADATA
    # ========================================================

    metadata = {
        "artifact":
            "Aire Optima RAG Generation Inputs v1",

        "status":
            "BUILT_NOT_YET_FROZEN",

        "rag_input_file":
            str(
                RAG_INPUT_FILE
            ),

        "rag_input_sha256":
            rag_input_hash,

        "source_artifacts": {
            "corpus_file":
                str(
                    CORPUS_FILE
                ),

            "corpus_sha256":
                corpus_hash,

            "query_file":
                str(
                    QUERY_FILE
                ),

            "query_sha256":
                query_hash,

            "retrieval_rankings_file":
                str(
                    RANKINGS_FILE
                ),
        },

        "configuration": {
            "query_count":
                EXPECTED_QUERIES,

            "methods":
                METHODS,

            "top_k":
                TOP_K,

            "expected_generation_inputs":
                EXPECTED_RAG_INPUTS,

            "context_source":
                "context_text",

            "context_order":
                "retrieval_rank",

            "ground_truth_in_generation_input":
                False,
        },

        "audit_summary": {
            "generation_input_count":
                len(
                    generation_inputs
                ),

            "method_counts":
                method_counts,

            "target_in_top5_counts":
                target_present_counts,

            "min_context_chars":
                min(
                    total_chars
                ),

            "max_context_chars":
                max(
                    total_chars
                ),

            "mean_context_chars":
                (
                    sum(
                        total_chars
                    )
                    / len(
                        total_chars
                    )
                ),

            "min_context_words":
                min(
                    total_words
                ),

            "max_context_words":
                max(
                    total_words
                ),

            "mean_context_words":
                (
                    sum(
                        total_words
                    )
                    / len(
                        total_words
                    )
                ),
        },
    }

    with RAG_METADATA_FILE.open(
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
    # FINAL PRINT
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "RAG INPUT AUDIT SUMMARY"
    )

    print(
        "=" * 70
    )

    print(
        f"Generation inputs : "
        f"{len(generation_inputs)}"
    )

    for method in METHODS:

        print(
            f"{method:<12}: "
            f"{method_counts[method]} inputs | "
            f"target Top-5 = "
            f"{target_present_counts[method]}"
        )

    print(
        "\nContext size:"
    )

    print(
        f"Chars  min/max/mean : "
        f"{min(total_chars)} / "
        f"{max(total_chars)} / "
        f"{sum(total_chars) / len(total_chars):.2f}"
    )

    print(
        f"Words  min/max/mean : "
        f"{min(total_words)} / "
        f"{max(total_words)} / "
        f"{sum(total_words) / len(total_words):.2f}"
    )

    print(
        "\nRAG INPUT SHA256:"
    )

    print(
        rag_input_hash
    )

    print(
        "\n✅ RAG GENERATION INPUT BUILD SELESAI"
    )

    print(
        "\nGeneration input:"
    )

    print(
        RAG_INPUT_FILE
    )

    print(
        "\nAudit:"
    )

    print(
        RAG_AUDIT_FILE
    )

    print(
        "\nMetadata:"
    )

    print(
        RAG_METADATA_FILE
    )

    print(
        "\nSTATUS:"
    )

    print(
        "BUILT - BELUM FROZEN"
    )


if __name__ == "__main__":
    main()