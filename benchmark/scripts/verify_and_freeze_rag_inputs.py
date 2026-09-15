import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


# ============================================================
# 1. PATHS
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

RAG_INPUT_FILE = Path(
    "benchmark/data/rag/"
    "rag_generation_inputs_v1.jsonl"
)

MANIFEST_FILE = Path(
    "benchmark/data/rag/"
    "rag_generation_inputs_manifest_v1.json"
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

EXPECTED_RAG_INPUT_SHA256 = (
    "742937ac9af494eb41fb0662dcdce02a2"
    "cf30f06c1eb02a57c98d37bc7067a93"
)


EXPECTED_DOCUMENTS = 145
EXPECTED_QUERIES = 40
EXPECTED_RAG_INPUTS = 120

METHODS = [
    "bm25",
    "e5",
    "hybrid_rrf",
]

TOP_K = 5


# ============================================================
# 3. SHA256
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
# 4. LOAD JSONL
# ============================================================

def load_jsonl(path: Path):

    rows = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            rows.append(
                json.loads(line)
            )

    return rows


# ============================================================
# 5. LOAD RETRIEVAL TOP-5
# ============================================================

def load_retrieval_rankings():

    groups = {}

    with RANKINGS_FILE.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:

        reader = csv.DictReader(file)

        temp = {}

        for row in reader:

            pair = (
                row["query_id"],
                row["method"],
            )

            temp.setdefault(
                pair,
                []
            )

            temp[pair].append({
                "rank":
                    int(
                        row[
                            "result_rank"
                        ]
                    ),

                "id":
                    row[
                        "result_id"
                    ],
            })

    for pair, rows in temp.items():

        rows.sort(
            key=lambda x:
                x["rank"]
        )

        if len(rows) != TOP_K:

            raise RuntimeError(
                f"{pair}: bukan Top-{TOP_K}"
            )

        expected_ranks = list(
            range(
                1,
                TOP_K + 1
            )
        )

        actual_ranks = [
            row["rank"]
            for row in rows
        ]

        if actual_ranks != expected_ranks:

            raise RuntimeError(
                f"{pair}: urutan ranking "
                "tidak valid."
            )

        groups[pair] = [
            row["id"]
            for row in rows
        ]

    return groups


# ============================================================
# 6. MAIN
# ============================================================

def main():

    print(
        "=" * 70
    )

    print(
        "VERIFY & FREEZE "
        "RAG GENERATION INPUTS V1"
    )

    print(
        "=" * 70
    )

    # ========================================================
    # HASH AUDIT
    # ========================================================

    corpus_hash = sha256_file(
        CORPUS_FILE
    )

    query_hash = sha256_file(
        QUERY_FILE
    )

    rankings_hash = sha256_file(
        RANKINGS_FILE
    )

    rag_hash = sha256_file(
        RAG_INPUT_FILE
    )

    print(
        "\nSHA256 AUDIT"
    )

    print(
        f"Corpus   : {corpus_hash}"
    )

    print(
        f"Query    : {query_hash}"
    )

    print(
        f"Rankings : {rankings_hash}"
    )

    print(
        f"RAG input: {rag_hash}"
    )

    if (
        corpus_hash
        != EXPECTED_CORPUS_SHA256
    ):

        raise RuntimeError(
            "Corpus SHA256 berubah."
        )

    if (
        query_hash
        != EXPECTED_QUERY_SHA256
    ):

        raise RuntimeError(
            "Query SHA256 berubah."
        )

    if (
        rag_hash
        != EXPECTED_RAG_INPUT_SHA256
    ):

        raise RuntimeError(
            "RAG input SHA256 berubah."
        )

    print(
        "\n✅ Semua frozen hash utama valid."
    )

    # ========================================================
    # LOAD
    # ========================================================

    corpus_rows = load_jsonl(
        CORPUS_FILE
    )

    query_rows = load_jsonl(
        QUERY_FILE
    )

    rag_rows = load_jsonl(
        RAG_INPUT_FILE
    )

    retrieval_rankings = (
        load_retrieval_rankings()
    )

    if (
        len(corpus_rows)
        != EXPECTED_DOCUMENTS
    ):
        raise RuntimeError(
            "Jumlah corpus berubah."
        )

    if (
        len(query_rows)
        != EXPECTED_QUERIES
    ):
        raise RuntimeError(
            "Jumlah query berubah."
        )

    if (
        len(rag_rows)
        != EXPECTED_RAG_INPUTS
    ):
        raise RuntimeError(
            "Jumlah RAG input bukan 120."
        )

    # ========================================================
    # MAPS
    # ========================================================

    corpus = {
        row["id"]: row
        for row in corpus_rows
    }

    queries = {
        row["query_id"]: row
        for row in query_rows
    }

    # ========================================================
    # LEAKAGE RULE
    # ========================================================

    forbidden_keys = {
        "relevant_id",
        "relevant_ids",
        "hard_negative_id",
        "hard_negative_ids",
        "ground_truth_answer",
        "target_rank",
        "is_relevant",
    }

    seen_pairs = set()

    method_counter = Counter()

    total_contexts = 0

    # ========================================================
    # RECORD-BY-RECORD AUDIT
    # ========================================================

    for record in rag_rows:

        query_id = record[
            "query_id"
        ]

        method = record[
            "method"
        ]

        pair = (
            query_id,
            method,
        )

        # ----------------------------------------------------
        # UNIQUE PAIR
        # ----------------------------------------------------

        if pair in seen_pairs:

            raise RuntimeError(
                "Duplicate RAG pair: "
                f"{pair}"
            )

        seen_pairs.add(pair)

        # ----------------------------------------------------
        # VALID METHOD
        # ----------------------------------------------------

        if method not in METHODS:

            raise RuntimeError(
                f"Method invalid: "
                f"{method}"
            )

        method_counter[
            method
        ] += 1

        # ----------------------------------------------------
        # QUERY VALIDATION
        # ----------------------------------------------------

        if query_id not in queries:

            raise RuntimeError(
                "Query tidak ada: "
                f"{query_id}"
            )

        frozen_question = (
            queries[
                query_id
            ][
                "question"
            ]
        )

        if (
            record["question"]
            != frozen_question
        ):

            raise RuntimeError(
                f"{query_id}: question "
                "berbeda dari frozen query."
            )

        # ----------------------------------------------------
        # LEAKAGE
        # ----------------------------------------------------

        leaked = (
            forbidden_keys
            & set(
                record.keys()
            )
        )

        if leaked:

            raise RuntimeError(
                f"{pair}: leakage "
                f"{leaked}"
            )

        # ----------------------------------------------------
        # RETRIEVED IDS
        # ----------------------------------------------------

        retrieved_ids = record[
            "retrieved_ids"
        ]

        contexts = record[
            "contexts"
        ]

        if (
            len(retrieved_ids)
            != TOP_K
        ):

            raise RuntimeError(
                f"{pair}: retrieved_ids "
                "bukan 5."
            )

        if (
            len(contexts)
            != TOP_K
        ):

            raise RuntimeError(
                f"{pair}: contexts "
                "bukan 5."
            )

        if (
            len(
                set(
                    retrieved_ids
                )
            )
            != TOP_K
        ):

            raise RuntimeError(
                f"{pair}: duplicate "
                "retrieved ID."
            )

        # ----------------------------------------------------
        # RETRIEVAL ORDER MUST MATCH
        # ----------------------------------------------------

        expected_ids = (
            retrieval_rankings.get(
                pair
            )
        )

        if expected_ids is None:

            raise RuntimeError(
                f"{pair}: ranking source "
                "tidak ditemukan."
            )

        if (
            retrieved_ids
            != expected_ids
        ):

            raise RuntimeError(
                f"{pair}: retrieved IDs "
                "tidak sama dengan "
                "retrieval rankings."
            )

        # ----------------------------------------------------
        # CONTEXT MUST MATCH CORPUS EXACTLY
        # ----------------------------------------------------

        for index, document_id in enumerate(
            retrieved_ids
        ):

            if document_id not in corpus:

                raise RuntimeError(
                    f"{pair}: "
                    f"{document_id} "
                    "tidak ada di corpus."
                )

            expected_context = (
                corpus[
                    document_id
                ][
                    "context_text"
                ]
                .strip()
            )

            actual_context = (
                contexts[
                    index
                ]
                .strip()
            )

            if (
                actual_context
                != expected_context
            ):

                raise RuntimeError(
                    f"{pair}: context "
                    f"rank {index + 1} "
                    "tidak identik dengan "
                    "corpus context_text."
                )

        # ----------------------------------------------------
        # CONTEXT BLOCK MUST BE EXACT
        # ----------------------------------------------------

        expected_block = (
            "\n\n".join(
                [
                    (
                        f"[CONTEXT {index}]\n"
                        f"{context}"
                    )

                    for index, context
                    in enumerate(
                        contexts,
                        start=1,
                    )
                ]
            )
        )

        if (
            record[
                "context_block"
            ]
            != expected_block
        ):

            raise RuntimeError(
                f"{pair}: context_block "
                "tidak konsisten."
            )

        total_contexts += len(
            contexts
        )

    # ========================================================
    # FINAL COUNTS
    # ========================================================

    if (
        len(seen_pairs)
        != EXPECTED_RAG_INPUTS
    ):

        raise RuntimeError(
            "Unique pair bukan 120."
        )

    for method in METHODS:

        if (
            method_counter[
                method
            ]
            != EXPECTED_QUERIES
        ):

            raise RuntimeError(
                f"{method}: jumlah input "
                "bukan 40."
            )

    if total_contexts != 600:

        raise RuntimeError(
            "Total context bukan 600."
        )

    # ========================================================
    # CREATE FREEZE MANIFEST
    # ========================================================

    manifest = {
        "artifact":
            "Aire Optima RAG "
            "Generation Inputs v1",

        "status":
            "FROZEN",

        "rag_input_file":
            str(
                RAG_INPUT_FILE
            ),

        "rag_input_sha256":
            rag_hash,

        "source_artifacts": {
            "corpus_sha256":
                corpus_hash,

            "query_sha256":
                query_hash,

            "retrieval_rankings_sha256":
                rankings_hash,
        },

        "configuration": {
            "query_count":
                40,

            "methods":
                METHODS,

            "inputs_per_method":
                40,

            "top_k":
                5,

            "generation_input_count":
                120,

            "total_context_count":
                600,

            "context_source":
                "context_text",

            "context_order":
                "retrieval_rank",

            "ground_truth_leakage":
                False,
        },

        "freeze_rule":
            (
                "Do not modify RAG generation "
                "inputs after generator "
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
    # SUCCESS
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "FINAL RAG INPUT AUDIT"
    )

    print(
        "=" * 70
    )

    print(
        f"RAG inputs       : "
        f"{len(rag_rows)}"
    )

    print(
        f"Unique pairs     : "
        f"{len(seen_pairs)}"
    )

    print(
        f"Contexts total   : "
        f"{total_contexts}"
    )

    print(
        f"BM25 inputs      : "
        f"{method_counter['bm25']}"
    )

    print(
        f"E5 inputs        : "
        f"{method_counter['e5']}"
    )

    print(
        f"Hybrid inputs    : "
        f"{method_counter['hybrid_rrf']}"
    )

    print(
        "\n✅ Question matches frozen query."
    )

    print(
        "✅ Retrieval Top-5 order matches."
    )

    print(
        "✅ Context matches context_text."
    )

    print(
        "✅ Context block matches."
    )

    print(
        "✅ No ground-truth leakage."
    )

    print(
        "\nRAG INPUT SHA256:"
    )

    print(
        rag_hash
    )

    print(
        "\nRETRIEVAL RANKINGS SHA256:"
    )

    print(
        rankings_hash
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