import csv
import hashlib
import json
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

OUTPUT_DIR = Path(
    "benchmark/data/evaluation"
)

OUTPUT_JSONL = (
    OUTPUT_DIR
    / "rag_evaluation_reference_candidate_v1.jsonl"
)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "rag_evaluation_reference_candidate_v1.csv"
)

METADATA_FILE = (
    OUTPUT_DIR
    / "rag_evaluation_reference_candidate_metadata_v1.json"
)


# ============================================================
# 2. EXPECTED FROZEN HASHES
# ============================================================

EXPECTED_CORPUS_SHA256 = (
    "7e5c041917e43b32298ecf7dc02b571e1"
    "1ffe6cf5c77b52dcf8b531d6b4609be"
)

EXPECTED_QUERY_SHA256 = (
    "dfbf771a3f7dc00d465be4e2605f9cca"
    "e7b0e2147933954d367cfcd0751bfb0d"
)

EXPECTED_DOCUMENT_COUNT = 145
EXPECTED_QUERY_COUNT = 40


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
                    f"{path}, line "
                    f"{line_number}."
                ) from exc

    return records


# ============================================================
# 5. GET SINGLE RELEVANT ID
# ============================================================

def get_relevant_id(query_record):

    # --------------------------------------------------------
    # Format utama benchmark:
    # relevant_ids = [...]
    # --------------------------------------------------------

    if "relevant_ids" in query_record:

        relevant_ids = (
            query_record[
                "relevant_ids"
            ]
        )

        if not isinstance(
            relevant_ids,
            list,
        ):

            raise RuntimeError(
                f"{query_record['query_id']}: "
                "relevant_ids bukan list."
            )

        if len(relevant_ids) != 1:

            raise RuntimeError(
                f"{query_record['query_id']}: "
                "benchmark membutuhkan tepat "
                "1 relevant document."
            )

        return relevant_ids[0]

    # --------------------------------------------------------
    # Fallback kalau schema menggunakan relevant_id.
    # --------------------------------------------------------

    if "relevant_id" in query_record:

        relevant_id = (
            query_record[
                "relevant_id"
            ]
        )

        if not relevant_id:

            raise RuntimeError(
                f"{query_record['query_id']}: "
                "relevant_id kosong."
            )

        return relevant_id

    raise RuntimeError(
        f"{query_record['query_id']}: "
        "tidak ditemukan relevant_id(s)."
    )


# ============================================================
# 6. MAIN
# ============================================================

def main():

    print(
        "=" * 70
    )

    print(
        "BUILD RAG EVALUATION "
        "REFERENCE CANDIDATES V1"
    )

    print(
        "=" * 70
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # VERIFY FROZEN ARTIFACTS
    # ========================================================

    corpus_hash = sha256_file(
        CORPUS_FILE
    )

    query_hash = sha256_file(
        QUERY_FILE
    )

    print(
        "\nSHA256 AUDIT"
    )

    print(
        f"Corpus : {corpus_hash}"
    )

    print(
        f"Query  : {query_hash}"
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

    print(
        "✅ Frozen corpus dan query valid."
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

    if (
        len(corpus_rows)
        != EXPECTED_DOCUMENT_COUNT
    ):

        raise RuntimeError(
            "Jumlah corpus bukan 145."
        )

    if (
        len(query_rows)
        != EXPECTED_QUERY_COUNT
    ):

        raise RuntimeError(
            "Jumlah query bukan 40."
        )

    corpus_by_id = {
        row["id"]:
            row

        for row
        in corpus_rows
    }

    # ========================================================
    # BUILD REFERENCE CANDIDATES
    # ========================================================

    reference_records = []

    seen_query_ids = set()

    for query in query_rows:

        query_id = (
            query[
                "query_id"
            ]
        )

        if query_id in seen_query_ids:

            raise RuntimeError(
                f"Duplicate query_id: "
                f"{query_id}"
            )

        seen_query_ids.add(
            query_id
        )

        relevant_id = (
            get_relevant_id(
                query
            )
        )

        if (
            relevant_id
            not in corpus_by_id
        ):

            raise RuntimeError(
                f"{query_id}: relevant ID "
                f"{relevant_id} tidak ada "
                "di frozen corpus."
            )

        document = (
            corpus_by_id[
                relevant_id
            ]
        )

        reference_context = (
            document.get(
                "context_text"
            )
            or ""
        ).strip()

        if not reference_context:

            raise RuntimeError(
                f"{query_id}: "
                "reference_context kosong."
            )

        record = {
            "query_id":
                query_id,

            "category":
                query.get(
                    "category"
                ),

            "domain":
                query.get(
                    "domain"
                ),

            "question":
                query[
                    "question"
                ],

            "relevant_id":
                relevant_id,

            "reference_context":
                reference_context,

            # --------------------------------------------
            # Sengaja kosong.
            #
            # Ini akan dikurasi secara terkontrol
            # berdasarkan QUESTION +
            # REFERENCE_CONTEXT.
            # --------------------------------------------
            "reference_answer":
                "",
        }

        reference_records.append(
            record
        )

    # ========================================================
    # VALIDATE
    # ========================================================

    if (
        len(reference_records)
        != EXPECTED_QUERY_COUNT
    ):

        raise RuntimeError(
            "Reference candidate "
            "count bukan 40."
        )

    # ========================================================
    # WRITE JSONL
    # ========================================================

    with OUTPUT_JSONL.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as file:

        for record in reference_records:

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
    # WRITE CSV
    # ========================================================

    fieldnames = [
        "query_id",
        "category",
        "domain",
        "question",
        "relevant_id",
        "reference_context",
        "reference_answer",
    ]

    with OUTPUT_CSV.open(
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
            reference_records
        )

    # ========================================================
    # HASH CANDIDATE
    # ========================================================

    candidate_hash = sha256_file(
        OUTPUT_JSONL
    )

    # ========================================================
    # METADATA
    # ========================================================

    metadata = {
        "artifact":
            "Aire Optima RAG Evaluation "
            "Reference Candidate v1",

        "status":
            "BUILT_NOT_YET_FROZEN",

        "record_count":
            len(
                reference_records
            ),

        "source_artifacts": {
            "corpus": {
                "file":
                    str(
                        CORPUS_FILE
                    ),

                "sha256":
                    corpus_hash,
            },

            "query_set": {
                "file":
                    str(
                        QUERY_FILE
                    ),

                "sha256":
                    query_hash,
            },
        },

        "reference_policy": {
            "relevant_documents_per_query":
                1,

            "reference_context_source":
                "frozen canonical context_text",

            "reference_answer_status":
                "EMPTY_PENDING_MANUAL_CURATION",

            "generated_answers_used":
                False,

            "retrieval_method_outputs_used":
                False,
        },

        "candidate_jsonl": {
            "file":
                str(
                    OUTPUT_JSONL
                ),

            "sha256":
                candidate_hash,
        },
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
    # PRINT
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "REFERENCE CANDIDATE SUMMARY"
    )

    print(
        "=" * 70
    )

    print(
        f"Records           : "
        f"{len(reference_records)}"
    )

    print(
        f"Reference context : "
        f"{len(reference_records)}"
    )

    print(
        "Reference answer  : "
        "0/40 — pending curation"
    )

    print(
        "\nCANDIDATE SHA256:"
    )

    print(
        candidate_hash
    )

    print(
        "\nCandidate JSONL:"
    )

    print(
        OUTPUT_JSONL
    )

    print(
        "\nCandidate CSV:"
    )

    print(
        OUTPUT_CSV
    )

    print(
        "\nMetadata:"
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