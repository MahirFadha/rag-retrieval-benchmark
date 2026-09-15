import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, List

from rank_bm25 import BM25Okapi


# ============================================================
# 1. CONFIG
# ============================================================

DEFAULT_K1 = 1.5
DEFAULT_B = 0.75
DEFAULT_EPSILON = 0.25


# ============================================================
# 2. BM25 TOKENIZER
# ============================================================

def normalize_bm25_text(
    text: str,
) -> str:
    """
    Normalisasi ringan dan deterministik.

    Tidak melakukan:
    - stemming
    - stopword removal
    - synonym expansion
    - semantic enrichment
    """

    if not text:
        return ""

    # Unicode normalization
    text = unicodedata.normalize(
        "NFKC",
        text,
    )

    text = text.lower()

    # Samakan berbagai jenis dash.
    text = text.replace(
        "–",
        "-",
    )

    text = text.replace(
        "—",
        "-",
    )

    # Normalisasi desimal PK:
    # 0,5 -> 0.5
    text = re.sub(
        r"(?<=\d),(?=\d)",
        ".",
        text,
    )

    # Hyphen diperlakukan sebagai separator.
    #
    # Contoh:
    # 1.5-2 PK
    # ->
    # 1.5 2 PK
    #
    # R-32
    # ->
    # R 32
    text = text.replace(
        "-",
        " ",
    )

    # Slash juga separator.
    text = text.replace(
        "/",
        " ",
    )

    return text


def tokenize_bm25(
    text: str,
) -> List[str]:
    """
    Tokenizer sederhana untuk baseline lexical.

    Contoh:

    "AC R-32 1.5-2 PK"

    menjadi:

    [
        "ac",
        "r",
        "32",
        "1.5",
        "2",
        "pk"
    ]
    """

    text = normalize_bm25_text(
        text
    )

    tokens = re.findall(
        r"[a-z0-9]+(?:\.[a-z0-9]+)*",
        text,
    )

    return tokens


# ============================================================
# 3. LOAD CORPUS
# ============================================================

def load_corpus(
    corpus_path: Path,
) -> List[Dict]:

    records = []

    with corpus_path.open(
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

            if not record.get(
                "index_text"
            ):
                raise ValueError(
                    "Record tidak mempunyai "
                    "index_text: "
                    f"{record.get('id')}"
                )

            records.append(
                record
            )

    return records


# ============================================================
# 4. BM25 RETRIEVER
# ============================================================

class BM25Retriever:

    def __init__(
        self,
        corpus_path: Path,
        k1: float = DEFAULT_K1,
        b: float = DEFAULT_B,
        epsilon: float = DEFAULT_EPSILON,
    ):

        self.corpus_path = corpus_path

        self.k1 = k1
        self.b = b
        self.epsilon = epsilon

        self.records = load_corpus(
            corpus_path
        )

        # ----------------------------------------------------
        # PENTING:
        # Hanya index_text yang dipakai.
        # ----------------------------------------------------

        self.tokenized_corpus = [
            tokenize_bm25(
                record["index_text"]
            )
            for record in self.records
        ]

        self.model = BM25Okapi(
            self.tokenized_corpus,
            k1=self.k1,
            b=self.b,
            epsilon=self.epsilon,
        )

    # ========================================================
    # SEARCH
    # ========================================================

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Dict]:

        query_tokens = tokenize_bm25(
            query
        )

        scores = self.model.get_scores(
            query_tokens
        )

        # Deterministic ranking.
        #
        # Primary:
        # score descending
        #
        # Tie breaker:
        # ID ascending
        ranked_indices = sorted(
            range(
                len(
                    self.records
                )
            ),
            key=lambda index: (
                -float(
                    scores[index]
                ),
                self.records[
                    index
                ]["id"],
            ),
        )

        top_indices = ranked_indices[
            :top_k
        ]

        results = []

        for rank, index in enumerate(
            top_indices,
            start=1,
        ):

            record = self.records[
                index
            ]

            results.append({
                "rank":
                    rank,

                "id":
                    record["id"],

                "type":
                    record["type"],

                "domain":
                    record["domain"],

                "name":
                    record["name"],

                "score":
                    float(
                        scores[index]
                    ),
            })

        return results