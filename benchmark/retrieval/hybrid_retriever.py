from pathlib import Path
from typing import Dict, List

from benchmark.retrieval.bm25_retriever import (
    BM25Retriever,
)

from benchmark.retrieval.e5_retriever import (
    E5Retriever,
)


# ============================================================
# 1. CONFIG
# ============================================================

DEFAULT_RRF_K = 60


# ============================================================
# 2. HYBRID RRF RETRIEVER
# ============================================================

class HybridRRFRetriever:

    def __init__(
        self,
        corpus_path: Path,
        rrf_k: int = DEFAULT_RRF_K,
    ):

        self.corpus_path = corpus_path
        self.rrf_k = rrf_k

        print(
            "Membangun BM25 untuk Hybrid..."
        )

        self.bm25 = BM25Retriever(
            corpus_path=corpus_path,
        )

        print(
            "Membangun E5 untuk Hybrid..."
        )

        self.e5 = E5Retriever(
            corpus_path=corpus_path,
        )

        # --------------------------------------------
        # Validasi kedua retriever memakai corpus sama
        # --------------------------------------------

        bm25_ids = {
            record["id"]
            for record in self.bm25.records
        }

        e5_ids = {
            record["id"]
            for record in self.e5.records
        }

        if bm25_ids != e5_ids:

            raise RuntimeError(
                "Corpus BM25 dan E5 "
                "tidak identik."
            )

        self.document_count = len(
            bm25_ids
        )

        # Metadata berdasarkan ID.
        self.record_map = {
            record["id"]: record
            for record in self.bm25.records
        }

        print(
            "✅ Hybrid corpus valid: "
            f"{self.document_count} dokumen."
        )


    # ========================================================
    # 3. SEARCH
    # ========================================================

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Dict]:

        # ----------------------------------------------------
        # Gunakan FULL RANKING seluruh corpus.
        # ----------------------------------------------------

        bm25_results = self.bm25.search(
            query=query,
            top_k=self.document_count,
        )

        e5_results = self.e5.search(
            query=query,
            top_k=self.document_count,
        )

        # ----------------------------------------------------
        # Mapping rank per document
        # ----------------------------------------------------

        bm25_rank = {
            result["id"]: result["rank"]
            for result in bm25_results
        }

        e5_rank = {
            result["id"]: result["rank"]
            for result in e5_results
        }

        # ----------------------------------------------------
        # Reciprocal Rank Fusion
        #
        # Equal weight:
        #
        # 1/(K + BM25 rank)
        # +
        # 1/(K + E5 rank)
        # ----------------------------------------------------

        fused_results = []

        for document_id in self.record_map:

            rank_bm25 = bm25_rank[
                document_id
            ]

            rank_e5 = e5_rank[
                document_id
            ]

            bm25_rrf = (
                1.0
                / (
                    self.rrf_k
                    + rank_bm25
                )
            )

            e5_rrf = (
                1.0
                / (
                    self.rrf_k
                    + rank_e5
                )
            )

            rrf_score = (
                bm25_rrf
                + e5_rrf
            )

            record = self.record_map[
                document_id
            ]

            fused_results.append({
                "id":
                    document_id,

                "type":
                    record["type"],

                "domain":
                    record["domain"],

                "name":
                    record["name"],

                "bm25_rank":
                    rank_bm25,

                "e5_rank":
                    rank_e5,

                "bm25_rrf":
                    bm25_rrf,

                "e5_rrf":
                    e5_rrf,

                "rrf_score":
                    rrf_score,
            })

        # ----------------------------------------------------
        # Deterministic sorting
        #
        # 1. RRF score descending
        # 2. ID ascending untuk tie
        # ----------------------------------------------------

        fused_results.sort(
            key=lambda result: (
                -result["rrf_score"],
                result["id"],
            )
        )

        top_results = (
            fused_results[
                :top_k
            ]
        )

        # Tambahkan final rank.
        for rank, result in enumerate(
            top_results,
            start=1,
        ):
            result["rank"] = rank

        return top_results