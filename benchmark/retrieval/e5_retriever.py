import json
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F

from transformers import (
    AutoModel,
    AutoTokenizer,
)


# ============================================================
# 1. CONFIG
# ============================================================

DEFAULT_MODEL_NAME = (
    "intfloat/multilingual-e5-base"
)

MODEL_MAX_TOKENS = 512

DEFAULT_BATCH_SIZE = 16


# ============================================================
# 2. LOAD CORPUS
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
                    "Record tidak memiliki "
                    "index_text: "
                    f"{record.get('id')}"
                )

            records.append(
                record
            )

    return records


# ============================================================
# 3. MEAN POOLING
# ============================================================

def average_pool(
    last_hidden_state: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Mean pooling untuk multilingual-E5.

    Shape:
    last_hidden_state:
        [batch_size, sequence_length, hidden_size]

    attention_mask:
        [batch_size, sequence_length]

    Attention mask diperluas menjadi:
        [batch_size, sequence_length, 1]

    sehingga bisa di-broadcast ke hidden dimension.
    """

    # --------------------------------------------------------
    # Expand mask:
    #
    # [batch, sequence]
    # ->
    # [batch, sequence, 1]
    # --------------------------------------------------------

    mask = attention_mask[
        ...,
        None
    ].bool()

    # --------------------------------------------------------
    # Hilangkan kontribusi padding token
    # --------------------------------------------------------

    masked_hidden_state = (
        last_hidden_state.masked_fill(
            ~mask,
            0.0,
        )
    )

    # --------------------------------------------------------
    # Jumlahkan embedding seluruh token aktif
    # --------------------------------------------------------

    summed_embeddings = (
        masked_hidden_state.sum(
            dim=1
        )
    )

    # --------------------------------------------------------
    # Hitung jumlah token aktif
    #
    # [batch, sequence]
    # ->
    # [batch, 1]
    # --------------------------------------------------------

    token_counts = (
        attention_mask
        .sum(dim=1)[
            ...,
            None
        ]
        .clamp(min=1)
    )

    # --------------------------------------------------------
    # Mean pooling
    # --------------------------------------------------------

    embeddings = (
        summed_embeddings
        / token_counts
    )

    return embeddings


# ============================================================
# 4. E5 RETRIEVER
# ============================================================

class E5Retriever:

    def __init__(
        self,
        corpus_path: Path,
        model_name: str = DEFAULT_MODEL_NAME,
        batch_size: int = DEFAULT_BATCH_SIZE,
        device: str = None,
    ):

        self.corpus_path = corpus_path
        self.model_name = model_name
        self.batch_size = batch_size

        # ----------------------------------------------------
        # DEVICE
        # ----------------------------------------------------

        if device is None:

            if torch.cuda.is_available():
                self.device = torch.device(
                    "cuda"
                )

            else:
                self.device = torch.device(
                    "cpu"
                )

        else:
            self.device = torch.device(
                device
            )

        # ----------------------------------------------------
        # LOAD DATA
        # ----------------------------------------------------

        self.records = load_corpus(
            corpus_path
        )

        # ----------------------------------------------------
        # TOKENIZER + MODEL
        # ----------------------------------------------------

        print(
            f"Memuat tokenizer: "
            f"{self.model_name}"
        )

        self.tokenizer = (
            AutoTokenizer
            .from_pretrained(
                self.model_name
            )
        )

        print(
            f"Memuat model: "
            f"{self.model_name}"
        )

        self.model = (
            AutoModel
            .from_pretrained(
                self.model_name
            )
        )

        self.model.to(
            self.device
        )

        self.model.eval()

        print(
            f"Device: {self.device}"
        )

        # ----------------------------------------------------
        # VALIDATE CORPUS LENGTH
        # ----------------------------------------------------

        self._validate_corpus_length()

        # ----------------------------------------------------
        # BUILD CORPUS EMBEDDINGS
        # ----------------------------------------------------

        print(
            "Membangun embedding corpus..."
        )

        self.corpus_embeddings = (
            self._build_corpus_embeddings()
        )

        print(
            "Embedding corpus siap."
        )

        print(
            "Shape: "
            f"{tuple(self.corpus_embeddings.shape)}"
        )


    # ========================================================
    # 5. VALIDATE TOKEN LENGTH
    # ========================================================

    def _validate_corpus_length(
        self,
    ):
        """
        Memastikan tidak ada dokumen yang
        diam-diam terkena truncation.

        Audit sebelumnya seharusnya sudah
        memastikan seluruh corpus <= 512.
        """

        over_limit = []

        for record in self.records:

            text = (
                "passage: "
                + record["index_text"]
            )

            encoded = self.tokenizer(
                text,
                add_special_tokens=True,
                truncation=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )

            token_count = len(
                encoded["input_ids"]
            )

            if (
                token_count
                > MODEL_MAX_TOKENS
            ):
                over_limit.append(
                    (
                        record["id"],
                        token_count,
                    )
                )

        if over_limit:

            message = [
                "Ditemukan corpus >512 token:"
            ]

            for (
                record_id,
                token_count,
            ) in over_limit:

                message.append(
                    f"- {record_id}: "
                    f"{token_count}"
                )

            raise RuntimeError(
                "\n".join(
                    message
                )
            )

        print(
            "✅ Semua corpus <= "
            f"{MODEL_MAX_TOKENS} token."
        )


    # ========================================================
    # 6. EMBED TEXT BATCH
    # ========================================================

    def _encode(
        self,
        texts: List[str],
    ) -> torch.Tensor:
        """
        Menghasilkan L2-normalized E5 embedding.
        """

        encoded = self.tokenizer(
            texts,
            max_length=MODEL_MAX_TOKENS,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        encoded = {
            key: value.to(
                self.device
            )
            for key, value
            in encoded.items()
        }

        with torch.inference_mode():

            outputs = self.model(
                **encoded
            )

            embeddings = average_pool(
                outputs.last_hidden_state,
                encoded[
                    "attention_mask"
                ],
            )

            # ------------------------------------------------
            # NORMALIZE
            #
            # Setelah normalization:
            # dot product = cosine similarity
            # ------------------------------------------------

            embeddings = F.normalize(
                embeddings,
                p=2,
                dim=1,
            )

        return embeddings


    # ========================================================
    # 7. BUILD CORPUS EMBEDDINGS
    # ========================================================

    def _build_corpus_embeddings(
        self,
    ) -> torch.Tensor:

        passages = [
            (
                "passage: "
                + record[
                    "index_text"
                ]
            )
            for record in self.records
        ]

        all_embeddings = []

        for start in range(
            0,
            len(passages),
            self.batch_size,
        ):

            end = (
                start
                + self.batch_size
            )

            batch = passages[
                start:end
            ]

            embeddings = (
                self._encode(
                    batch
                )
            )

            # Simpan di CPU.
            #
            # Corpus hanya 145 dokumen,
            # sehingga matrix exact sangat ringan.
            all_embeddings.append(
                embeddings.cpu()
            )

        return torch.cat(
            all_embeddings,
            dim=0,
        )


    # ========================================================
    # 8. QUERY EMBEDDING
    # ========================================================

    def encode_query(
        self,
        query: str,
    ) -> torch.Tensor:

        text = (
            "query: "
            + query.strip()
        )

        embedding = self._encode(
            [text]
        )

        return embedding.cpu()


    # ========================================================
    # 9. SEARCH
    # ========================================================

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Dict]:

        query_embedding = (
            self.encode_query(
                query
            )
        )

        # ----------------------------------------------------
        # Exact cosine similarity.
        #
        # Kedua embedding sudah L2-normalized,
        # sehingga matrix multiplication = cosine.
        # ----------------------------------------------------

        scores = torch.matmul(
            query_embedding,
            self.corpus_embeddings.T,
        )[0]

        # ----------------------------------------------------
        # Deterministic ranking:
        #
        # 1. cosine descending
        # 2. ID ascending jika score sama
        # ----------------------------------------------------

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