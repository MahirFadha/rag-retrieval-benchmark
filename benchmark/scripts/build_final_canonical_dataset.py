import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

from transformers import AutoTokenizer


# ============================================================
# 1. CONFIG
# ============================================================

MODEL_NAME = "intfloat/multilingual-e5-base"

INPUT_FILE = Path(
    "benchmark/data/canonical/"
    "aire_catalog_raw_canonical_v1.jsonl"
)

OUTPUT_JSONL = Path(
    "benchmark/data/canonical/"
    "aire_catalog_final_v1.jsonl"
)

OUTPUT_AUDIT = Path(
    "benchmark/data/canonical/"
    "aire_catalog_final_audit_v1.csv"
)

EXPECTED_TOTAL = 145

MODEL_MAX_TOKENS = 512

# ============================================================
# DATA QUALITY EXCLUSIONS
# ============================================================

DETAIL_EXCLUSIONS = {
    "SRVC003": (
        "Konflik kapasitas: nama layanan 0.5-2 PK, "
        "sedangkan service_detail menyebut 0.5-1 PK."
    ),
    "SRVC004": (
        "Konflik kapasitas: nama layanan 3-5 PK, "
        "sedangkan service_detail menyebut 1.5-2 PK."
    ),
    "SRVC015": (
        "Konflik jenis layanan dan kapasitas: "
        "service_items menyatakan penambahan Freon R-22 "
        "1.5-2 PK, sedangkan service_detail menjelaskan "
        "pengisian penuh 0.5-1 PK."
    ),
    "SRVC074": (
        "Konflik kapasitas: nama layanan pemasangan Indoor "
        "1.5-2 PK, sedangkan bagian indikator service_detail "
        "menyebut 0.5-1 PK."
    ),
}


# ============================================================
# 2. GENERAL HELPERS
# ============================================================

def normalize_whitespace(text: str) -> str:
    if not text:
        return ""

    return " ".join(str(text).split())


def clean_derived_text(text: str) -> str:
    """
    Cleaning ringan untuk text hasil derivasi.

    Tidak mengubah makna.
    Hanya membersihkan artefak punctuation/whitespace.
    """

    if not text:
        return ""

    text = normalize_whitespace(text)

    # Contoh:
    # "Kebutuhan Cuci:." -> "Kebutuhan Cuci:"
    # "penting?." -> "penting?"
    text = re.sub(
        r"([:!?])\s*\.\s*",
        r"\1 ",
        text,
    )

    # Titik kosong di awal.
    text = re.sub(
        r"^\s*\.\s*",
        "",
        text,
    )

    # "Pengerjaan . Durasi" -> "Pengerjaan. Durasi"
    text = re.sub(
        r"\s+\.\s+",
        ". ",
        text,
    )

    # Spasi sebelum titik.
    text = re.sub(
        r"\s+\.",
        ".",
        text,
    )

    # Titik berulang.
    text = re.sub(
        r"\.{2,}",
        ".",
        text,
    )

    return normalize_whitespace(text).strip()


def join_nonempty(parts: List[str]) -> str:
    return " ".join(
        clean_derived_text(part)
        for part in parts
        if part and str(part).strip()
    )


def format_rupiah(value) -> str:
    if value is None:
        return ""

    try:
        return "Rp" + format(
            int(value),
            ",",
        ).replace(",", ".")

    except (ValueError, TypeError):
        return str(value)

def get_usable_service_detail(record: Dict) -> str:
    """
    Mengambil service detail yang layak digunakan
    dalam index_text/context_text.

    Detail yang telah terverifikasi bertentangan
    dengan field utama service_items tidak digunakan.
    """

    record_id = record.get("id", "")

    if record_id in DETAIL_EXCLUSIONS:
        return ""

    return record.get("detail", "") or ""

# ============================================================
# 3. LOAD RAW CANONICAL DATA
# ============================================================

def load_records() -> List[Dict]:
    records = []

    with INPUT_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:
            line = line.strip()

            if not line:
                continue

            records.append(
                json.loads(line)
            )

    return records


# ============================================================
# 4. PRODUCT TEXT
# ============================================================

def build_product_text(record: Dict) -> str:
    """
    Untuk produk:
    index_text = context_text.

    Semua produk sebelumnya sudah terbukti
    berada jauh di bawah 512 token.
    """

    parts = [
        f"Nama Produk: {record['name']}.",
        f"Kategori: {record['domain']}.",
    ]

    if record.get("brand"):
        parts.append(
            f"Merek: {record['brand']}."
        )

    if record.get("price") is not None:
        parts.append(
            "Harga Unit: "
            f"{format_rupiah(record['price'])}."
        )

    if record.get("description"):
        parts.append(
            "Keterangan Produk: "
            f"{record['description']}"
        )

    if record.get("bundling_service_name"):
        parts.append(
            "Jasa Bundling: "
            f"{record['bundling_service_name']}."
        )

    if (
        record.get("bundling_service_price")
        is not None
    ):
        parts.append(
            "Biaya Jasa Bundling: "
            f"{format_rupiah(record['bundling_service_price'])}."
        )

    if record.get(
        "bundling_service_description"
    ):
        parts.append(
            "Deskripsi Jasa Bundling: "
            f"{record['bundling_service_description']}"
        )

    if record.get("detail"):
        parts.append(
            "Detail Produk: "
            f"{record['detail']}"
        )

    return join_nonempty(parts)


# ============================================================
# 5. AC SERVICE SECTION PARSER
# ============================================================

SECTION_PATTERNS = [
    (
        "process",
        re.compile(
            r"\bProses\s+Pengerjaan\b",
            flags=re.IGNORECASE,
        ),
    ),
    (
        "duration",
        re.compile(
            r"\bDurasi(?:\s+Pengerjaan)?\b",
            flags=re.IGNORECASE,
        ),
    ),
    (
        "benefit",
        re.compile(
            r"\bManfaat"
            r"(?:\s+Pengerjaan\s+Profesional)?\b",
            flags=re.IGNORECASE,
        ),
    ),
    (
        "why",
        re.compile(
            r"\bKenapa\s+"
            r"(?:Service|Pelayanan)"
            r"\s+(?:Ini\s+)?Penting\b",
            flags=re.IGNORECASE,
        ),
    ),
    (
        "indicator",
        re.compile(
            r"\bIndikator\b",
            flags=re.IGNORECASE,
        ),
    ),
]


KEEP_AC_SECTIONS = {
    "duration",
    "benefit",
    "why",
    "indicator",
}


def find_ac_sections(
    detail: str,
) -> List[Tuple[int, str]]:
    """
    Mencari posisi heading section dalam detail jasa AC.
    """

    markers = []

    for section_name, pattern in SECTION_PATTERNS:
        for match in pattern.finditer(detail):
            markers.append(
                (
                    match.start(),
                    section_name,
                )
            )

    markers.sort(
        key=lambda item: item[0]
    )

    # Hindari marker duplicate pada posisi sama.
    unique_markers = []

    used_positions = set()

    for position, section_name in markers:

        if position in used_positions:
            continue

        used_positions.add(position)

        unique_markers.append(
            (
                position,
                section_name,
            )
        )

    return unique_markers


def build_ac_retrieval_detail(
    detail: str,
) -> Tuple[str, List[str]]:
    """
    Membuat detail retrieval untuk jasa AC.

    Prinsip:
    - Proses pengerjaan teknis TIDAK dimasukkan.
    - Durasi, manfaat, alasan penting,
      dan indikator kebutuhan dipertahankan.
    - Jika struktur heading tidak dikenali,
      detail lengkap dipertahankan sebagai fallback.

    Tidak ada LLM dan tidak ada summarization.
    """

    detail = clean_derived_text(detail)

    if not detail:
        return "", []

    markers = find_ac_sections(detail)

    # Tidak punya struktur yang dikenali:
    # jangan buang data.
    if not markers:
        return detail, []

    kept_sections = []
    kept_names = []

    for index, (
        start_position,
        section_name,
    ) in enumerate(markers):

        # Tentukan akhir section berdasarkan
        # heading berikutnya.
        if index + 1 < len(markers):
            end_position = markers[
                index + 1
            ][0]
        else:
            end_position = len(detail)

        section_text = detail[
            start_position:end_position
        ].strip()

        if (
            section_name
            in KEEP_AC_SECTIONS
            and section_text
        ):
            kept_sections.append(
                section_text
            )

            kept_names.append(
                section_name
            )

    # Jika parser menemukan heading,
    # tetapi tidak menemukan section retrieval-relevant,
    # gunakan detail penuh agar tidak kehilangan informasi.
    if not kept_sections:
        return detail, []

    return (
        join_nonempty(kept_sections),
        kept_names,
    )


# ============================================================
# 6. CCTV DETAIL CLEANING
# ============================================================

CCTV_CTA_PATTERNS = [
    re.compile(
        r"\bSiap\s+Mengamankan\s+Properti\s+Anda\b",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\bHubungi\s+kami\s+untuk\s+konsultasi\b",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\bWhatsApp\s*:",
        flags=re.IGNORECASE,
    ),
]


def remove_cctv_cta(
    detail: str,
) -> Tuple[str, bool]:
    """
    Menghapus bagian CTA/kontak dari detail CCTV.

    Informasi teknis:
    - spesifikasi
    - fitur
    - isi paket
    - garansi
    - ketentuan instalasi

    tetap dipertahankan.
    """

    detail = clean_derived_text(detail)

    if not detail:
        return "", False

    positions = []

    for pattern in CCTV_CTA_PATTERNS:
        match = pattern.search(detail)

        if match:
            positions.append(
                match.start()
            )

    if not positions:
        return detail, False

    cut_position = min(positions)

    cleaned = detail[
        :cut_position
    ].strip()

    return cleaned, True


# ============================================================
# 7. SERVICE BASE TEXT
# ============================================================

def build_service_base_parts(
    record: Dict,
) -> List[str]:

    parts = [
        f"Nama Layanan: {record['name']}.",
        f"Kategori: {record['domain']}.",
    ]

    if record.get("package_name"):
        parts.append(
            "Kelompok Layanan: "
            f"{record['package_name']}."
        )

    if record.get("price") is not None:
        parts.append(
            "Harga: "
            f"{format_rupiah(record['price'])}."
        )

    if record.get("description"):
        parts.append(
            "Deskripsi Layanan: "
            f"{record['description']}"
        )

    return parts


# ============================================================
# 8. SERVICE AC INDEX
# ============================================================

def build_service_ac_index(
    record: Dict,
) -> Tuple[str, List[str]]:

    parts = build_service_base_parts(
        record
    )

    usable_detail = get_usable_service_detail(
    record
    )

    retrieval_detail, kept_sections = (
        build_ac_retrieval_detail(
            usable_detail
        )
    )

    if retrieval_detail:
        parts.append(
            "Informasi Relevan Layanan: "
            f"{retrieval_detail}"
        )

    return (
        join_nonempty(parts),
        kept_sections,
    )


# ============================================================
# 9. SERVICE CCTV INDEX
# ============================================================

def build_service_cctv_index(
    record: Dict,
) -> Tuple[str, bool]:

    parts = build_service_base_parts(
        record
    )

    cctv_detail, cta_removed = (
        remove_cctv_cta(
            record.get(
                "detail",
                "",
            )
        )
    )

    if cctv_detail:
        parts.append(
            "Detail Layanan: "
            f"{cctv_detail}"
        )

    # package_description sengaja TIDAK
    # dimasukkan ke index_text.
    #
    # Alasannya:
    # - bersifat berulang pada setiap package,
    # - informasi utamanya sudah tercermin
    #   oleh package_name + description + detail,
    # - membantu menjaga panjang di bawah
    #   kapasitas E5.

    return (
        join_nonempty(parts),
        cta_removed,
    )


# ============================================================
# 10. SERVICE CONTEXT TEXT
# ============================================================

def build_service_context(
    record: Dict,
) -> str:
    """
    Context untuk LLM dibuat lebih lengkap
    dibanding index_text.

    Untuk AC:
    proses pengerjaan lengkap tetap tersedia.

    Untuk CCTV:
    CTA/kontak dibuang karena bukan informasi
    substantif mengenai paket.
    """

    parts = build_service_base_parts(
        record
    )

    if record.get("package_description"):
        parts.append(
            "Deskripsi Kelompok Layanan: "
            f"{record['package_description']}"
        )

    detail = get_usable_service_detail(
        record
    )

    if (
        record.get("domain", "").upper()
        == "CCTV"
    ):
        detail, _ = remove_cctv_cta(
            detail
        )
    else:
        detail = clean_derived_text(
            detail
        )

    if detail:
        parts.append(
            "Detail Layanan Lengkap: "
            f"{detail}"
        )

    return join_nonempty(parts)


# ============================================================
# 11. BUILD FINAL REPRESENTATION
# ============================================================

def build_final_record(
    record: Dict,
) -> Dict:

    final_record = dict(record)

    record_id = record.get("id", "")

    if record_id in DETAIL_EXCLUSIONS:
        final_record["detail_quality"] = (
            "excluded_conflict"
        )

        final_record["detail_exclusion_reason"] = (
            DETAIL_EXCLUSIONS[record_id]
        )
    else:
        final_record["detail_quality"] = "accepted"

        final_record["detail_exclusion_reason"] = ""

    record_type = (
        record.get(
            "type",
            "",
        )
        .strip()
        .lower()
    )

    domain = (
        record.get(
            "domain",
            "",
        )
        .strip()
        .upper()
    )

    # --------------------------------------------------------
    # PRODUCT
    # --------------------------------------------------------

    if record_type == "product":

        index_text = build_product_text(
            record
        )

        context_text = index_text

        strategy = (
            "product_full"
        )

        kept_sections = []
        cta_removed = False

    # --------------------------------------------------------
    # SERVICE AC
    # --------------------------------------------------------

    elif (
        record_type == "service"
        and domain == "AC"
    ):

        (
            index_text,
            kept_sections,
        ) = build_service_ac_index(
            record
        )

        context_text = (
            build_service_context(
                record
            )
        )

        if record["id"] in DETAIL_EXCLUSIONS:
            strategy = (
                "service_ac_core_only_conflict_excluded"
            )
        else:
            strategy = (
                "service_ac_section_aware"
            )

        cta_removed = False

    # --------------------------------------------------------
    # SERVICE CCTV
    # --------------------------------------------------------

    elif (
        record_type == "service"
        and domain == "CCTV"
    ):

        (
            index_text,
            cta_removed,
        ) = build_service_cctv_index(
            record
        )

        context_text = (
            build_service_context(
                record
            )
        )

        strategy = (
            "service_cctv_trimmed"
        )

        kept_sections = []

    else:
        raise ValueError(
            "Record tidak dikenali: "
            f"{record.get('id')} | "
            f"type={record_type} | "
            f"domain={domain}"
        )

    final_record[
        "index_text"
    ] = index_text

    final_record[
        "context_text"
    ] = context_text

    final_record[
        "index_strategy"
    ] = strategy

    final_record[
        "_kept_sections"
    ] = kept_sections

    final_record[
        "_cctv_cta_removed"
    ] = cta_removed

    return final_record


# ============================================================
# 12. E5 TOKEN COUNT
# ============================================================

def count_index_tokens(
    tokenizer,
    index_text: str,
) -> int:
    """
    Menghitung panjang sebenarnya yang akan
    masuk ke multilingual-E5-base.

    E5 document prefix ikut dihitung.
    """

    e5_document = (
        f"passage: {index_text}"
    )

    encoded = tokenizer(
        e5_document,
        add_special_tokens=True,
        truncation=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )

    return len(
        encoded["input_ids"]
    )


# ============================================================
# 13. VALIDATION
# ============================================================

def validate_base_records(
    records: List[Dict],
):
    if len(records) != EXPECTED_TOTAL:
        raise ValueError(
            "Jumlah raw record tidak sesuai. "
            f"Expected={EXPECTED_TOTAL}, "
            f"Actual={len(records)}"
        )

    ids = [
        record.get("id", "")
        for record in records
    ]

    if len(ids) != len(set(ids)):
        raise ValueError(
            "Ditemukan duplicate ID."
        )

    for record in records:

        if not record.get("id"):
            raise ValueError(
                "Ada record tanpa ID."
            )

        if not record.get("name"):
            raise ValueError(
                "Ada record tanpa nama: "
                f"{record.get('id')}"
            )


# ============================================================
# 14. AUDIT CSV
# ============================================================

def write_audit_csv(
    records: List[Dict],
):
    OUTPUT_AUDIT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fields = [
        "id",
        "type",
        "domain",
        "name",
        "index_strategy",
        "detail_quality",
        "detail_exclusion_reason",
        "index_chars",
        "context_chars",
        "index_token_count",
        "over_512",
        "kept_ac_sections",
        "cctv_cta_removed",
    ]

    with OUTPUT_AUDIT.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fields,
        )

        writer.writeheader()

        for record in records:

            writer.writerow({
                "id":
                    record["id"],

                "type":
                    record["type"],

                "domain":
                    record["domain"],

                "name":
                    record["name"],

                "index_strategy":
                    record[
                        "index_strategy"
                    ],

                "detail_quality":
                    record.get(
                        "detail_quality",
                        "accepted",
                    ),

                "detail_exclusion_reason":
                    record.get(
                        "detail_exclusion_reason",
                        "",
                    ),

                "index_chars":
                    len(
                        record[
                            "index_text"
                        ]
                    ),

                "context_chars":
                    len(
                        record[
                            "context_text"
                        ]
                    ),

                "index_token_count":
                    record[
                        "index_token_count"
                    ],

                "over_512":
                    (
                        record[
                            "index_token_count"
                        ]
                        > MODEL_MAX_TOKENS
                    ),

                "kept_ac_sections":
                    ",".join(
                        record.get(
                            "_kept_sections",
                            [],
                        )
                    ),

                "cctv_cta_removed":
                    record.get(
                        "_cctv_cta_removed",
                        False,
                    ),
            })


# ============================================================
# 15. WRITE FINAL JSONL
# ============================================================

def write_final_jsonl(
    records: List[Dict],
):
    OUTPUT_JSONL.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_JSONL.open(
        "w",
        encoding="utf-8",
    ) as file:

        for record in records:

            # Internal audit/helper fields
            # tidak perlu berada di final dataset.
            clean_record = dict(record)

            clean_record.pop(
                "_kept_sections",
                None,
            )

            clean_record.pop(
                "_cctv_cta_removed",
                None,
            )

            file.write(
                json.dumps(
                    clean_record,
                    ensure_ascii=False,
                )
                + "\n"
            )


# ============================================================
# 16. SHA256
# ============================================================

def sha256_file(
    path: Path,
) -> str:

    hasher = hashlib.sha256()

    with path.open(
        "rb"
    ) as file:

        while True:

            block = file.read(
                1024 * 1024
            )

            if not block:
                break

            hasher.update(block)

    return hasher.hexdigest()


# ============================================================
# 17. SUMMARY
# ============================================================

def print_summary(
    records: List[Dict],
):
    print(
        "\n=== FINAL CANONICAL SUMMARY ==="
    )

    print(
        f"Total dokumen : "
        f"{len(records)}"
    )

    groups = {}

    for record in records:

        key = (
            record["type"],
            record["domain"],
            record["index_strategy"],
        )

        groups[key] = (
            groups.get(
                key,
                0,
            )
            + 1
        )

    for key, count in sorted(
        groups.items()
    ):
        print(
            f"{key[0]:8} | "
            f"{key[1]:5} | "
            f"{key[2]:30} | "
            f"{count}"
        )

    token_counts = [
        record[
            "index_token_count"
        ]
        for record in records
    ]

    print(
        "\n=== TOKEN INDEX ==="
    )

    print(
        f"Min token  : "
        f"{min(token_counts)}"
    )

    print(
        f"Max token  : "
        f"{max(token_counts)}"
    )

    print(
        f"Mean token : "
        f"{sum(token_counts) / len(token_counts):.2f}"
    )

    over_512 = [
        record
        for record in records
        if (
            record[
                "index_token_count"
            ]
            > MODEL_MAX_TOKENS
        )
    ]

    print(
        f">512 token : "
        f"{len(over_512)}"
    )

    if over_512:

        print(
            "\n❌ DOKUMEN MASIH "
            "MELEBIHI 512 TOKEN:"
        )

        for record in sorted(
            over_512,
            key=lambda item: (
                item[
                    "index_token_count"
                ]
            ),
            reverse=True,
        ):
            print(
                f"- {record['id']} | "
                f"{record['index_token_count']} | "
                f"{record['name']}"
            )

    # ----------------------------------------
    # Top longest
    # ----------------------------------------

    print(
        "\n=== 15 INDEX TERPANJANG ==="
    )

    longest = sorted(
        records,
        key=lambda item: (
            item[
                "index_token_count"
            ]
        ),
        reverse=True,
    )[:15]

    for number, record in enumerate(
        longest,
        start=1,
    ):
        print(
            f"{number:02}. "
            f"{record['id']} | "
            f"{record['index_token_count']} token | "
            f"{record['type']} | "
            f"{record['domain']} | "
            f"{record['name']}"
        )


# ============================================================
# 18. MAIN
# ============================================================

def main():
    print(
        "Membangun Final Canonical "
        "Dataset Aire Optima..."
    )

    raw_records = load_records()

    validate_base_records(
        raw_records
    )

    print(
        f"Raw records: "
        f"{len(raw_records)}"
    )

    print(
        "Memuat tokenizer "
        f"{MODEL_NAME}..."
    )

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            MODEL_NAME
        )
    )

    final_records = []

    for raw_record in raw_records:

        final_record = (
            build_final_record(
                raw_record
            )
        )

        token_count = (
            count_index_tokens(
                tokenizer,
                final_record[
                    "index_text"
                ],
            )
        )

        final_record[
            "index_token_count"
        ] = token_count

        final_records.append(
            final_record
        )

    final_records = sorted(
        final_records,
        key=lambda record: (
            record["id"]
        ),
    )

    # Audit selalu ditulis,
    # bahkan jika ada record >512,
    # agar mudah diperiksa.
    write_audit_csv(
        final_records
    )

    print_summary(
        final_records
    )

    # --------------------------------------------------------
    # HARD FAIL:
    # jangan diam-diam truncate.
    # --------------------------------------------------------

    over_limit = [
        record
        for record in final_records
        if (
            record[
                "index_token_count"
            ]
            > MODEL_MAX_TOKENS
        )
    ]

    if over_limit:
        print(
            "\n❌ FINAL DATASET BELUM "
            "DITULIS."
        )

        print(
            "Masih ada index_text "
            "di atas 512 token."
        )

        print(
            f"Lihat audit CSV: "
            f"{OUTPUT_AUDIT}"
        )

        raise RuntimeError(
            "Final canonical dataset "
            "belum memenuhi batas E5."
        )

    # --------------------------------------------------------
    # Semua sudah <=512
    # --------------------------------------------------------

    write_final_jsonl(
        final_records
    )

    checksum = sha256_file(
        OUTPUT_JSONL
    )

    print(
        "\n✅ FINAL CANONICAL DATASET "
        "BERHASIL DIBUAT"
    )

    print(
        f"JSONL : {OUTPUT_JSONL}"
    )

    print(
        f"Audit : {OUTPUT_AUDIT}"
    )

    print(
        f"SHA256: {checksum}"
    )


if __name__ == "__main__":
    main()