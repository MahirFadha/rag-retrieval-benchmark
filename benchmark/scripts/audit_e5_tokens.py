import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

from transformers import AutoTokenizer


# ============================================================
# 1. CONFIG
# ============================================================

MODEL_NAME = "intfloat/multilingual-e5-base"

INPUT_FILE = Path(
    "benchmark/data/canonical/"
    "aire_catalog_raw_canonical_v1.jsonl"
)

OUTPUT_FILE = Path(
    "benchmark/data/canonical/"
    "aire_catalog_token_audit_v1.csv"
)

MODEL_MAX_TOKENS = 512


# ============================================================
# 2. HELPERS
# ============================================================

def format_rupiah(value):
    if value is None:
        return ""

    return "Rp" + format(
        int(value),
        ","
    ).replace(",", ".")


def join_nonempty(parts):
    return " ".join(
        part.strip()
        for part in parts
        if part and part.strip()
    )


# ============================================================
# 3. BUILD FULL CANDIDATE TEXT
# ============================================================

def build_product_text(record):
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
            "Keterangan: "
            f"{record['description']}"
        )

    if record.get("bundling_service_name"):
        parts.append(
            "Jasa Bundling: "
            f"{record['bundling_service_name']}."
        )

    if record.get("bundling_service_price") is not None:
        parts.append(
            "Biaya Jasa Bundling: "
            f"{format_rupiah(record['bundling_service_price'])}."
        )

    if record.get("bundling_service_description"):
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


def build_service_text(record):
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

    if record.get("package_description"):
        parts.append(
            "Deskripsi Kelompok: "
            f"{record['package_description']}"
        )

    if record.get("detail"):
        parts.append(
            "Detail Layanan: "
            f"{record['detail']}"
        )

    return join_nonempty(parts)


def build_full_candidate_text(record):
    if record["type"] == "product":
        return build_product_text(record)

    if record["type"] == "service":
        return build_service_text(record)

    raise ValueError(
        f"Tipe tidak dikenal: {record['type']}"
    )


# ============================================================
# 4. LOAD DATA
# ============================================================

def load_records():
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
# 5. TOKEN COUNT
# ============================================================

def count_tokens(tokenizer, text):
    """
    Hitung panjang input yang benar-benar akan diberikan
    ke E5 sebagai document/passage.

    Tidak melakukan truncation.
    """
    e5_text = f"passage: {text}"

    encoded = tokenizer(
        e5_text,
        add_special_tokens=True,
        truncation=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )

    return len(encoded["input_ids"])


# ============================================================
# 6. SUMMARY
# ============================================================

def percentile(values, p):
    values = sorted(values)

    if not values:
        return 0

    index = (
        (len(values) - 1)
        * p
        / 100
    )

    lower = int(index)
    upper = min(
        lower + 1,
        len(values) - 1,
    )

    fraction = index - lower

    return (
        values[lower]
        + (
            values[upper]
            - values[lower]
        )
        * fraction
    )


def print_group_summary(label, rows):
    tokens = [
        row["token_count"]
        for row in rows
    ]

    over = [
        value
        for value in tokens
        if value > MODEL_MAX_TOKENS
    ]

    print(f"\n=== {label} ===")
    print(f"Jumlah dokumen : {len(tokens)}")
    print(f"Min token      : {min(tokens)}")
    print(f"Median         : {statistics.median(tokens):.1f}")
    print(f"Mean           : {statistics.mean(tokens):.2f}")
    print(f"P90            : {percentile(tokens, 90):.1f}")
    print(f"P95            : {percentile(tokens, 95):.1f}")
    print(f"Max token      : {max(tokens)}")
    print(
        f"> 512 token    : "
        f"{len(over)} "
        f"({len(over) / len(tokens) * 100:.2f}%)"
    )


# ============================================================
# 7. MAIN
# ============================================================

def main():
    print("Memuat tokenizer E5...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    print(
        f"Tokenizer siap: {MODEL_NAME}"
    )

    records = load_records()

    print(
        f"Jumlah record: {len(records)}"
    )

    rows = []

    for record in records:

        full_text = build_full_candidate_text(
            record
        )

        token_count = count_tokens(
            tokenizer,
            full_text,
        )

        rows.append({
            "id": record["id"],
            "type": record["type"],
            "domain": record["domain"],
            "name": record["name"],
            "char_count": len(full_text),
            "token_count": token_count,
            "over_512": (
                token_count
                > MODEL_MAX_TOKENS
            ),
        })

    # ----------------------------------------
    # SAVE CSV
    # ----------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "id",
                "type",
                "domain",
                "name",
                "char_count",
                "token_count",
                "over_512",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    # ----------------------------------------
    # SUMMARY
    # ----------------------------------------

    print_group_summary(
        "SEMUA DOKUMEN",
        rows,
    )

    print_group_summary(
        "PRODUCT AC",
        [
            row
            for row in rows
            if (
                row["type"] == "product"
                and row["domain"] == "AC"
            )
        ],
    )

    print_group_summary(
        "SERVICE AC",
        [
            row
            for row in rows
            if (
                row["type"] == "service"
                and row["domain"] == "AC"
            )
        ],
    )

    print_group_summary(
        "SERVICE CCTV",
        [
            row
            for row in rows
            if (
                row["type"] == "service"
                and row["domain"] == "CCTV"
            )
        ],
    )

    # ----------------------------------------
    # TOP 15 LONGEST
    # ----------------------------------------

    print("\n=== 15 DOKUMEN TOKEN TERPANJANG ===")

    longest = sorted(
        rows,
        key=lambda row: row["token_count"],
        reverse=True,
    )[:15]

    for index, row in enumerate(
        longest,
        start=1,
    ):
        print(
            f"{index:02}. "
            f"{row['id']} | "
            f"{row['token_count']} token | "
            f"{row['type']} | "
            f"{row['domain']} | "
            f"{row['name']}"
        )

    print(
        "\n✅ Audit selesai:"
    )
    print(
        OUTPUT_FILE
    )


if __name__ == "__main__":
    main()