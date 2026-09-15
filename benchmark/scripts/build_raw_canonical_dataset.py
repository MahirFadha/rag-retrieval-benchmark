import csv
import html
import json
import re
from pathlib import Path
from typing import Any, Dict, List

from psycopg2.extras import RealDictCursor

from data.database import get_db_connection


# ============================================================
# 1. KONFIGURASI
# ============================================================

OUTPUT_DIR = Path("benchmark/data/canonical")

JSONL_OUTPUT = OUTPUT_DIR / "aire_catalog_raw_canonical_v1.jsonl"
CSV_AUDIT_OUTPUT = OUTPUT_DIR / "aire_catalog_raw_audit_v1.csv"

EXPECTED_PRODUCTS = 53
EXPECTED_SERVICES = 92
EXPECTED_TOTAL = 145


# ============================================================
# 2. TEXT CLEANING
# ============================================================

def normalize_whitespace(text: str) -> str:
    """
    Mengubah whitespace berlebih menjadi satu spasi.
    """
    if not text:
        return ""

    return " ".join(text.split())


def normalize_pk_decimal(text: str) -> str:
    """
    Normalisasi penulisan PK:
    0,5 PK -> 0.5 PK
    1,5 PK -> 1.5 PK

    Hanya mengubah angka desimal yang secara langsung
    diikuti satuan PK.
    """
    if not text:
        return ""

    pattern = r"(\d+),(\d+)(\s*PK\b)"

    return re.sub(
        pattern,
        lambda m: f"{m.group(1)}.{m.group(2)}{m.group(3)}",
        text,
        flags=re.IGNORECASE,
    )


def clean_html_text(raw_html: str) -> str:
    """
    Membersihkan HTML secara deterministik tanpa LLM.

    Proses:
    1. Decode HTML entity.
    2. Memberi separator pada elemen HTML block/list.
    3. Menghapus seluruh tag HTML.
    4. Merapikan whitespace.
    5. Membersihkan artefak tanda baca hasil penghapusan HTML.
    6. Normalisasi desimal PK, misalnya 0,5 PK -> 0.5 PK.
    """

    if not raw_html:
        return ""

    # Pastikan input berupa string
    text = str(raw_html)

    # ========================================================
    # 1. DECODE HTML ENTITY
    # ========================================================
    text = html.unescape(text)

    # ========================================================
    # 2. UBAH <br> MENJADI PEMISAH KALIMAT
    # ========================================================
    text = re.sub(
        r"<\s*br\s*/?\s*>",
        ". ",
        text,
        flags=re.IGNORECASE,
    )

    # ========================================================
    # 3. BERIKAN PEMISAH PADA CLOSING BLOCK TAG
    # ========================================================
    text = re.sub(
        r"</\s*(p|li|div|ul|ol|h[1-6])\s*>",
        ". ",
        text,
        flags=re.IGNORECASE,
    )

    # ========================================================
    # 4. HAPUS SELURUH TAG HTML YANG TERSISA
    # ========================================================
    text = re.sub(
        r"<[^>]+>",
        " ",
        text,
    )

    # ========================================================
    # 5. NORMALISASI WHITESPACE
    # ========================================================
    text = normalize_whitespace(text)

    # ========================================================
    # 6. RAPikan ARTEFAK TANDA BACA
    # ========================================================

    # Contoh:
    # "Indikator Kerusakan:." -> "Indikator Kerusakan:"
    # "Siap Mengamankan Properti Anda?." ->
    # "Siap Mengamankan Properti Anda?"
    text = re.sub(
        r"([:!?])\s*\.\s*",
        r"\1 ",
        text,
    )

    # Hilangkan titik kosong di awal teks
    # ". Pembersihan AC..." -> "Pembersihan AC..."
    text = re.sub(
        r"^\s*\.\s*",
        "",
        text,
    )

    # Contoh:
    # "Proses Pengerjaan . Survei Lokasi"
    # ->
    # "Proses Pengerjaan. Survei Lokasi"
    text = re.sub(
        r"\s+\.\s+",
        ". ",
        text,
    )

    # Hilangkan spasi sebelum titik
    # "AC dingin ." -> "AC dingin."
    text = re.sub(
        r"\s+\.",
        ".",
        text,
    )

    # Rapikan titik bertumpuk
    # "AC dingin..." -> "AC dingin."
    text = re.sub(
        r"\.{2,}",
        ".",
        text,
    )

    # Rapikan whitespace lagi setelah regex
    text = normalize_whitespace(text)

    # ========================================================
    # 7. NORMALISASI PENULISAN PK
    # ========================================================
    text = normalize_pk_decimal(text)

    return text.strip()


def clean_plain_text(value: Any) -> str:
    """
    Cleaning untuk field yang bukan HTML.
    """
    if value is None:
        return ""

    text = html.unescape(str(value))
    text = normalize_whitespace(text)
    text = normalize_pk_decimal(text)

    return text.strip()


def clean_domain(value: Any) -> str:
    """
    Membersihkan nama kategori/domain seperti:
    'AC ' -> 'AC'
    """
    return clean_plain_text(value)


def numeric_value(value: Any):
    """
    Mengubah Decimal PostgreSQL menjadi int/float
    agar dapat disimpan sebagai JSON.
    """
    if value is None:
        return None

    try:
        number = float(value)

        if number.is_integer():
            return int(number)

        return number

    except (TypeError, ValueError):
        return value


# ============================================================
# 3. QUERY PRODUCT
# ============================================================

PRODUCT_QUERY = """
SELECT
    p.kdprod AS id,
    p.prod_name AS name,
    p.price AS price,
    p.ket_prod AS description,

    j.nmjens AS domain,
    b.nmmerk AS brand,

    pd.detail_product AS detail_html,

    s.srvc_id AS bundling_service_id,
    s.srvc_name AS bundling_service_name,
    s.srvc_desc AS bundling_service_description,
    s.base_price AS bundling_service_price

FROM catalog.products p

LEFT JOIN catalog.brands b
    ON p.kdmerk = b.kdmerk
    AND b.is_active = TRUE

LEFT JOIN catalog.jenis_products j
    ON p.kdjens = j.kdjens
    AND j.is_active = TRUE

LEFT JOIN catalog.product_detail pd
    ON p.kdprod = pd.kdprod
    AND pd.is_active = TRUE

LEFT JOIN catalog.service_items s
    ON p.srvc_id = s.srvc_id
    AND s.is_active = TRUE

WHERE p.is_active = TRUE

ORDER BY p.kdprod;
"""


# ============================================================
# 4. QUERY SERVICE
# ============================================================

SERVICE_QUERY = """
SELECT
    s.srvc_id AS id,
    s.srvc_name AS name,
    s.base_price AS price,
    s.srvc_desc AS description,

    j.nmjens AS domain,

    sp.srv_package_name AS package_name,
    sp.srv_short_desc AS package_description,

    sd.problem_json ->> 'prob_detail' AS detail_html

FROM catalog.service_items s

LEFT JOIN catalog.service_packages sp
    ON s.srvprodid = sp.srv_prodid
    AND sp.is_active = TRUE

LEFT JOIN catalog.jenis_products j
    ON sp.kdjens = j.kdjens
    AND j.is_active = TRUE

LEFT JOIN catalog.service_detail sd
    ON s.srvc_id = sd.srvc_id
    AND sd.is_active = TRUE

WHERE s.is_active = TRUE

ORDER BY s.srvc_id;
"""


# ============================================================
# 5. BUILD PRODUCT RECORD
# ============================================================

def build_product_record(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": clean_plain_text(row["id"]),
        "type": "product",
        "domain": clean_domain(row["domain"]),

        "name": clean_plain_text(row["name"]),
        "brand": clean_plain_text(row["brand"]),
        "price": numeric_value(row["price"]),

        "description": clean_plain_text(
            row["description"]
        ),

        "detail": clean_html_text(
            row["detail_html"]
        ),

        "bundling_service_id": clean_plain_text(
            row["bundling_service_id"]
        ),

        "bundling_service_name": clean_plain_text(
            row["bundling_service_name"]
        ),

        "bundling_service_description": clean_plain_text(
            row["bundling_service_description"]
        ),

        "bundling_service_price": numeric_value(
            row["bundling_service_price"]
        ),
    }


# ============================================================
# 6. BUILD SERVICE RECORD
# ============================================================

def build_service_record(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": clean_plain_text(row["id"]),
        "type": "service",
        "domain": clean_domain(row["domain"]),

        "name": clean_plain_text(row["name"]),
        "price": numeric_value(row["price"]),

        "description": clean_plain_text(
            row["description"]
        ),

        "package_name": clean_plain_text(
            row["package_name"]
        ),

        "package_description": clean_plain_text(
            row["package_description"]
        ),

        "detail": clean_html_text(
            row["detail_html"]
        ),
    }


# ============================================================
# 7. LOAD DARI POSTGRESQL
# ============================================================

def fetch_records():
    conn = get_db_connection()

    try:
        with conn.cursor(
            cursor_factory=RealDictCursor
        ) as cursor:

            # ----------------------------
            # Products
            # ----------------------------
            cursor.execute(PRODUCT_QUERY)
            product_rows = cursor.fetchall()

            products = [
                build_product_record(dict(row))
                for row in product_rows
            ]

            # ----------------------------
            # Services
            # ----------------------------
            cursor.execute(SERVICE_QUERY)
            service_rows = cursor.fetchall()

            services = [
                build_service_record(dict(row))
                for row in service_rows
            ]

        return products, services

    finally:
        conn.close()


# ============================================================
# 8. VALIDASI
# ============================================================

def validate_records(
    products: List[Dict[str, Any]],
    services: List[Dict[str, Any]],
):
    all_records = products + services

    print("\n=== VALIDASI DATASET ===")
    print(f"Produk  : {len(products)}")
    print(f"Jasa    : {len(services)}")
    print(f"Total   : {len(all_records)}")

    if len(products) != EXPECTED_PRODUCTS:
        raise ValueError(
            f"Jumlah produk tidak sesuai. "
            f"Expected={EXPECTED_PRODUCTS}, "
            f"Actual={len(products)}"
        )

    if len(services) != EXPECTED_SERVICES:
        raise ValueError(
            f"Jumlah jasa tidak sesuai. "
            f"Expected={EXPECTED_SERVICES}, "
            f"Actual={len(services)}"
        )

    if len(all_records) != EXPECTED_TOTAL:
        raise ValueError(
            f"Jumlah total tidak sesuai. "
            f"Expected={EXPECTED_TOTAL}, "
            f"Actual={len(all_records)}"
        )

    # ----------------------------------------
    # ID harus unik
    # ----------------------------------------
    ids = [
        record["id"]
        for record in all_records
    ]

    duplicate_ids = sorted({
        item_id
        for item_id in ids
        if ids.count(item_id) > 1
    })

    if duplicate_ids:
        raise ValueError(
            f"Ditemukan duplicate ID: "
            f"{duplicate_ids}"
        )

    # ----------------------------------------
    # ID tidak boleh kosong
    # ----------------------------------------
    empty_ids = [
        record
        for record in all_records
        if not record["id"]
    ]

    if empty_ids:
        raise ValueError(
            "Ditemukan record dengan ID kosong."
        )

    # ----------------------------------------
    # Nama tidak boleh kosong
    # ----------------------------------------
    empty_names = [
        record["id"]
        for record in all_records
        if not record.get("name")
    ]

    if empty_names:
        raise ValueError(
            f"Ditemukan nama kosong pada: "
            f"{empty_names}"
        )

    print("✅ Jumlah record sesuai.")
    print("✅ ID unik.")
    print("✅ Tidak ada ID kosong.")
    print("✅ Tidak ada nama kosong.")


# ============================================================
# 9. JSONL OUTPUT
# ============================================================

def write_jsonl(
    records: List[Dict[str, Any]]
):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with JSONL_OUTPUT.open(
        "w",
        encoding="utf-8",
    ) as file:

        for record in records:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )


# ============================================================
# 10. AUDIT CSV
# ============================================================

def write_audit_csv(
    records: List[Dict[str, Any]]
):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "id",
        "type",
        "domain",
        "name",
        "has_description",
        "has_detail",
        "description_chars",
        "detail_chars",
        "package_name",
        "bundling_service_id",
    ]

    with CSV_AUDIT_OUTPUT.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for record in records:

            description = (
                record.get("description")
                or ""
            )

            detail = (
                record.get("detail")
                or ""
            )

            writer.writerow({
                "id": record["id"],
                "type": record["type"],
                "domain": record["domain"],
                "name": record["name"],

                "has_description": bool(
                    description
                ),

                "has_detail": bool(
                    detail
                ),

                "description_chars": len(
                    description
                ),

                "detail_chars": len(
                    detail
                ),

                "package_name": record.get(
                    "package_name",
                    "",
                ),

                "bundling_service_id": record.get(
                    "bundling_service_id",
                    "",
                ),
            })


# ============================================================
# 11. SUMMARY
# ============================================================

def print_summary(
    records: List[Dict[str, Any]]
):
    print("\n=== DISTRIBUSI ===")

    groups = {}

    for record in records:
        key = (
            record["type"],
            record["domain"],
        )

        groups[key] = (
            groups.get(key, 0)
            + 1
        )

    for key, count in sorted(groups.items()):
        record_type, domain = key

        print(
            f"{record_type:8} | "
            f"{domain:6} | "
            f"{count}"
        )

    print("\n=== DETAIL KOSONG ===")

    empty_details = [
        record
        for record in records
        if not record.get("detail")
    ]

    print(
        f"Jumlah detail kosong: "
        f"{len(empty_details)}"
    )

    for record in empty_details:
        print(
            f"- {record['id']} | "
            f"{record['type']} | "
            f"{record['name']}"
        )


# ============================================================
# 12. MAIN
# ============================================================

def main():
    print(
        "Membangun Raw Canonical Dataset "
        "Aire Optima..."
    )

    products, services = fetch_records()

    validate_records(
        products,
        services,
    )

    all_records = (
        products
        + services
    )

    # Sorting final berdasarkan ID
    all_records = sorted(
        all_records,
        key=lambda record: record["id"],
    )

    write_jsonl(all_records)
    write_audit_csv(all_records)

    print_summary(all_records)

    print("\n✅ RAW CANONICAL DATASET SELESAI")
    print(
        f"JSONL : {JSONL_OUTPUT}"
    )
    print(
        f"Audit : {CSV_AUDIT_OUTPUT}"
    )


if __name__ == "__main__":
    main()