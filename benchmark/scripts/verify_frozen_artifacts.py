import hashlib
from pathlib import Path


# ============================================================
# 1. FROZEN ARTIFACTS
# ============================================================

CORPUS_FILE = Path(
    "benchmark/data/canonical/"
    "aire_catalog_final_v1.jsonl"
)

QUERY_FILE = Path(
    "benchmark/data/canonical/"
    "aire_benchmark_queries_v1.jsonl"
)


EXPECTED_CORPUS_SHA256 = (
    "7e5c041917e43b32298ecf7dc02b571e"
    "11ffe6cf5c77b52dcf8b531d6b4609be"
)

EXPECTED_QUERY_SHA256 = (
    "dfbf771a3f7dc00d465be4e2605f9cca"
    "e7b0e2147933954d367cfcd0751bfb0d"
)


# ============================================================
# 2. SHA256
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
# 3. VERIFY ONE FILE
# ============================================================

def verify_file(
    label: str,
    path: Path,
    expected_hash: str,
):
    if not path.exists():
        raise FileNotFoundError(
            f"{label} tidak ditemukan:\n"
            f"{path}"
        )

    actual_hash = sha256_file(
        path
    )

    print(
        f"\n=== {label} ==="
    )

    print(
        f"File     : {path}"
    )

    print(
        f"Expected : {expected_hash}"
    )

    print(
        f"Actual   : {actual_hash}"
    )

    if actual_hash == expected_hash:

        print(
            "✅ SHA256 COCOK"
        )

        return True

    print(
        "❌ SHA256 TIDAK COCOK"
    )

    return False


# ============================================================
# 4. MAIN
# ============================================================

def main():

    print(
        "Verifikasi Frozen Benchmark Artifacts..."
    )

    corpus_ok = verify_file(
        label="CANONICAL CORPUS V1",
        path=CORPUS_FILE,
        expected_hash=(
            EXPECTED_CORPUS_SHA256
        ),
    )

    query_ok = verify_file(
        label="QUERY SET V1",
        path=QUERY_FILE,
        expected_hash=(
            EXPECTED_QUERY_SHA256
        ),
    )

    print(
        "\n=== FINAL RESULT ==="
    )

    if corpus_ok and query_ok:

        print(
            "✅ Semua frozen artifact valid."
        )

        print(
            "Benchmark boleh dilanjutkan."
        )

    else:

        raise RuntimeError(
            "Frozen artifact berubah. "
            "Jangan jalankan benchmark "
            "sebelum masalah diperiksa."
        )


if __name__ == "__main__":
    main()