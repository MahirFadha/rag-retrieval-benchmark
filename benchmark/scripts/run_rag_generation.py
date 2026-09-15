import hashlib
import json
import os
import re
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from google.genai import errors

from benchmark.generation.gemini_generator import (
    GeminiRAGGenerator,
)


# ============================================================
# 1. PATHS
# ============================================================

RAG_INPUT_FILE = Path(
    "benchmark/data/rag/"
    "rag_generation_inputs_v1.jsonl"
)

OUTPUT_DIR = Path(
    "benchmark/results/rag"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "rag_generated_answers_v2.jsonl"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "rag_generation_summary_v2.json"
)


# ============================================================
# 2. FROZEN INPUT
# ============================================================

EXPECTED_RAG_INPUT_SHA256 = (
    "742937ac9af494eb41fb0662dcdce02a2"
    "cf30f06c1eb02a57c98d37bc7067a93"
)

EXPECTED_INPUT_COUNT = 120


# ============================================================
# 3. RETRY CONFIG
# ============================================================

RETRYABLE_CODES = {
    408,
    429,
    500,
    502,
    503,
    504,
}

# Maksimal jumlah putaran mencoba seluruh project
# untuk satu query jika terjadi kegagalan API.
MAX_ROUNDS = 3

DEFAULT_WAIT_SECONDS = 10

MAX_WAIT_SECONDS = 90


# ============================================================
# 4. RPM RATE LIMIT CONFIG
# ============================================================

# Setiap API key pada eksperimen ini diasumsikan berasal
# dari Google Cloud / Gemini project yang BERBEDA.
#
# Quota masing-masing project:
# 5 Requests Per Minute.
RPM_LIMIT = 5

# Gunakan 62 detik sebagai safety margin untuk rolling
# window server yang secara nominal 60 detik.
RPM_WINDOW_SECONDS = 62.0

# Tambahan kecil saat semua project sedang penuh.
RATE_LIMIT_SLEEP_BUFFER = 0.5

# Jika server memberi 429 tetapi tidak menyediakan
# retry delay yang dapat dibaca, project tersebut
# dicooldown selama satu window penuh.
DEFAULT_429_COOLDOWN_SECONDS = RPM_WINDOW_SECONDS


# ============================================================
# 5. SHA256
# ============================================================

def sha256_file(
    path: Path,
) -> str:

    hasher = hashlib.sha256()

    with path.open("rb") as file:

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
# 6. LOAD API KEYS
# ============================================================

def load_api_keys():
    """
    Load:

    GEMINI_API_KEY
    GEMINI_API_KEY_2
    GEMINI_API_KEY_3
    ...

    Key value tidak pernah ditampilkan.
    """

    load_dotenv()

    discovered = []

    # --------------------------------------------------------
    # PRIMARY KEY
    # --------------------------------------------------------

    primary = os.getenv(
        "GEMINI_API_KEY"
    )

    if primary:

        discovered.append(
            (
                1,
                "GEMINI_API_KEY",
                primary.strip(),
            )
        )

    # --------------------------------------------------------
    # NUMBERED KEYS
    #
    # Tidak berhenti jika ada nomor yang terlewat.
    # --------------------------------------------------------

    pattern = re.compile(
        r"^GEMINI_API_KEY_(\d+)$"
    )

    for env_name, value in os.environ.items():

        match = pattern.match(
            env_name
        )

        if not match:
            continue

        if not value:
            continue

        number = int(
            match.group(1)
        )

        discovered.append(
            (
                number,
                env_name,
                value.strip(),
            )
        )

    # Urut:
    # primary, _2, _3, ...
    discovered.sort(
        key=lambda item:
            item[0]
    )

    # --------------------------------------------------------
    # REMOVE DUPLICATE KEY VALUES
    # --------------------------------------------------------

    unique = []

    seen_values = set()

    for _, label, value in discovered:

        if value in seen_values:
            continue

        seen_values.add(
            value
        )

        unique.append(
            (
                label,
                value,
            )
        )

    if not unique:

        raise RuntimeError(
            "Tidak ada Gemini API key "
            "pada environment/.env."
        )

    print(
        f"API keys / projects tersedia: "
        f"{len(unique)}"
    )

    for label, _ in unique:

        print(
            f"- {label}"
        )

    return unique


# ============================================================
# 7. LOAD INPUT
# ============================================================

def load_inputs():

    records = []

    with RAG_INPUT_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            records.append(
                json.loads(
                    line
                )
            )

    if (
        len(records)
        != EXPECTED_INPUT_COUNT
    ):

        raise RuntimeError(
            "Jumlah RAG input bukan 120.\n"
            f"Actual: {len(records)}"
        )

    return records


# ============================================================
# 8. LOAD CHECKPOINT
# ============================================================

def load_completed_pairs():
    """
    Membaca output JSONL yang sudah berhasil.

    Key:
        (query_id, method)

    Dengan demikian script dapat dilanjutkan
    tanpa generate ulang jawaban yang sudah sukses.
    """

    completed = {}

    if not OUTPUT_FILE.exists():

        return completed

    with OUTPUT_FILE.open(
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

                record = json.loads(
                    line
                )

            except json.JSONDecodeError as exc:

                raise RuntimeError(
                    "Checkpoint JSONL rusak "
                    f"pada line {line_number}."
                ) from exc

            pair = (
                record["query_id"],
                record["method"],
            )

            if pair in completed:

                raise RuntimeError(
                    "Duplicate checkpoint pair: "
                    f"{pair}"
                )

            completed[
                pair
            ] = record

    return completed


# ============================================================
# 9. EXTRACT RETRY DELAY
# ============================================================

def extract_retry_delay(
    message: str,
):
    """
    Mencoba membaca retry delay dari error Gemini.

    Contoh kemungkinan:
        retry in 4.8s
        retryDelay: '4s'
        retry after 5s
    """

    if not message:

        return None

    patterns = [
        r"retry in\s+([\d.]+)\s*s",
        r"retryDelay['\"]?\s*:\s*['\"]?([\d.]+)s",
        r"retry after\s+([\d.]+)\s*s",
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            message,
            re.IGNORECASE,
        )

        if not match:
            continue

        try:

            delay = float(
                match.group(1)
            )

            # +1 detik sebagai safety buffer.
            return min(
                delay + 1.0,
                MAX_WAIT_SECONDS,
            )

        except Exception:
            pass

    return None


# ============================================================
# 10. PROJECT RPM RATE LIMITER
# ============================================================

class ProjectRateLimiter:
    """
    Sliding-window RPM limiter untuk SATU project.

    Karena setiap GEMINI_API_KEY pada eksperimen ini
    berasal dari project berbeda, setiap key memiliki
    limiter sendiri.

    Contoh:
        RPM_LIMIT = 5
        RPM_WINDOW_SECONDS = 62

    Maka satu project tidak akan sengaja menerima lebih
    dari 5 request dalam rolling window 62 detik.
    """

    def __init__(
        self,
        rpm_limit: int = RPM_LIMIT,
        window_seconds: float = RPM_WINDOW_SECONDS,
    ):

        self.rpm_limit = (
            rpm_limit
        )

        self.window_seconds = (
            window_seconds
        )

        # Timestamp request yang masih
        # berada di rolling window.
        self.request_times = deque()

        # Digunakan ketika server memberi
        # explicit cooldown / 429.
        self.blocked_until = 0.0

    # ========================================================
    # CLEAN OLD REQUESTS
    # ========================================================

    def cleanup(
        self,
    ):

        now = time.time()

        while self.request_times:

            age = (
                now
                - self.request_times[0]
            )

            if (
                age
                >= self.window_seconds
            ):

                self.request_times.popleft()

            else:

                break

    # ========================================================
    # REGISTER REQUEST
    # ========================================================

    def register_request(
        self,
        timestamp=None,
    ):
        """
        Catat request sebagai bagian dari RPM.

        Request dicatat SEBELUM request API dikirim,
        sehingga failed request juga dianggap mencoba
        menggunakan quota.
        """

        self.cleanup()

        if timestamp is None:

            timestamp = (
                time.time()
            )

        self.request_times.append(
            float(
                timestamp
            )
        )

    # ========================================================
    # TEMPORARY BLOCK
    # ========================================================

    def block_for(
        self,
        seconds: float,
    ):

        if seconds <= 0:
            return

        blocked_until = (
            time.time()
            + seconds
        )

        self.blocked_until = max(
            self.blocked_until,
            blocked_until,
        )

    # ========================================================
    # SECONDS UNTIL AVAILABLE
    # ========================================================

    def seconds_until_available(
        self,
    ) -> float:

        self.cleanup()

        now = time.time()

        waits = []

        # ----------------------------------------------------
        # SERVER COOLDOWN
        # ----------------------------------------------------

        if (
            self.blocked_until
            > now
        ):

            waits.append(
                self.blocked_until
                - now
            )

        # ----------------------------------------------------
        # LOCAL 5 RPM WINDOW
        # ----------------------------------------------------

        if (
            len(
                self.request_times
            )
            >= self.rpm_limit
        ):

            oldest = (
                self.request_times[0]
            )

            window_wait = (
                self.window_seconds
                - (
                    now
                    - oldest
                )
            )

            waits.append(
                max(
                    0.0,
                    window_wait,
                )
            )

        if not waits:

            return 0.0

        return max(
            waits
        )

    # ========================================================
    # CURRENT ACTIVE REQUEST COUNT
    # ========================================================

    def active_request_count(
        self,
    ) -> int:

        self.cleanup()

        return len(
            self.request_times
        )


# ============================================================
# 11. BUILD RATE LIMITERS
# ============================================================

def build_rate_limiters(
    api_keys,
    completed,
):
    """
    Buat satu RPM limiter untuk setiap key/project.

    Jika script direstart dalam waktu kurang dari
    RPM_WINDOW_SECONDS, checkpoint digunakan untuk
    me-reconstruct successful requests terbaru.

    generated_at_utc berasal dari checkpoint output.
    """

    limiters = {
        index:
            ProjectRateLimiter()

        for index
        in range(
            len(api_keys)
        )
    }

    label_to_index = {
        label:
            index

        for index, (
            label,
            _
        )
        in enumerate(
            api_keys
        )
    }

    now = time.time()

    restored = 0

    for record in completed.values():

        key_label = (
            record.get(
                "api_key_label"
            )
        )

        generated_at = (
            record.get(
                "generated_at_utc"
            )
        )

        if (
            key_label
            not in label_to_index
        ):

            continue

        if not generated_at:

            continue

        try:

            timestamp = (
                datetime.fromisoformat(
                    generated_at
                )
                .timestamp()
            )

        except Exception:

            continue

        age = (
            now
            - timestamp
        )

        # ----------------------------------------------------
        # Hanya restore jika masih berada dalam rolling window.
        #
        # generated_at adalah waktu selesai response,
        # sehingga ini sedikit konservatif dibanding waktu
        # request sebenarnya — itu justru aman.
        # ----------------------------------------------------

        if (
            0
            <= age
            < RPM_WINDOW_SECONDS
        ):

            key_index = (
                label_to_index[
                    key_label
                ]
            )

            limiters[
                key_index
            ].register_request(
                timestamp
            )

            restored += 1

    print(
        f"RPM state dari checkpoint: "
        f"{restored} recent request(s)"
    )

    # Debug ringan.
    for index, (
        label,
        _
    ) in enumerate(
        api_keys
    ):

        active = (
            limiters[
                index
            ]
            .active_request_count()
        )

        if active > 0:

            print(
                f"  {label}: "
                f"{active}/{RPM_LIMIT} "
                "slot masih aktif"
            )

    return limiters


# ============================================================
# 12. CHOOSE AVAILABLE PROJECT
# ============================================================

def choose_available_key(
    api_keys,
    rate_limiters,
    start_key_index,
):
    """
    Memilih key/project yang masih punya slot RPM.

    Urutan pencarian:
        start_key_index
        start_key_index + 1
        ...
        lalu kembali ke index 0.

    Jika semua project penuh/cooldown:
        tunggu hingga project tercepat tersedia.
    """

    total_keys = len(
        api_keys
    )

    while True:

        candidates = []

        for offset in range(
            total_keys
        ):

            key_index = (
                start_key_index
                + offset
            ) % total_keys

            wait_seconds = (
                rate_limiters[
                    key_index
                ]
                .seconds_until_available()
            )

            candidates.append(
                (
                    wait_seconds,
                    offset,
                    key_index,
                )
            )

        # ----------------------------------------------------
        # PROJECT YANG BISA DIPAKAI SEKARANG
        # ----------------------------------------------------

        available = [
            item
            for item
            in candidates
            if item[0] <= 0
        ]

        if available:

            # candidates sudah disusun berdasarkan
            # urutan round-robin dari start_key_index.
            _, _, key_index = (
                available[0]
            )

            return key_index

        # ----------------------------------------------------
        # SEMUA PROJECT PENUH
        # ----------------------------------------------------

        (
            wait_seconds,
            _,
            next_key_index,
        ) = min(
            candidates,
            key=lambda item:
                item[0],
        )

        wait_seconds = (
            wait_seconds
            + RATE_LIMIT_SLEEP_BUFFER
        )

        next_label = (
            api_keys[
                next_key_index
            ][0]
        )

        print(
            f"    ⏳ Semua project sedang "
            f"mencapai limit/cooldown."
        )

        print(
            f"    Project tercepat tersedia: "
            f"{next_label}"
        )

        print(
            f"    Menunggu "
            f"{wait_seconds:.1f}s..."
        )

        time.sleep(
            wait_seconds
        )


# ============================================================
# 13. APPEND CHECKPOINT
# ============================================================

def append_result(
    record,
):
    """
    Setiap successful answer langsung ditulis ke disk.

    Jadi apabila:
        - Ctrl+C
        - internet mati
        - terminal ditutup
        - quota habis

    hasil sebelumnya tetap aman.
    """

    with OUTPUT_FILE.open(
        "a",
        encoding="utf-8",
        newline="\n",
    ) as file:

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

        file.flush()

        os.fsync(
            file.fileno()
        )


# ============================================================
# 14. GENERATE WITH RPM-AWARE FAILOVER
# ============================================================

def generate_with_failover(
    input_record,
    api_keys,
    rate_limiters,
    start_key_index,
):

    question = (
        input_record[
            "question"
        ]
    )

    context_block = (
        input_record[
            "context_block"
        ]
    )

    total_keys = len(
        api_keys
    )

    max_attempts = (
        total_keys
        * MAX_ROUNDS
    )

    attempts = 0

    last_error = None

    while (
        attempts
        < max_attempts
    ):

        # ====================================================
        # CARI PROJECT YANG MASIH PUNYA RPM SLOT
        # ====================================================

        key_index = (
            choose_available_key(
                api_keys=
                    api_keys,

                rate_limiters=
                    rate_limiters,

                start_key_index=
                    start_key_index,
            )
        )

        key_label, api_key = (
            api_keys[
                key_index
            ]
        )

        limiter = (
            rate_limiters[
                key_index
            ]
        )

        active_before = (
            limiter
            .active_request_count()
        )

        print(
            f"    key: {key_label} "
            f"| RPM lokal: "
            f"{active_before}/{RPM_LIMIT}"
        )

        generator = None

        try:

            generator = (
                GeminiRAGGenerator(
                    api_key=api_key
                )
            )

            # =================================================
            # REGISTER SEBELUM API CALL
            # =================================================

            limiter.register_request()

            started = (
                time.perf_counter()
            )

            result = (
                generator.generate(
                    question=question,
                    context_block=
                        context_block,
                )
            )

            latency_seconds = (
                time.perf_counter()
                - started
            )

            return {
                "result":
                    result,

                "api_key_label":
                    key_label,

                "api_key_index":
                    key_index,

                "latency_seconds":
                    latency_seconds,
            }

        # ====================================================
        # GEMINI API ERROR
        # ====================================================

        except errors.APIError as exc:

            last_error = exc

            attempts += 1

            code = getattr(
                exc,
                "code",
                None,
            )

            # Beberapa SDK dapat memberi numeric string.
            try:

                if code is not None:
                    code = int(code)

            except Exception:
                pass

            message = str(
                exc
            )

            print(
                f"    ⚠️ API error "
                f"{code}: "
                f"{message[:200]}"
            )

            # =================================================
            # NON-RETRYABLE ERROR
            # =================================================

            if (
                code
                not in RETRYABLE_CODES
            ):

                raise

            # =================================================
            # 429 — RATE LIMIT / QUOTA
            # =================================================

            if code == 429:

                server_delay = (
                    extract_retry_delay(
                        message
                    )
                )

                local_delay = (
                    limiter
                    .seconds_until_available()
                )

                if server_delay is not None:

                    cooldown = max(
                        server_delay,
                        local_delay,
                    )

                elif local_delay > 0:

                    cooldown = (
                        local_delay
                    )

                else:

                    # Tidak tahu jenis quota apa yang memicu
                    # 429. Safe fallback = satu window penuh.
                    cooldown = (
                        DEFAULT_429_COOLDOWN_SECONDS
                    )

                limiter.block_for(
                    cooldown
                )

                print(
                    f"    Project "
                    f"{key_label} cooldown "
                    f"{cooldown:.1f}s."
                )

                print(
                    "    Beralih ke project "
                    "berikutnya..."
                )

                start_key_index = (
                    key_index
                    + 1
                ) % total_keys

                # Tidak sleep di sini.
                #
                # choose_available_key() akan mencari project
                # lain yang masih tersedia. Kalau semuanya
                # penuh, barulah fungsi tersebut menunggu.
                continue

            # =================================================
            # 408 / 500 / 502 / 503 / 504
            # =================================================

            server_wait = min(
                DEFAULT_WAIT_SECONDS
                * attempts,
                MAX_WAIT_SECONDS,
            )

            limiter.block_for(
                server_wait
            )

            print(
                f"    Project "
                f"{key_label} cooldown "
                f"{server_wait:.1f}s "
                "karena server error."
            )

            start_key_index = (
                key_index
                + 1
            ) % total_keys

            continue

        # ====================================================
        # TECHNICAL / NETWORK ERROR
        # ====================================================

        except Exception as exc:

            last_error = exc

            attempts += 1

            print(
                "    ⚠️ Technical error: "
                f"{type(exc).__name__}: "
                f"{str(exc)[:200]}"
            )

            limiter.block_for(
                DEFAULT_WAIT_SECONDS
            )

            start_key_index = (
                key_index
                + 1
            ) % total_keys

            continue

        finally:

            if generator is not None:

                generator.close()

    raise RuntimeError(
        "Semua project/retry gagal "
        f"untuk "
        f"{input_record['query_id']} / "
        f"{input_record['method']}.\n"
        f"Attempts: {attempts}\n"
        f"Last error: {last_error}"
    )


# ============================================================
# 15. DETERMINE RESUME KEY
# ============================================================

def determine_start_key_index(
    api_keys,
    completed,
):
    """
    Saat fresh run:
        mulai key 0.

    Saat resume:
        mulai dari project setelah key yang terakhir
        berhasil disimpan pada checkpoint.
    """

    if not completed:

        return 0

    # dict mempertahankan insertion order.
    last_record = list(
        completed.values()
    )[-1]

    last_label = (
        last_record.get(
            "api_key_label"
        )
    )

    if not last_label:

        return 0

    label_to_index = {
        label:
            index

        for index, (
            label,
            _
        )
        in enumerate(
            api_keys
        )
    }

    if (
        last_label
        not in label_to_index
    ):

        return 0

    return (
        label_to_index[
            last_label
        ]
        + 1
    ) % len(api_keys)


# ============================================================
# 16. MAIN
# ============================================================

def main():

    print(
        "=" * 70
    )

    print(
        "AIRE OPTIMA RAG GENERATION V1"
    )

    print(
        "=" * 70
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # VERIFY FROZEN RAG INPUT
    # ========================================================

    rag_hash = sha256_file(
        RAG_INPUT_FILE
    )

    if (
        rag_hash
        != EXPECTED_RAG_INPUT_SHA256
    ):

        raise RuntimeError(
            "RAG input SHA256 berubah.\n"
            f"Expected: "
            f"{EXPECTED_RAG_INPUT_SHA256}\n"
            f"Actual  : "
            f"{rag_hash}"
        )

    print(
        "✅ Frozen RAG input valid."
    )

    # ========================================================
    # LOAD INPUT + KEYS + CHECKPOINT
    # ========================================================

    inputs = load_inputs()

    api_keys = load_api_keys()

    completed = (
        load_completed_pairs()
    )

    # ========================================================
    # RATE LIMITERS
    # ========================================================

    rate_limiters = (
        build_rate_limiters(
            api_keys=
                api_keys,

            completed=
                completed,
        )
    )

    print(
        f"\nRPM configuration:"
    )

    print(
        f"- Limit per project : "
        f"{RPM_LIMIT} RPM"
    )

    print(
        f"- Window            : "
        f"{RPM_WINDOW_SECONDS:.1f}s"
    )

    print(
        f"- Projects          : "
        f"{len(api_keys)}"
    )

    print(
        f"- Theoretical max   : "
        f"{RPM_LIMIT * len(api_keys)} "
        "requests/window"
    )

    print(
        f"\nTotal input     : "
        f"{len(inputs)}"
    )

    print(
        f"Sudah selesai   : "
        f"{len(completed)}"
    )

    print(
        f"Belum selesai   : "
        f"{len(inputs) - len(completed)}"
    )

    # ========================================================
    # TOKEN TOTALS FROM CHECKPOINT
    # ========================================================

    total_prompt_tokens = sum(
        row.get(
            "prompt_tokens"
        )
        or 0

        for row
        in completed.values()
    )

    total_output_tokens = sum(
        row.get(
            "output_tokens"
        )
        or 0

        for row
        in completed.values()
    )

    total_thoughts_tokens = sum(
        row.get(
            "thoughts_tokens"
        )
        or 0

        for row
        in completed.values()
    )

    total_tokens = sum(
        row.get(
            "total_tokens"
        )
        or 0

        for row
        in completed.values()
    )

    # ========================================================
    # START KEY
    # ========================================================

    current_key_index = (
        determine_start_key_index(
            api_keys=
                api_keys,

            completed=
                completed,
        )
    )

    print(
        "\nRound-robin dimulai dari:"
    )

    print(
        api_keys[
            current_key_index
        ][0]
    )

    # ========================================================
    # GENERATION
    # ========================================================

    for sequence_number, input_record in enumerate(
        inputs,
        start=1,
    ):

        pair = (
            input_record[
                "query_id"
            ],
            input_record[
                "method"
            ],
        )

        # ----------------------------------------------------
        # CHECKPOINT / RESUME
        # ----------------------------------------------------

        if pair in completed:

            print(
                f"[{sequence_number:03d}/120] "
                f"{pair[0]} | "
                f"{pair[1]} "
                "→ SKIP (checkpoint)"
            )

            continue

        print(
            "\n"
            + "-" * 70
        )

        print(
            f"[{sequence_number:03d}/120] "
            f"{pair[0]} | "
            f"{pair[1]}"
        )

        try:

            generated = (
                generate_with_failover(
                    input_record=
                        input_record,

                    api_keys=
                        api_keys,

                    rate_limiters=
                        rate_limiters,

                    start_key_index=
                        current_key_index,
                )
            )

        except KeyboardInterrupt:

            print(
                "\n\n⚠️ PROCESS DIBATALKAN USER."
            )

            print(
                "Checkpoint jawaban yang sudah "
                "berhasil tetap tersimpan."
            )

            print(
                "Jalankan command yang sama "
                "untuk resume."
            )

            return

        except Exception as exc:

            print(
                "\n❌ GENERATION STOPPED"
            )

            print(
                f"{type(exc).__name__}: "
                f"{exc}"
            )

            print(
                "\nCheckpoint tersimpan."
            )

            print(
                "Jalankan command yang sama "
                "untuk melanjutkan."
            )

            raise

        result = (
            generated[
                "result"
            ]
        )

        # ====================================================
        # ROUND-ROBIN
        #
        # Ini berbeda dari kode lama.
        #
        # Setelah key N sukses, request berikutnya
        # MULAI mencari dari key N+1.
        # ====================================================

        current_key_index = (
            generated[
                "api_key_index"
            ]
            + 1
        ) % len(api_keys)

        # ====================================================
        # OUTPUT RECORD
        # ====================================================

        output_record = {
            "query_id":
                input_record[
                    "query_id"
                ],

            "method":
                input_record[
                    "method"
                ],

            "answer":
                result[
                    "answer"
                ],

            "model":
                result[
                    "model"
                ],

            "model_version":
                result[
                    "model_version"
                ],

            "finish_reason":
                result[
                    "finish_reason"
                ],

            "response_id":
                result[
                    "response_id"
                ],

            # Hanya LABEL variable.
            # API key actual TIDAK disimpan.
            "api_key_label":
                generated[
                    "api_key_label"
                ],

            "prompt_tokens":
                result[
                    "prompt_tokens"
                ],

            "output_tokens":
                result[
                    "output_tokens"
                ],

            "thoughts_tokens":
                result[
                    "thoughts_tokens"
                ],

            "total_tokens":
                result[
                    "total_tokens"
                ],

            "latency_seconds":
                round(
                    generated[
                        "latency_seconds"
                    ],
                    4,
                ),

            "generated_at_utc":
                datetime.now(
                    timezone.utc
                ).isoformat(),
        }

        # ====================================================
        # CHECKPOINT IMMEDIATELY
        # ====================================================

        append_result(
            output_record
        )

        completed[
            pair
        ] = output_record

        # ====================================================
        # TOKEN TOTALS
        # ====================================================

        total_prompt_tokens += (
            result[
                "prompt_tokens"
            ]
            or 0
        )

        total_output_tokens += (
            result[
                "output_tokens"
            ]
            or 0
        )

        total_thoughts_tokens += (
            result[
                "thoughts_tokens"
            ]
            or 0
        )

        total_tokens += (
            result[
                "total_tokens"
            ]
            or 0
        )

        print(
            "    ✅ tersimpan"
        )

        print(
            f"    model_version: "
            f"{result['model_version']}"
        )

        print(
            f"    tokens: "
            f"{result['total_tokens']}"
        )

        print(
            f"    latency: "
            f"{generated['latency_seconds']:.2f}s"
        )

        next_label = (
            api_keys[
                current_key_index
            ][0]
        )

        print(
            f"    next start key: "
            f"{next_label}"
        )

    # ========================================================
    # FINAL VALIDATION
    # ========================================================

    completed = (
        load_completed_pairs()
    )

    if (
        len(completed)
        != EXPECTED_INPUT_COUNT
    ):

        raise RuntimeError(
            "Generation belum lengkap. "
            f"Current: "
            f"{len(completed)}/"
            f"{EXPECTED_INPUT_COUNT}"
        )

    # ========================================================
    # HASH FINAL OUTPUT
    # ========================================================

    output_hash = sha256_file(
        OUTPUT_FILE
    )

    # ========================================================
    # SUMMARY COUNTS
    # ========================================================

    key_usage = {}

    method_counts = {}

    model_versions = {}

    latency_values = []

    for record in completed.values():

        key_label = (
            record.get(
                "api_key_label",
                "unknown",
            )
        )

        key_usage[
            key_label
        ] = (
            key_usage.get(
                key_label,
                0,
            )
            + 1
        )

        method = (
            record[
                "method"
            ]
        )

        method_counts[
            method
        ] = (
            method_counts.get(
                method,
                0,
            )
            + 1
        )

        model_version = (
            record.get(
                "model_version"
            )
        )

        model_versions[
            model_version
        ] = (
            model_versions.get(
                model_version,
                0,
            )
            + 1
        )

        latency = record.get(
            "latency_seconds"
        )

        if latency is not None:

            try:

                latency_values.append(
                    float(
                        latency
                    )
                )

            except Exception:
                pass

    # ========================================================
    # RECOMPUTE TOKEN TOTALS FROM FINAL CHECKPOINT
    #
    # Supaya summary selalu sesuai file final,
    # bahkan setelah resume beberapa kali.
    # ========================================================

    total_prompt_tokens = sum(
        row.get(
            "prompt_tokens"
        )
        or 0

        for row
        in completed.values()
    )

    total_output_tokens = sum(
        row.get(
            "output_tokens"
        )
        or 0

        for row
        in completed.values()
    )

    total_thoughts_tokens = sum(
        row.get(
            "thoughts_tokens"
        )
        or 0

        for row
        in completed.values()
    )

    total_tokens = sum(
        row.get(
            "total_tokens"
        )
        or 0

        for row
        in completed.values()
    )

    mean_latency = None

    if latency_values:

        mean_latency = (
            sum(
                latency_values
            )
            / len(
                latency_values
            )
        )

    # ========================================================
    # SUMMARY JSON
    # ========================================================

    summary = {
        "artifact":
            "Aire Optima RAG "
            "Generated Answers v2",

        "status":
            "COMPLETE",

        "generation_count":
            len(
                completed
            ),

        "rag_input_sha256":
            rag_hash,

        "generated_answers_file":
            str(
                OUTPUT_FILE
            ),

        "generated_answers_sha256":
            output_hash,

        "method_counts":
            method_counts,

        "model_versions":
            model_versions,

        "api_key_usage":
            key_usage,

        "rate_limit": {
            "strategy":
                "per_project_sliding_window_round_robin",

            "project_count":
                len(
                    api_keys
                ),

            "rpm_per_project":
                RPM_LIMIT,

            "window_seconds":
                RPM_WINDOW_SECONDS,

            "safety_buffer_seconds":
                RATE_LIMIT_SLEEP_BUFFER,
        },

        "token_usage": {
            "prompt_tokens":
                total_prompt_tokens,

            "output_tokens":
                total_output_tokens,

            "thoughts_tokens":
                total_thoughts_tokens,

            "total_tokens":
                total_tokens,
        },

        "latency": {
            "mean_seconds":
                mean_latency,
        },
    }

    with SUMMARY_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
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
        "RAG GENERATION COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"Answers       : "
        f"{len(completed)}"
    )

    print(
        f"Prompt tokens : "
        f"{total_prompt_tokens}"
    )

    print(
        f"Output tokens : "
        f"{total_output_tokens}"
    )

    print(
        f"Thought tokens: "
        f"{total_thoughts_tokens}"
    )

    print(
        f"Total tokens  : "
        f"{total_tokens}"
    )

    if (
        mean_latency
        is not None
    ):

        print(
            f"Mean latency  : "
            f"{mean_latency:.2f}s"
        )

    print(
        "\nAPI KEY / PROJECT USAGE:"
    )

    for (
        key_label,
        count,
    ) in sorted(
        key_usage.items()
    ):

        print(
            f"{key_label:<22} "
            f"{count}"
        )

    print(
        "\nMODEL VERSIONS:"
    )

    for (
        model_version,
        count,
    ) in model_versions.items():

        print(
            f"{model_version}: "
            f"{count}"
        )

    print(
        "\nGENERATED ANSWERS SHA256:"
    )

    print(
        output_hash
    )

    print(
        f"\nAnswers:\n"
        f"{OUTPUT_FILE}"
    )

    print(
        f"\nSummary:\n"
        f"{SUMMARY_FILE}"
    )


if __name__ == "__main__":
    main()