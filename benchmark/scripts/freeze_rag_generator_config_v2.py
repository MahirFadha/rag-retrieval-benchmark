import hashlib
import json
from pathlib import Path


SYSTEM_PROMPT_FILE = Path(
    "benchmark/config/"
    "rag_system_prompt_v1.txt"
)

GENERATOR_CONFIG_FILE = Path(
    "benchmark/config/"
    "rag_generator_config_v2.json"
)

RAG_INPUT_FILE = Path(
    "benchmark/data/rag/"
    "rag_generation_inputs_v1.jsonl"
)

MANIFEST_FILE = Path(
    "benchmark/config/"
    "rag_generator_manifest_v2.json"
)


EXPECTED_RAG_INPUT_SHA256 = (
    "742937ac9af494eb41fb0662dcdce02a2"
    "cf30f06c1eb02a57c98d37bc7067a93"
)


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


def main():

    print(
        "=" * 70
    )

    print(
        "FREEZE RAG GENERATOR CONFIG V1"
    )

    print(
        "=" * 70
    )

    for path in [
        SYSTEM_PROMPT_FILE,
        GENERATOR_CONFIG_FILE,
        RAG_INPUT_FILE,
    ]:

        if not path.exists():

            raise FileNotFoundError(
                f"File tidak ditemukan: {path}"
            )

    # --------------------------------------------------------
    # VALIDATE CONFIG
    # --------------------------------------------------------

    with GENERATOR_CONFIG_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:

        config = json.load(file)

    expected = {
        "model":
            "gemini-3.6-flash",

        "thinking_level":
            "low",

        "max_output_tokens":
            2048,

        "candidate_count":
            1,

        "rag_top_k":
            5,

        "expected_inputs":
            120,
    }

    for key, expected_value in expected.items():

        actual = config.get(key)

        if actual != expected_value:

            raise RuntimeError(
                f"Config {key} berubah.\n"
                f"Expected: {expected_value}\n"
                f"Actual  : {actual}"
            )

    # Sampling must remain unset/default.
    for key in [
        "temperature",
        "top_p",
        "top_k",
    ]:

        if config.get(key) is not None:

            raise RuntimeError(
                f"{key} harus null/default."
            )

    # --------------------------------------------------------
    # HASH
    # --------------------------------------------------------

    system_prompt_hash = sha256_file(
        SYSTEM_PROMPT_FILE
    )

    generator_config_hash = sha256_file(
        GENERATOR_CONFIG_FILE
    )

    rag_input_hash = sha256_file(
        RAG_INPUT_FILE
    )

    if (
        rag_input_hash
        != EXPECTED_RAG_INPUT_SHA256
    ):

        raise RuntimeError(
            "RAG generation input berubah."
        )

    # --------------------------------------------------------
    # MANIFEST
    # --------------------------------------------------------

    manifest = {
        "artifact":
            "Aire Optima RAG Generator Configuration v1",

        "status":
            "FROZEN",

        "system_prompt": {
            "file":
                str(SYSTEM_PROMPT_FILE),

            "sha256":
                system_prompt_hash,
        },

        "generator_config": {
            "file":
                str(GENERATOR_CONFIG_FILE),

            "sha256":
                generator_config_hash,
        },

        "rag_input": {
            "file":
                str(RAG_INPUT_FILE),

            "sha256":
                rag_input_hash,
        },

        "generator": config,

        "freeze_rule": (
            "Do not modify system prompt, "
            "generator configuration, or "
            "RAG generation inputs after "
            "generation begins."
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

    print(
        "\nSYSTEM PROMPT SHA256:"
    )

    print(
        system_prompt_hash
    )

    print(
        "\nGENERATOR CONFIG SHA256:"
    )

    print(
        generator_config_hash
    )

    print(
        "\nRAG INPUT SHA256:"
    )

    print(
        rag_input_hash
    )

    print(
        "\nSTATUS: FROZEN"
    )

    print(
        f"\nManifest:\n{MANIFEST_FILE}"
    )


if __name__ == "__main__":
    main()