import json
from pathlib import Path
from typing import Dict, Optional

from google import genai
from google.genai import types


SYSTEM_PROMPT_FILE = Path(
    "benchmark/config/"
    "rag_system_prompt_v1.txt"
)

GENERATOR_CONFIG_FILE = Path(
    "benchmark/config/"
    "rag_generator_config_v2.json"
)


class GeminiRAGGenerator:

    def __init__(
        self,
        api_key: str,
    ):

        if not api_key:
            raise ValueError(
                "API key kosong."
            )

        with GENERATOR_CONFIG_FILE.open(
            "r",
            encoding="utf-8",
        ) as file:

            self.config = json.load(
                file
            )

        self.model_name = (
            self.config["model"]
        )

        self.thinking_level = (
            self.config[
                "thinking_level"
            ]
        )

        self.max_output_tokens = (
            self.config[
                "max_output_tokens"
            ]
        )

        self.system_prompt = (
            SYSTEM_PROMPT_FILE
            .read_text(
                encoding="utf-8"
            )
            .strip()
        )

        if not self.system_prompt:
            raise RuntimeError(
                "System prompt kosong."
            )

        self.client = genai.Client(
            api_key=api_key
        )

    # ========================================================
    # CLOSE CLIENT
    # ========================================================

    def close(self):

        try:
            self.client.close()

        except Exception:
            pass

    # ========================================================
    # USER PROMPT
    # ========================================================

    @staticmethod
    def build_user_prompt(
        question: str,
        context_block: str,
    ) -> str:

        return (
            "Berikut adalah informasi "
            "yang tersedia:\n\n"
            f"{context_block}\n\n"
            "PERTANYAAN PENGGUNA:\n"
            f"{question}\n\n"
            "Jawablah pertanyaan pengguna "
            "berdasarkan informasi yang "
            "tersedia di atas."
        )

    # ========================================================
    # SAFE INT
    # ========================================================

    @staticmethod
    def _safe_int(
        value,
    ) -> Optional[int]:

        if value is None:
            return None

        try:
            return int(value)

        except Exception:
            return None

    # ========================================================
    # GENERATE
    # ========================================================

    def generate(
        self,
        question: str,
        context_block: str,
    ) -> Dict:

        user_prompt = (
            self.build_user_prompt(
                question=question,
                context_block=context_block,
            )
        )

        response = (
            self.client.models.generate_content(
                model=self.model_name,

                contents=user_prompt,

                config=types.GenerateContentConfig(
                    system_instruction=(
                        self.system_prompt
                    ),

                    candidate_count=1,

                    max_output_tokens=(
                        self.max_output_tokens
                    ),

                    thinking_config=(
                        types.ThinkingConfig(
                            thinking_level=(
                                self.thinking_level
                            )
                        )
                    ),

                    # Benchmark tidak memakai tools.
                    # Ini juga menghilangkan warning AFC.
                    automatic_function_calling=(
                        types
                        .AutomaticFunctionCallingConfig(
                            disable=True
                        )
                    ),
                ),
            )
        )

        answer = (
            response.text
            or ""
        ).strip()

        if not answer:

            raise RuntimeError(
                "Gemini menghasilkan "
                "jawaban kosong."
            )

        usage = getattr(
            response,
            "usage_metadata",
            None,
        )

        prompt_tokens = None
        output_tokens = None
        thoughts_tokens = None
        total_tokens = None

        if usage is not None:

            prompt_tokens = self._safe_int(
                getattr(
                    usage,
                    "prompt_token_count",
                    None,
                )
            )

            output_tokens = self._safe_int(
                getattr(
                    usage,
                    "candidates_token_count",
                    None,
                )
            )

            thoughts_tokens = self._safe_int(
                getattr(
                    usage,
                    "thoughts_token_count",
                    None,
                )
            )

            total_tokens = self._safe_int(
                getattr(
                    usage,
                    "total_token_count",
                    None,
                )
            )

        model_version = getattr(
            response,
            "model_version",
            None,
        )

        response_id = getattr(
            response,
            "response_id",
            None,
        )

        finish_reason = None

        candidates = getattr(
            response,
            "candidates",
            None,
        )

        if candidates:

            reason = getattr(
                candidates[0],
                "finish_reason",
                None,
            )

            if reason is not None:

                finish_reason = (
                    getattr(
                        reason,
                        "name",
                        None,
                    )
                    or str(reason)
                )

        return {
            "answer":
                answer,

            "model":
                self.model_name,

            "model_version":
                model_version,

            "response_id":
                response_id,

            "finish_reason":
                finish_reason,

            "prompt_tokens":
                prompt_tokens,

            "output_tokens":
                output_tokens,

            "thoughts_tokens":
                thoughts_tokens,

            "total_tokens":
                total_tokens,
        }