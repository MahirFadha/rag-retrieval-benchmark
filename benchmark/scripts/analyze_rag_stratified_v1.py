import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy
from scipy import stats


# ============================================================
# 1. PATHS
# ============================================================

MASTER_JSONL_FILE = Path(
    "benchmark/results/analysis/"
    "rag_master_analysis_v1.jsonl"
)

MASTER_CSV_FILE = Path(
    "benchmark/results/analysis/"
    "rag_master_analysis_v1.csv"
)

MASTER_MANIFEST_FILE = Path(
    "benchmark/results/analysis/"
    "rag_master_analysis_manifest_v1.json"
)

OUTPUT_DIR = Path(
    "benchmark/results/analysis/stratified"
)

CATEGORY_SUMMARY_CSV = (
    OUTPUT_DIR
    / "rag_category_summary_v1.csv"
)

CATEGORY_FRIEDMAN_CSV = (
    OUTPUT_DIR
    / "rag_category_friedman_v1.csv"
)

DOMAIN_SUMMARY_CSV = (
    OUTPUT_DIR
    / "rag_domain_summary_v1.csv"
)

DOMAIN_FRIEDMAN_CSV = (
    OUTPUT_DIR
    / "rag_domain_friedman_v1.csv"
)

TARGET_TOP5_CSV = (
    OUTPUT_DIR
    / "rag_target_top5_analysis_v1.csv"
)

RANK_PROFILE_CSV = (
    OUTPUT_DIR
    / "rag_rank_profile_v1.csv"
)

RANK_SPEARMAN_CSV = (
    OUTPUT_DIR
    / "rag_rank_spearman_v1.csv"
)

SUMMARY_JSON = (
    OUTPUT_DIR
    / "rag_stratified_analysis_summary_v1.json"
)

MANIFEST_FILE = (
    OUTPUT_DIR
    / "rag_stratified_analysis_manifest_v1.json"
)


# ============================================================
# 2. FROZEN SOURCE HASHES
# ============================================================

EXPECTED_MASTER_JSONL_SHA256 = (
    "211936a8cbfe06e1f1b658a745371599"
    "f0362c8da5f80344c1ade4f695e95280"
)

EXPECTED_MASTER_CSV_SHA256 = (
    "cbfccaeadbb72ce549cedcac154e93e3"
    "cabda64a761275cf945259363901c4cb"
)

EXPECTED_MASTER_MANIFEST_SHA256 = (
    "e6eed1b98ac9b03baad57072bb184eed"
    "df0f0d8d4b63fe535b1af8b08bf4a52d"
)


# ============================================================
# 3. CONSTANTS
# ============================================================

EXPECTED_ROWS = 120
EXPECTED_QUERIES = 40

ALPHA = 0.05


METHODS = [
    "bm25",
    "e5",
    "hybrid_rrf",
]


RAG_METRICS = [
    "faithfulness",
    "answer_relevancy",
    "answer_accuracy",
]


CATEGORIES = [
    "exact",
    "semantic",
    "multi_constraint",
    "fine_grained",
]


DOMAINS = [
    "product_ac",
    "service_ac",
    "cctv",
]


# ============================================================
# 4. HELPERS
# ============================================================

def sha256_file(path):

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


def load_json(path):

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(
            file
        )


def load_jsonl(path):

    records = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        for (
            line_number,
            line,
        ) in enumerate(
            file,
            start=1,
        ):

            line = line.strip()

            if not line:
                continue

            try:

                records.append(
                    json.loads(
                        line
                    )
                )

            except json.JSONDecodeError as exc:

                raise RuntimeError(
                    f"Invalid JSON "
                    f"{path}, "
                    f"line {line_number}"
                ) from exc

    return records


def write_json(
    path,
    data,
):

    with path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as file:

        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )

        file.write(
            "\n"
        )


def finite(
    value,
    label,
):

    value = float(
        value
    )

    if not math.isfinite(
        value
    ):

        raise RuntimeError(
            f"Non-finite value: "
            f"{label}"
        )

    return value


# ============================================================
# 5. HOLM CORRECTION
# ============================================================

def holm_adjust(
    p_values,
):

    n = len(
        p_values
    )

    indexed = list(
        enumerate(
            p_values
        )
    )

    indexed.sort(
        key=lambda x:
            x[1]
    )

    output = [
        0.0
    ] * n

    previous = 0.0

    for (
        position,
        (
            original_index,
            p_value,
        ),
    ) in enumerate(
        indexed
    ):

        multiplier = (
            n - position
        )

        adjusted = (
            p_value
            *
            multiplier
        )

        adjusted = max(
            adjusted,
            previous,
        )

        adjusted = min(
            adjusted,
            1.0,
        )

        output[
            original_index
        ] = adjusted

        previous = adjusted

    return output


# ============================================================
# 6. DESCRIPTIVE
# ============================================================

def describe(
    values,
):

    values = np.asarray(
        values,
        dtype=float,
    )

    n = len(
        values
    )

    if n == 0:

        return {
            "n": 0,
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "max": None,
        }

    if n == 1:

        std = None

    else:

        std = float(
            np.std(
                values,
                ddof=1,
            )
        )

    return {
        "n":
            int(n),

        "mean":
            float(
                np.mean(
                    values
                )
            ),

        "median":
            float(
                np.median(
                    values
                )
            ),

        "std":
            std,

        "min":
            float(
                np.min(
                    values
                )
            ),

        "max":
            float(
                np.max(
                    values
                )
            ),
    }


# ============================================================
# 7. SAFE FRIEDMAN
# ============================================================

def safe_friedman(
    arrays,
):

    arrays = [
        np.asarray(
            array,
            dtype=float,
        )
        for array
        in arrays
    ]

    # --------------------------------------------------------
    # If every paired observation is exactly identical
    # across all three methods:
    # chi2 = 0, p = 1.
    # --------------------------------------------------------

    all_equal = True

    for index in range(
        len(
            arrays[0]
        )
    ):

        values = {
            arrays[
                method_index
            ][
                index
            ]

            for method_index
            in range(
                len(
                    arrays
                )
            )
        }

        if len(values) != 1:

            all_equal = False
            break

    if all_equal:

        return (
            0.0,
            1.0,
            0.0,
        )

    result = (
        stats.friedmanchisquare(
            *arrays
        )
    )

    statistic = finite(
        result.statistic,
        "friedman statistic",
    )

    p_value = finite(
        result.pvalue,
        "friedman p",
    )

    n = len(
        arrays[0]
    )

    k = len(
        arrays
    )

    kendalls_w = (
        statistic
        /
        (
            n
            *
            (
                k - 1
            )
        )
    )

    return (
        statistic,
        p_value,
        float(
            kendalls_w
        ),
    )


# ============================================================
# 8. VALIDATE MASTER
# ============================================================

def build_query_index(
    rows,
):

    index = defaultdict(
        dict
    )

    for row in rows:

        query_id = (
            row[
                "query_id"
            ]
        )

        method = (
            row[
                "method"
            ]
        )

        if (
            method
            in
            index[
                query_id
            ]
        ):

            raise RuntimeError(
                f"Duplicate "
                f"{query_id} × "
                f"{method}"
            )

        index[
            query_id
        ][
            method
        ] = row

    if (
        len(
            index
        )
        !=
        EXPECTED_QUERIES
    ):

        raise RuntimeError(
            "Query count bukan 40."
        )

    for query_id in sorted(
        index.keys()
    ):

        if (
            set(
                index[
                    query_id
                ].keys()
            )
            !=
            set(
                METHODS
            )
        ):

            raise RuntimeError(
                f"{query_id} tidak "
                "memiliki tiga method."
            )

    return index


# ============================================================
# 9. STRATUM ANALYSIS
# ============================================================

def analyze_stratum(
    rows,
    query_index,
    stratum_field,
    levels,
):

    summaries = []

    friedman_records = []

    for level in levels:

        query_ids = sorted(
            {
                row[
                    "query_id"
                ]

                for row in rows

                if (
                    row[
                        stratum_field
                    ]
                    ==
                    level
                )
            }
        )

        # ----------------------------------------------------
        # All three method rows must be present.
        # ----------------------------------------------------

        for query_id in query_ids:

            if (
                query_index[
                    query_id
                ][
                    METHODS[0]
                ][
                    stratum_field
                ]
                !=
                level
            ):

                raise RuntimeError(
                    f"{query_id}: "
                    f"{stratum_field} mismatch."
                )

        # ====================================================
        # DESCRIPTIVE METHOD SUMMARY
        # ====================================================

        for method in METHODS:

            method_rows = [
                query_index[
                    query_id
                ][
                    method
                ]

                for query_id
                in query_ids
            ]

            summary = {
                "stratum":
                    stratum_field,

                "level":
                    level,

                "method":
                    method,

                "n_queries":
                    len(
                        method_rows
                    ),

                "hit_at1":
                    float(
                        np.mean(
                            [
                                row[
                                    "hit_at1"
                                ]
                                for row
                                in method_rows
                            ]
                        )
                    ),

                "hit_at3":
                    float(
                        np.mean(
                            [
                                row[
                                    "hit_at3"
                                ]
                                for row
                                in method_rows
                            ]
                        )
                    ),

                "hit_at5":
                    float(
                        np.mean(
                            [
                                row[
                                    "hit_at5"
                                ]
                                for row
                                in method_rows
                            ]
                        )
                    ),

                "mrr_at5":
                    float(
                        np.mean(
                            [
                                row[
                                    "reciprocal_rank_at5"
                                ]
                                for row
                                in method_rows
                            ]
                        )
                    ),
            }

            for metric in RAG_METRICS:

                values = [
                    finite(
                        row[
                            metric
                        ],
                        (
                            f"{level}/"
                            f"{method}/"
                            f"{metric}"
                        ),
                    )

                    for row
                    in method_rows
                ]

                descriptive = (
                    describe(
                        values
                    )
                )

                summary[
                    f"{metric}_mean"
                ] = (
                    descriptive[
                        "mean"
                    ]
                )

                summary[
                    f"{metric}_median"
                ] = (
                    descriptive[
                        "median"
                    ]
                )

                summary[
                    f"{metric}_std"
                ] = (
                    descriptive[
                        "std"
                    ]
                )

            summaries.append(
                summary
            )

        # ====================================================
        # RAG FRIEDMAN WITHIN STRATUM
        # ====================================================

        for metric in RAG_METRICS:

            arrays = []

            for method in METHODS:

                arrays.append(
                    [
                        finite(
                            query_index[
                                query_id
                            ][
                                method
                            ][
                                metric
                            ],
                            (
                                f"{query_id}/"
                                f"{method}/"
                                f"{metric}"
                            ),
                        )

                        for query_id
                        in query_ids
                    ]
                )

            (
                statistic,
                p_value,
                kendalls_w,
            ) = safe_friedman(
                arrays
            )

            friedman_records.append(
                {
                    "stratum":
                        stratum_field,

                    "level":
                        level,

                    "metric":
                        metric,

                    "n_queries":
                        len(
                            query_ids
                        ),

                    "chi_square":
                        statistic,

                    "df":
                        len(
                            METHODS
                        )
                        - 1,

                    "p_raw":
                        p_value,

                    "kendalls_w":
                        kendalls_w,
                }
            )

    # ========================================================
    # HOLM:
    #
    # Across strata within each metric.
    #
    # Category:
    # 4 omnibus tests per metric.
    #
    # Domain:
    # 3 omnibus tests per metric.
    # ========================================================

    for metric in RAG_METRICS:

        relevant_records = [
            record
            for record
            in friedman_records
            if (
                record[
                    "metric"
                ]
                ==
                metric
            )
        ]

        adjusted = holm_adjust(
            [
                record[
                    "p_raw"
                ]

                for record
                in relevant_records
            ]
        )

        for (
            record,
            adjusted_p,
        ) in zip(
            relevant_records,
            adjusted,
        ):

            record[
                "p_holm"
            ] = float(
                adjusted_p
            )

            record[
                "significant_raw"
            ] = bool(
                record[
                    "p_raw"
                ]
                <
                ALPHA
            )

            record[
                "significant_holm"
            ] = bool(
                adjusted_p
                <
                ALPHA
            )

    return (
        summaries,
        friedman_records,
    )


# ============================================================
# 10. TARGET-IN-TOP5 ANALYSIS
#
# IMPORTANT:
#
# Only 2-3 failures per method.
# Therefore this is descriptive, not a primary
# inferential comparison.
# ============================================================

def analyze_target_top5(
    rows,
):

    records = []

    for method in METHODS:

        method_rows = [
            row
            for row
            in rows
            if (
                row[
                    "method"
                ]
                ==
                method
            )
        ]

        for metric in RAG_METRICS:

            hit_values = [
                finite(
                    row[
                        metric
                    ],
                    (
                        f"{method}/"
                        f"{metric}/hit"
                    ),
                )

                for row
                in method_rows

                if (
                    row[
                        "target_in_top5"
                    ]
                )
            ]

            miss_values = [
                finite(
                    row[
                        metric
                    ],
                    (
                        f"{method}/"
                        f"{metric}/miss"
                    ),
                )

                for row
                in method_rows

                if not (
                    row[
                        "target_in_top5"
                    ]
                )
            ]

            hit_desc = describe(
                hit_values
            )

            miss_desc = describe(
                miss_values
            )

            delta = (
                hit_desc[
                    "mean"
                ]
                -
                miss_desc[
                    "mean"
                ]
            )

            records.append(
                {
                    "method":
                        method,

                    "metric":
                        metric,

                    "n_target_in_top5":
                        hit_desc[
                            "n"
                        ],

                    "mean_target_in_top5":
                        hit_desc[
                            "mean"
                        ],

                    "median_target_in_top5":
                        hit_desc[
                            "median"
                        ],

                    "n_target_not_in_top5":
                        miss_desc[
                            "n"
                        ],

                    "mean_target_not_in_top5":
                        miss_desc[
                            "mean"
                        ],

                    "median_target_not_in_top5":
                        miss_desc[
                            "median"
                        ],

                    "mean_difference_in_minus_out":
                        float(
                            delta
                        ),
                }
            )

    return records


# ============================================================
# 11. RANK PROFILE
# ============================================================

def build_rank_profile(
    rows,
):

    records = []

    for method in METHODS:

        for metric in RAG_METRICS:

            for rank in range(
                1,
                6,
            ):

                values = [
                    finite(
                        row[
                            metric
                        ],
                        (
                            f"{method}/"
                            f"{metric}/rank{rank}"
                        ),
                    )

                    for row
                    in rows

                    if (
                        row[
                            "method"
                        ]
                        ==
                        method
                    )
                    and
                    (
                        row[
                            "target_rank_top5"
                        ]
                        ==
                        rank
                    )
                ]

                desc = describe(
                    values
                )

                records.append(
                    {
                        "method":
                            method,

                        "metric":
                            metric,

                        "target_rank":
                            rank,

                        "n":
                            desc[
                                "n"
                            ],

                        "mean":
                            desc[
                                "mean"
                            ],

                        "median":
                            desc[
                                "median"
                            ],

                        "std":
                            desc[
                                "std"
                            ],
                    }
                )

    return records


# ============================================================
# 12. SPEARMAN RANK SENSITIVITY
#
# Only observations where target is in Top-5.
#
# Rank:
# 1 = better retrieval position
# 5 = worse retrieval position
#
# Negative rho would indicate that worse numerical rank
# tends to correspond to lower RAG quality.
# ============================================================

def analyze_rank_spearman(
    rows,
):

    records = []

    for metric in RAG_METRICS:

        metric_records = []

        p_values = []

        for method in METHODS:

            selected = [
                row
                for row
                in rows

                if (
                    row[
                        "method"
                    ]
                    ==
                    method
                )
                and
                (
                    row[
                        "target_in_top5"
                    ]
                )
            ]

            ranks = np.asarray(
                [
                    row[
                        "target_rank_top5"
                    ]
                    for row
                    in selected
                ],
                dtype=float,
            )

            values = np.asarray(
                [
                    finite(
                        row[
                            metric
                        ],
                        (
                            f"{method}/"
                            f"{metric}"
                        ),
                    )
                    for row
                    in selected
                ],
                dtype=float,
            )

            constant_input = (
                len(
                    set(
                        ranks.tolist()
                    )
                )
                <
                2
                or
                len(
                    set(
                        values.tolist()
                    )
                )
                <
                2
            )

            if constant_input:

                rho = 0.0
                p_value = 1.0

            else:

                result = stats.spearmanr(
                    ranks,
                    values,
                )

                rho = finite(
                    result.statistic,
                    (
                        f"{method}/"
                        f"{metric}/rho"
                    ),
                )

                p_value = finite(
                    result.pvalue,
                    (
                        f"{method}/"
                        f"{metric}/p"
                    ),
                )

            record = {
                "metric":
                    metric,

                "method":
                    method,

                "n":
                    len(
                        selected
                    ),

                "spearman_rho":
                    float(
                        rho
                    ),

                "p_raw":
                    float(
                        p_value
                    ),

                "constant_input":
                    bool(
                        constant_input
                    ),
            }

            metric_records.append(
                record
            )

            p_values.append(
                p_value
            )

        adjusted = holm_adjust(
            p_values
        )

        for (
            record,
            adjusted_p,
        ) in zip(
            metric_records,
            adjusted,
        ):

            record[
                "p_holm"
            ] = float(
                adjusted_p
            )

            record[
                "significant_raw"
            ] = bool(
                record[
                    "p_raw"
                ]
                <
                ALPHA
            )

            record[
                "significant_holm"
            ] = bool(
                adjusted_p
                <
                ALPHA
            )

            records.append(
                record
            )

    return records


# ============================================================
# 13. CSV WRITER
# ============================================================

def write_csv(
    path,
    records,
):

    if not records:

        raise RuntimeError(
            f"Tidak ada records "
            f"untuk {path}"
        )

    fields = list(
        records[
            0
        ].keys()
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fields,
            lineterminator="\n",
        )

        writer.writeheader()

        writer.writerows(
            records
        )


# ============================================================
# 14. MAIN
# ============================================================

def main():

    print(
        "=" * 78
    )

    print(
        "RAG STRATIFIED ANALYSIS V1"
    )

    print(
        "=" * 78
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # SOURCE AUDIT
    # ========================================================

    for path in [
        MASTER_JSONL_FILE,
        MASTER_CSV_FILE,
        MASTER_MANIFEST_FILE,
    ]:

        if not path.exists():

            raise FileNotFoundError(
                f"Missing: {path}"
            )

    jsonl_hash = sha256_file(
        MASTER_JSONL_FILE
    )

    csv_hash = sha256_file(
        MASTER_CSV_FILE
    )

    manifest_hash = sha256_file(
        MASTER_MANIFEST_FILE
    )

    print(
        "\nSOURCE HASH AUDIT"
    )

    print(
        f"master_jsonl    : "
        f"{jsonl_hash}"
    )

    print(
        f"master_csv      : "
        f"{csv_hash}"
    )

    print(
        f"master_manifest : "
        f"{manifest_hash}"
    )

    if (
        jsonl_hash
        !=
        EXPECTED_MASTER_JSONL_SHA256
    ):

        raise RuntimeError(
            "Master JSONL hash berubah."
        )

    if (
        csv_hash
        !=
        EXPECTED_MASTER_CSV_SHA256
    ):

        raise RuntimeError(
            "Master CSV hash berubah."
        )

    if (
        manifest_hash
        !=
        EXPECTED_MASTER_MANIFEST_SHA256
    ):

        raise RuntimeError(
            "Master manifest hash berubah."
        )

    master_manifest = load_json(
        MASTER_MANIFEST_FILE
    )

    if (
        master_manifest.get(
            "status"
        )
        !=
        "FROZEN"
    ):

        raise RuntimeError(
            "Master belum FROZEN."
        )

    rows = load_jsonl(
        MASTER_JSONL_FILE
    )

    if (
        len(
            rows
        )
        !=
        EXPECTED_ROWS
    ):

        raise RuntimeError(
            "Master bukan 120 rows."
        )

    query_index = (
        build_query_index(
            rows
        )
    )

    print(
        "\n✅ Frozen master valid."
    )

    print(
        "✅ 120 rows / 40 paired queries valid."
    )

    # ========================================================
    # CATEGORY
    # ========================================================

    (
        category_summary,
        category_friedman,
    ) = analyze_stratum(
        rows=rows,
        query_index=query_index,
        stratum_field="category",
        levels=CATEGORIES,
    )

    # ========================================================
    # DOMAIN
    # ========================================================

    (
        domain_summary,
        domain_friedman,
    ) = analyze_stratum(
        rows=rows,
        query_index=query_index,
        stratum_field="domain",
        levels=DOMAINS,
    )

    # ========================================================
    # TARGET TOP-5
    # ========================================================

    target_top5 = (
        analyze_target_top5(
            rows
        )
    )

    # ========================================================
    # RANK PROFILE
    # ========================================================

    rank_profile = (
        build_rank_profile(
            rows
        )
    )

    # ========================================================
    # RANK CORRELATION
    # ========================================================

    rank_spearman = (
        analyze_rank_spearman(
            rows
        )
    )

    # ========================================================
    # WRITE CSV
    # ========================================================

    write_csv(
        CATEGORY_SUMMARY_CSV,
        category_summary,
    )

    write_csv(
        CATEGORY_FRIEDMAN_CSV,
        category_friedman,
    )

    write_csv(
        DOMAIN_SUMMARY_CSV,
        domain_summary,
    )

    write_csv(
        DOMAIN_FRIEDMAN_CSV,
        domain_friedman,
    )

    write_csv(
        TARGET_TOP5_CSV,
        target_top5,
    )

    write_csv(
        RANK_PROFILE_CSV,
        rank_profile,
    )

    write_csv(
        RANK_SPEARMAN_CSV,
        rank_spearman,
    )

    # ========================================================
    # SUMMARY JSON
    # ========================================================

    summary = {
        "artifact":
            (
                "Aire Optima RAG "
                "Stratified Analysis v1"
            ),

        "status":
            "COMPLETE_NOT_YET_FROZEN",

        "source_master_sha256":
            jsonl_hash,

        "alpha":
            ALPHA,

        "category": {
            "summary":
                category_summary,

            "friedman":
                category_friedman,
        },

        "domain": {
            "summary":
                domain_summary,

            "friedman":
                domain_friedman,
        },

        "target_top5":
            target_top5,

        "rank_profile":
            rank_profile,

        "rank_spearman":
            rank_spearman,

        "interpretation_policy": {
            "category_domain_tests":
                (
                    "Friedman repeated-measures tests "
                    "within each predefined category/domain. "
                    "Holm correction is applied across strata "
                    "within each metric family."
                ),

            "target_top5":
                (
                    "Descriptive only because the number "
                    "of target-not-in-Top5 observations is "
                    "very small (2-3 per retrieval method)."
                ),

            "rank_sensitivity":
                (
                    "Exploratory Spearman correlation using "
                    "only observations where the relevant "
                    "document appears within Top-5. "
                    "No artificial rank is assigned to "
                    "Top-5 failures."
                ),
        },
    }

    write_json(
        SUMMARY_JSON,
        summary,
    )

    # ========================================================
    # HASH OUTPUTS
    # ========================================================

    output_files = {
        "category_summary_csv":
            CATEGORY_SUMMARY_CSV,

        "category_friedman_csv":
            CATEGORY_FRIEDMAN_CSV,

        "domain_summary_csv":
            DOMAIN_SUMMARY_CSV,

        "domain_friedman_csv":
            DOMAIN_FRIEDMAN_CSV,

        "target_top5_csv":
            TARGET_TOP5_CSV,

        "rank_profile_csv":
            RANK_PROFILE_CSV,

        "rank_spearman_csv":
            RANK_SPEARMAN_CSV,

        "summary_json":
            SUMMARY_JSON,
    }

    output_hashes = {
        name:
            sha256_file(
                path
            )

        for (
            name,
            path,
        )
        in output_files.items()
    }

    # ========================================================
    # FREEZE MANIFEST
    # ========================================================

    manifest = {
        "artifact":
            (
                "Aire Optima RAG "
                "Stratified Analysis v1"
            ),

        "status":
            "FROZEN",

        "source": {
            "master_jsonl_sha256":
                jsonl_hash,

            "master_csv_sha256":
                csv_hash,

            "master_manifest_sha256":
                manifest_hash,
        },

        "environment": {
            "numpy":
                np.__version__,

            "scipy":
                scipy.__version__,
        },

        "analysis": {
            "categories":
                CATEGORIES,

            "domains":
                DOMAINS,

            "methods":
                METHODS,

            "rag_metrics":
                RAG_METRICS,

            "alpha":
                ALPHA,

            "target_top5_analysis":
                "descriptive",

            "rank_analysis":
                (
                    "Spearman correlation, "
                    "Top-5 hits only"
                ),
        },

        "outputs":
            output_hashes,

        "freeze_rule":
            (
                "All downstream interpretation and "
                "case analysis must use these frozen "
                "outputs or the frozen master table."
            ),
    }

    write_json(
        MANIFEST_FILE,
        manifest,
    )

    final_manifest_hash = (
        sha256_file(
            MANIFEST_FILE
        )
    )

    # ========================================================
    # PRINT CATEGORY
    # ========================================================

    print(
        "\n"
        + "=" * 78
    )

    print(
        "CATEGORY ANALYSIS"
    )

    print(
        "=" * 78
    )

    for category in CATEGORIES:

        print(
            f"\n{category.upper()}"
        )

        for record in (
            category_summary
        ):

            if (
                record[
                    "level"
                ]
                !=
                category
            ):

                continue

            print(
                f"{record['method']:<12} "
                f"H1={record['hit_at1']:.3f} "
                f"H3={record['hit_at3']:.3f} "
                f"H5={record['hit_at5']:.3f} "
                f"MRR5={record['mrr_at5']:.3f} "
                f"| F={record['faithfulness_mean']:.3f} "
                f"R={record['answer_relevancy_mean']:.3f} "
                f"A={record['answer_accuracy_mean']:.3f}"
            )

    print(
        "\nCATEGORY FRIEDMAN"
    )

    for record in (
        category_friedman
    ):

        print(
            f"{record['level']:<18} "
            f"{record['metric']:<18} "
            f"p={record['p_raw']:.6f} "
            f"Holm={record['p_holm']:.6f} "
            f"W={record['kendalls_w']:.4f} "
            f"sig={record['significant_holm']}"
        )

    # ========================================================
    # PRINT DOMAIN
    # ========================================================

    print(
        "\n"
        + "=" * 78
    )

    print(
        "DOMAIN ANALYSIS"
    )

    print(
        "=" * 78
    )

    for domain in DOMAINS:

        print(
            f"\n{domain.upper()}"
        )

        for record in (
            domain_summary
        ):

            if (
                record[
                    "level"
                ]
                !=
                domain
            ):

                continue

            print(
                f"{record['method']:<12} "
                f"H1={record['hit_at1']:.3f} "
                f"H3={record['hit_at3']:.3f} "
                f"H5={record['hit_at5']:.3f} "
                f"MRR5={record['mrr_at5']:.3f} "
                f"| F={record['faithfulness_mean']:.3f} "
                f"R={record['answer_relevancy_mean']:.3f} "
                f"A={record['answer_accuracy_mean']:.3f}"
            )

    print(
        "\nDOMAIN FRIEDMAN"
    )

    for record in (
        domain_friedman
    ):

        print(
            f"{record['level']:<15} "
            f"{record['metric']:<18} "
            f"p={record['p_raw']:.6f} "
            f"Holm={record['p_holm']:.6f} "
            f"W={record['kendalls_w']:.4f} "
            f"sig={record['significant_holm']}"
        )

    # ========================================================
    # TARGET TOP5
    # ========================================================

    print(
        "\n"
        + "=" * 78
    )

    print(
        "TARGET-IN-TOP5 ANALYSIS"
    )

    print(
        "=" * 78
    )

    for record in (
        target_top5
    ):

        print(
            f"{record['method']:<12} "
            f"{record['metric']:<18} "
            f"| IN n="
            f"{record['n_target_in_top5']:>2} "
            f"mean="
            f"{record['mean_target_in_top5']:.4f} "
            f"| OUT n="
            f"{record['n_target_not_in_top5']:>2} "
            f"mean="
            f"{record['mean_target_not_in_top5']:.4f} "
            f"| Δ="
            f"{record['mean_difference_in_minus_out']:+.4f}"
        )

    # ========================================================
    # SPEARMAN
    # ========================================================

    print(
        "\n"
        + "=" * 78
    )

    print(
        "TARGET RANK SENSITIVITY"
    )

    print(
        "=" * 78
    )

    for record in (
        rank_spearman
    ):

        print(
            f"{record['method']:<12} "
            f"{record['metric']:<18} "
            f"n={record['n']:>2} "
            f"rho={record['spearman_rho']:+.4f} "
            f"p={record['p_raw']:.6f} "
            f"Holm={record['p_holm']:.6f} "
            f"sig={record['significant_holm']}"
        )

    # ========================================================
    # FINAL
    # ========================================================

    print(
        "\n"
        + "=" * 78
    )

    print(
        "STRATIFIED ANALYSIS FROZEN"
    )

    print(
        "=" * 78
    )

    print(
        "\nOUTPUT HASHES"
    )

    for (
        name,
        file_hash,
    ) in output_hashes.items():

        print(
            f"{name:<25}: "
            f"{file_hash}"
        )

    print(
        "\nMANIFEST SHA256:"
    )

    print(
        final_manifest_hash
    )

    print(
        "\nSTATUS: FROZEN"
    )


if __name__ == "__main__":
    main()