import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
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
    "benchmark/results/analysis/statistics"
)

DESCRIPTIVE_JSON_FILE = (
    OUTPUT_DIR
    / "rag_descriptive_statistics_v1.json"
)

DESCRIPTIVE_CSV_FILE = (
    OUTPUT_DIR
    / "rag_descriptive_statistics_v1.csv"
)

FRIEDMAN_JSON_FILE = (
    OUTPUT_DIR
    / "rag_friedman_test_v1.json"
)

FRIEDMAN_CSV_FILE = (
    OUTPUT_DIR
    / "rag_friedman_test_v1.csv"
)

PAIRWISE_JSON_FILE = (
    OUTPUT_DIR
    / "rag_pairwise_wilcoxon_v1.json"
)

PAIRWISE_CSV_FILE = (
    OUTPUT_DIR
    / "rag_pairwise_wilcoxon_v1.csv"
)

SUMMARY_JSON_FILE = (
    OUTPUT_DIR
    / "rag_statistical_analysis_summary_v1.json"
)

MANIFEST_FILE = (
    OUTPUT_DIR
    / "rag_statistical_analysis_manifest_v1.json"
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
# 3. EXPERIMENT CONSTANTS
# ============================================================

EXPECTED_ROWS = 120
EXPECTED_QUERY_COUNT = 40

ALPHA = 0.05

METHODS = [
    "bm25",
    "e5",
    "hybrid_rrf",
]

METRICS = [
    "faithfulness",
    "answer_relevancy",
    "answer_accuracy",
]

PAIRWISE_COMPARISONS = [
    (
        "bm25",
        "e5",
    ),
    (
        "bm25",
        "hybrid_rrf",
    ),
    (
        "e5",
        "hybrid_rrf",
    ),
]


# ============================================================
# 4. HELPERS
# ============================================================

def sha256_file(
    path: Path,
) -> str:

    hasher = hashlib.sha256()

    with path.open(
        "rb"
    ) as file:

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


def load_json(
    path: Path,
):

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(
            file
        )


def load_jsonl(
    path: Path,
):

    rows = []

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

            line = (
                line.strip()
            )

            if not line:
                continue

            try:

                rows.append(
                    json.loads(
                        line
                    )
                )

            except json.JSONDecodeError as exc:

                raise RuntimeError(
                    f"Invalid JSON pada "
                    f"{path}, "
                    f"line {line_number}."
                ) from exc

    return rows


def write_json(
    path: Path,
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


def ensure_finite(
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
            f"{label} = {value}"
        )

    return value


# ============================================================
# 5. HOLM CORRECTION
#
# Family = 3 pairwise comparisons
# within each RAG metric.
# ============================================================

def holm_adjust(
    p_values,
):

    count = len(
        p_values
    )

    indexed = list(
        enumerate(
            p_values
        )
    )

    indexed.sort(
        key=lambda item:
            item[1]
    )

    adjusted = [
        None
    ] * count

    previous_adjusted = 0.0

    for (
        rank_zero_based,
        (
            original_index,
            p_value,
        ),
    ) in enumerate(
        indexed
    ):

        multiplier = (
            count
            -
            rank_zero_based
        )

        raw_adjusted = (
            multiplier
            *
            p_value
        )

        holm_value = max(
            previous_adjusted,
            raw_adjusted,
        )

        holm_value = min(
            1.0,
            holm_value,
        )

        adjusted[
            original_index
        ] = holm_value

        previous_adjusted = (
            holm_value
        )

    return adjusted


# ============================================================
# 6. RANK-BISERIAL EFFECT SIZE
#
# Positive:
# first method tends to score higher.
#
# Negative:
# second method tends to score higher.
#
# Zero differences are removed consistently with
# Wilcoxon zero_method="wilcox".
# ============================================================

def rank_biserial_effect_size(
    first_values,
    second_values,
):

    first_values = np.asarray(
        first_values,
        dtype=float,
    )

    second_values = np.asarray(
        second_values,
        dtype=float,
    )

    differences = (
        first_values
        -
        second_values
    )

    nonzero = (
        differences
        !=
        0
    )

    differences = (
        differences[
            nonzero
        ]
    )

    if (
        len(
            differences
        )
        ==
        0
    ):

        return {
            "rank_biserial":
                0.0,

            "w_plus":
                0.0,

            "w_minus":
                0.0,

            "n_nonzero":
                0,
        }

    absolute_ranks = (
        stats.rankdata(
            np.abs(
                differences
            ),
            method="average",
        )
    )

    w_plus = float(
        np.sum(
            absolute_ranks[
                differences > 0
            ]
        )
    )

    w_minus = float(
        np.sum(
            absolute_ranks[
                differences < 0
            ]
        )
    )

    denominator = (
        w_plus
        +
        w_minus
    )

    if (
        denominator
        ==
        0
    ):

        rank_biserial = (
            0.0
        )

    else:

        rank_biserial = (
            (
                w_plus
                -
                w_minus
            )
            /
            denominator
        )

    return {
        "rank_biserial":
            float(
                rank_biserial
            ),

        "w_plus":
            w_plus,

        "w_minus":
            w_minus,

        "n_nonzero":
            int(
                len(
                    differences
                )
            ),
    }


# ============================================================
# 7. DESCRIPTIVE STATISTICS
# ============================================================

def descriptive_statistics(
    values,
):

    values = np.asarray(
        values,
        dtype=float,
    )

    n = len(
        values
    )

    if (
        n
        <
        2
    ):

        raise RuntimeError(
            "Descriptive statistics "
            "membutuhkan minimal 2 values."
        )

    mean_value = float(
        np.mean(
            values
        )
    )

    median_value = float(
        np.median(
            values
        )
    )

    standard_deviation = float(
        np.std(
            values,
            ddof=1,
        )
    )

    standard_error = (
        standard_deviation
        /
        math.sqrt(
            n
        )
    )

    t_critical = float(
        stats.t.ppf(
            1.0
            -
            ALPHA / 2.0,
            df=n - 1,
        )
    )

    margin = (
        t_critical
        *
        standard_error
    )

    ci_low = (
        mean_value
        -
        margin
    )

    ci_high = (
        mean_value
        +
        margin
    )

    q1 = float(
        np.percentile(
            values,
            25,
        )
    )

    q3 = float(
        np.percentile(
            values,
            75,
        )
    )

    return {
        "n":
            int(
                n
            ),

        "mean":
            mean_value,

        "median":
            median_value,

        "std_sample":
            standard_deviation,

        "standard_error":
            float(
                standard_error
            ),

        "min":
            float(
                np.min(
                    values
                )
            ),

        "q1":
            q1,

        "q3":
            q3,

        "iqr":
            float(
                q3
                -
                q1
            ),

        "max":
            float(
                np.max(
                    values
                )
            ),

        "ci95_mean_low":
            float(
                ci_low
            ),

        "ci95_mean_high":
            float(
                ci_high
            ),
    }


# ============================================================
# 8. MAIN
# ============================================================

def main():

    print(
        "=" * 76
    )

    print(
        "RAG STATISTICAL ANALYSIS V1 "
        "- DESCRIPTIVE + PAIRED TESTS"
    )

    print(
        "=" * 76
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # SOURCE FILES
    # ========================================================

    for path in [
        MASTER_JSONL_FILE,
        MASTER_CSV_FILE,
        MASTER_MANIFEST_FILE,
    ]:

        if not path.exists():

            raise FileNotFoundError(
                f"File tidak ditemukan: "
                f"{path}"
            )

    # ========================================================
    # HASH AUDIT
    # ========================================================

    master_jsonl_hash = (
        sha256_file(
            MASTER_JSONL_FILE
        )
    )

    master_csv_hash = (
        sha256_file(
            MASTER_CSV_FILE
        )
    )

    master_manifest_hash = (
        sha256_file(
            MASTER_MANIFEST_FILE
        )
    )

    print(
        "\nSOURCE HASH AUDIT"
    )

    print(
        f"master_jsonl     : "
        f"{master_jsonl_hash}"
    )

    print(
        f"master_csv       : "
        f"{master_csv_hash}"
    )

    print(
        f"master_manifest  : "
        f"{master_manifest_hash}"
    )

    if (
        master_jsonl_hash
        !=
        EXPECTED_MASTER_JSONL_SHA256
    ):

        raise RuntimeError(
            "Master JSONL hash berubah."
        )

    if (
        master_csv_hash
        !=
        EXPECTED_MASTER_CSV_SHA256
    ):

        raise RuntimeError(
            "Master CSV hash berubah."
        )

    if (
        master_manifest_hash
        !=
        EXPECTED_MASTER_MANIFEST_SHA256
    ):

        raise RuntimeError(
            "Master manifest hash berubah."
        )

    master_manifest = (
        load_json(
            MASTER_MANIFEST_FILE
        )
    )

    if (
        master_manifest.get(
            "status"
        )
        !=
        "FROZEN"
    ):

        raise RuntimeError(
            "Master analysis "
            "belum berstatus FROZEN."
        )

    print(
        "\n✅ Frozen master analysis valid."
    )

    # ========================================================
    # LOAD MASTER
    # ========================================================

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
            f"Master rows bukan "
            f"{EXPECTED_ROWS}."
        )

    # ========================================================
    # INDEX:
    #
    # query_id
    #   method
    #       row
    # ========================================================

    rows_by_query = defaultdict(
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
            not in METHODS
        ):

            raise RuntimeError(
                f"Unknown method: "
                f"{method}"
            )

        if (
            method
            in
            rows_by_query[
                query_id
            ]
        ):

            raise RuntimeError(
                f"Duplicate "
                f"{query_id} × "
                f"{method}"
            )

        rows_by_query[
            query_id
        ][
            method
        ] = row

    expected_query_ids = [
        f"Q{number:03d}"
        for number
        in range(
            1,
            EXPECTED_QUERY_COUNT + 1,
        )
    ]

    if (
        set(
            rows_by_query.keys()
        )
        !=
        set(
            expected_query_ids
        )
    ):

        raise RuntimeError(
            "Query set pada master "
            "tidak Q001-Q040."
        )

    # ========================================================
    # QUERY-LEVEL CONSISTENCY
    # ========================================================

    for query_id in (
        expected_query_ids
    ):

        method_rows = (
            rows_by_query[
                query_id
            ]
        )

        if (
            set(
                method_rows.keys()
            )
            !=
            set(
                METHODS
            )
        ):

            raise RuntimeError(
                f"{query_id} tidak "
                "memiliki 3 methods."
            )

        categories = {
            method_rows[
                method
            ][
                "category"
            ]

            for method
            in METHODS
        }

        domains = {
            method_rows[
                method
            ][
                "domain"
            ]

            for method
            in METHODS
        }

        questions = {
            method_rows[
                method
            ][
                "question"
            ]

            for method
            in METHODS
        }

        relevant_ids = {
            method_rows[
                method
            ][
                "relevant_id"
            ]

            for method
            in METHODS
        }

        if (
            len(
                categories
            )
            !=
            1
        ):

            raise RuntimeError(
                f"{query_id}: "
                "category berbeda "
                "antar-method."
            )

        if (
            len(
                domains
            )
            !=
            1
        ):

            raise RuntimeError(
                f"{query_id}: "
                "domain berbeda "
                "antar-method."
            )

        if (
            len(
                questions
            )
            !=
            1
        ):

            raise RuntimeError(
                f"{query_id}: "
                "question berbeda "
                "antar-method."
            )

        if (
            len(
                relevant_ids
            )
            !=
            1
        ):

            raise RuntimeError(
                f"{query_id}: "
                "relevant_id berbeda "
                "antar-method."
            )

    print(
        "✅ 40 paired queries valid."
    )

    # ========================================================
    # BUILD ARRAYS
    #
    # data[metric][method] = 40 values ordered Q001-Q040
    # ========================================================

    data = {
        metric: {
            method: []
            for method
            in METHODS
        }
        for metric
        in METRICS
    }

    for query_id in (
        expected_query_ids
    ):

        for method in METHODS:

            row = (
                rows_by_query[
                    query_id
                ][
                    method
                ]
            )

            for metric in METRICS:

                value = ensure_finite(
                    row[
                        metric
                    ],
                    (
                        f"{query_id}/"
                        f"{method}/"
                        f"{metric}"
                    ),
                )

                data[
                    metric
                ][
                    method
                ].append(
                    value
                )

    # ========================================================
    # DESCRIPTIVE ANALYSIS
    # ========================================================

    descriptive_records = []

    for metric in METRICS:

        for method in METHODS:

            result = (
                descriptive_statistics(
                    data[
                        metric
                    ][
                        method
                    ]
                )
            )

            record = {
                "metric":
                    metric,

                "method":
                    method,

                **result,
            }

            descriptive_records.append(
                record
            )

    # ========================================================
    # FRIEDMAN OMNIBUS TEST
    #
    # H0:
    # score distributions/ranks across the
    # three paired retrieval methods are equal.
    #
    # Kendall W:
    #
    # W = chi-square / [n(k-1)]
    # ========================================================

    friedman_records = []

    friedman_by_metric = {}

    for metric in METRICS:

        arrays = [
            np.asarray(
                data[
                    metric
                ][
                    method
                ],
                dtype=float,
            )
            for method
            in METHODS
        ]

        result = (
            stats.friedmanchisquare(
                *arrays
            )
        )

        statistic = ensure_finite(
            result.statistic,
            f"{metric}/friedman/statistic",
        )

        p_value = ensure_finite(
            result.pvalue,
            f"{metric}/friedman/p",
        )

        n = (
            EXPECTED_QUERY_COUNT
        )

        k = len(
            METHODS
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

        kendalls_w = ensure_finite(
            kendalls_w,
            (
                f"{metric}/"
                "friedman/kendalls_w"
            ),
        )

        record = {
            "metric":
                metric,

            "n":
                n,

            "k":
                k,

            "statistic_chi_square":
                statistic,

            "degrees_of_freedom":
                k - 1,

            "p_value":
                p_value,

            "alpha":
                ALPHA,

            "significant":
                bool(
                    p_value
                    <
                    ALPHA
                ),

            "kendalls_w":
                kendalls_w,
        }

        friedman_records.append(
            record
        )

        friedman_by_metric[
            metric
        ] = record

    # ========================================================
    # PAIRWISE WILCOXON
    #
    # method="approx" deliberately fixed for reproducibility
    # with ties / zero differences.
    #
    # Holm correction performed separately within each metric.
    # ========================================================

    pairwise_records = []

    for metric in METRICS:

        metric_records = []

        raw_p_values = []

        for (
            method_a,
            method_b,
        ) in PAIRWISE_COMPARISONS:

            values_a = np.asarray(
                data[
                    metric
                ][
                    method_a
                ],
                dtype=float,
            )

            values_b = np.asarray(
                data[
                    metric
                ][
                    method_b
                ],
                dtype=float,
            )

            differences = (
                values_a
                -
                values_b
            )

            wins_a = int(
                np.sum(
                    differences > 0
                )
            )

            wins_b = int(
                np.sum(
                    differences < 0
                )
            )

            ties = int(
                np.sum(
                    differences == 0
                )
            )

            mean_difference = float(
                np.mean(
                    differences
                )
            )

            median_difference = float(
                np.median(
                    differences
                )
            )

            effect = (
                rank_biserial_effect_size(
                    values_a,
                    values_b,
                )
            )

            # ================================================
            # All paired observations exactly equal.
            # ================================================

            if (
                effect[
                    "n_nonzero"
                ]
                ==
                0
            ):

                wilcoxon_statistic = (
                    0.0
                )

                raw_p_value = (
                    1.0
                )

            else:

                wilcoxon_result = (
                    stats.wilcoxon(
                        values_a,
                        values_b,
                        zero_method="wilcox",
                        correction=False,
                        alternative="two-sided",
                        method="approx",
                    )
                )

                wilcoxon_statistic = (
                    ensure_finite(
                        wilcoxon_result.statistic,
                        (
                            f"{metric}/"
                            f"{method_a}/"
                            f"{method_b}/"
                            "wilcoxon_statistic"
                        ),
                    )
                )

                raw_p_value = (
                    ensure_finite(
                        wilcoxon_result.pvalue,
                        (
                            f"{metric}/"
                            f"{method_a}/"
                            f"{method_b}/"
                            "wilcoxon_p"
                        ),
                    )
                )

            raw_p_values.append(
                raw_p_value
            )

            metric_records.append(
                {
                    "metric":
                        metric,

                    "method_a":
                        method_a,

                    "method_b":
                        method_b,

                    "n_total":
                        EXPECTED_QUERY_COUNT,

                    "n_nonzero":
                        effect[
                            "n_nonzero"
                        ],

                    "wins_method_a":
                        wins_a,

                    "wins_method_b":
                        wins_b,

                    "ties":
                        ties,

                    "mean_method_a":
                        float(
                            np.mean(
                                values_a
                            )
                        ),

                    "mean_method_b":
                        float(
                            np.mean(
                                values_b
                            )
                        ),

                    "mean_difference_a_minus_b":
                        mean_difference,

                    "median_difference_a_minus_b":
                        median_difference,

                    "wilcoxon_statistic":
                        wilcoxon_statistic,

                    "raw_p_value":
                        raw_p_value,

                    "rank_biserial":
                        effect[
                            "rank_biserial"
                        ],

                    "w_plus":
                        effect[
                            "w_plus"
                        ],

                    "w_minus":
                        effect[
                            "w_minus"
                        ],
                }
            )

        # ====================================================
        # HOLM CORRECTION WITHIN THIS METRIC
        # ====================================================

        adjusted_p_values = (
            holm_adjust(
                raw_p_values
            )
        )

        for (
            record,
            adjusted_p,
        ) in zip(
            metric_records,
            adjusted_p_values,
        ):

            record[
                "holm_adjusted_p"
            ] = float(
                adjusted_p
            )

            record[
                "alpha"
            ] = ALPHA

            record[
                "significant_raw"
            ] = bool(
                record[
                    "raw_p_value"
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

            record[
                "friedman_omnibus_significant"
            ] = (
                friedman_by_metric[
                    metric
                ][
                    "significant"
                ]
            )

            pairwise_records.append(
                record
            )

    # ========================================================
    # WRITE DESCRIPTIVE JSON
    # ========================================================

    descriptive_json = {
        "artifact":
            "RAG Descriptive Statistics v1",

        "source_master_sha256":
            master_jsonl_hash,

        "alpha":
            ALPHA,

        "confidence_interval":
            (
                "Two-sided 95% Student-t "
                "confidence interval for the mean."
            ),

        "records":
            descriptive_records,
    }

    write_json(
        DESCRIPTIVE_JSON_FILE,
        descriptive_json,
    )

    # ========================================================
    # WRITE DESCRIPTIVE CSV
    # ========================================================

    descriptive_fields = [
        "metric",
        "method",
        "n",
        "mean",
        "median",
        "std_sample",
        "standard_error",
        "min",
        "q1",
        "q3",
        "iqr",
        "max",
        "ci95_mean_low",
        "ci95_mean_high",
    ]

    with DESCRIPTIVE_CSV_FILE.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=(
                descriptive_fields
            ),
            lineterminator="\n",
        )

        writer.writeheader()

        writer.writerows(
            descriptive_records
        )

    # ========================================================
    # WRITE FRIEDMAN
    # ========================================================

    friedman_json = {
        "artifact":
            "RAG Friedman Test v1",

        "source_master_sha256":
            master_jsonl_hash,

        "alpha":
            ALPHA,

        "method":
            (
                "Friedman repeated-measures "
                "non-parametric omnibus test."
            ),

        "effect_size":
            "Kendall's W",

        "records":
            friedman_records,
    }

    write_json(
        FRIEDMAN_JSON_FILE,
        friedman_json,
    )

    friedman_fields = [
        "metric",
        "n",
        "k",
        "statistic_chi_square",
        "degrees_of_freedom",
        "p_value",
        "alpha",
        "significant",
        "kendalls_w",
    ]

    with FRIEDMAN_CSV_FILE.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=(
                friedman_fields
            ),
            lineterminator="\n",
        )

        writer.writeheader()

        writer.writerows(
            friedman_records
        )

    # ========================================================
    # WRITE PAIRWISE
    # ========================================================

    pairwise_json = {
        "artifact":
            "RAG Pairwise Wilcoxon v1",

        "source_master_sha256":
            master_jsonl_hash,

        "alpha":
            ALPHA,

        "test":
            (
                "Two-sided paired Wilcoxon "
                "signed-rank test."
            ),

        "wilcoxon_zero_method":
            "wilcox",

        "wilcoxon_method":
            "approx",

        "multiple_comparison":
            (
                "Holm correction separately "
                "within each metric family "
                "of three pairwise tests."
            ),

        "effect_size":
            (
                "Matched-pairs rank-biserial "
                "correlation. Positive values "
                "favor method_a; negative values "
                "favor method_b."
            ),

        "records":
            pairwise_records,
    }

    write_json(
        PAIRWISE_JSON_FILE,
        pairwise_json,
    )

    pairwise_fields = [
        "metric",
        "method_a",
        "method_b",
        "n_total",
        "n_nonzero",
        "wins_method_a",
        "wins_method_b",
        "ties",
        "mean_method_a",
        "mean_method_b",
        "mean_difference_a_minus_b",
        "median_difference_a_minus_b",
        "wilcoxon_statistic",
        "raw_p_value",
        "holm_adjusted_p",
        "alpha",
        "significant_raw",
        "significant_holm",
        "friedman_omnibus_significant",
        "rank_biserial",
        "w_plus",
        "w_minus",
    ]

    with PAIRWISE_CSV_FILE.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=(
                pairwise_fields
            ),
            lineterminator="\n",
        )

        writer.writeheader()

        writer.writerows(
            pairwise_records
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    summary = {
        "artifact":
            (
                "Aire Optima RAG "
                "Statistical Analysis v1"
            ),

        "status":
            "COMPLETE_NOT_YET_FROZEN",

        "source_master": {
            "file":
                str(
                    MASTER_JSONL_FILE
                ),

            "sha256":
                master_jsonl_hash,
        },

        "design": {
            "query_count":
                EXPECTED_QUERY_COUNT,

            "methods":
                METHODS,

            "metrics":
                METRICS,

            "paired_repeated_measures":
                True,

            "alpha":
                ALPHA,
        },

        "descriptive":
            descriptive_records,

        "friedman":
            friedman_records,

        "pairwise_wilcoxon":
            pairwise_records,
    }

    write_json(
        SUMMARY_JSON_FILE,
        summary,
    )

    # ========================================================
    # OUTPUT HASHES
    # ========================================================

    output_hashes = {
        "descriptive_json":
            sha256_file(
                DESCRIPTIVE_JSON_FILE
            ),

        "descriptive_csv":
            sha256_file(
                DESCRIPTIVE_CSV_FILE
            ),

        "friedman_json":
            sha256_file(
                FRIEDMAN_JSON_FILE
            ),

        "friedman_csv":
            sha256_file(
                FRIEDMAN_CSV_FILE
            ),

        "pairwise_json":
            sha256_file(
                PAIRWISE_JSON_FILE
            ),

        "pairwise_csv":
            sha256_file(
                PAIRWISE_CSV_FILE
            ),

        "summary_json":
            sha256_file(
                SUMMARY_JSON_FILE
            ),
    }

    # ========================================================
    # FREEZE MANIFEST
    # ========================================================

    manifest = {
        "artifact":
            (
                "Aire Optima RAG "
                "Statistical Analysis v1"
            ),

        "status":
            "FROZEN",

        "source": {
            "master_jsonl": {
                "file":
                    str(
                        MASTER_JSONL_FILE
                    ),

                "sha256":
                    master_jsonl_hash,
            },

            "master_csv": {
                "file":
                    str(
                        MASTER_CSV_FILE
                    ),

                "sha256":
                    master_csv_hash,
            },

            "master_manifest": {
                "file":
                    str(
                        MASTER_MANIFEST_FILE
                    ),

                "sha256":
                    master_manifest_hash,
            },
        },

        "analysis_environment": {
            "numpy":
                np.__version__,

            "scipy":
                scipy.__version__,
        },

        "analysis_policy": {
            "alpha":
                ALPHA,

            "descriptive_ci":
                (
                    "Two-sided 95% Student-t "
                    "CI around sample mean."
                ),

            "omnibus_test":
                "Friedman",

            "omnibus_effect_size":
                "Kendall's W",

            "pairwise_test":
                (
                    "Two-sided paired Wilcoxon "
                    "signed-rank"
                ),

            "wilcoxon_zero_method":
                "wilcox",

            "wilcoxon_method":
                "approx",

            "multiple_comparison":
                (
                    "Holm correction within "
                    "each metric family."
                ),

            "pairwise_effect_size":
                (
                    "Matched-pairs "
                    "rank-biserial correlation."
                ),
        },

        "outputs": {
            name: {
                "sha256":
                    file_hash
            }

            for (
                name,
                file_hash,
            )
            in output_hashes.items()
        },

        "freeze_rule":
            (
                "Do not modify the frozen statistical "
                "outputs. Subsequent category, domain, "
                "retrieval-success, and case analyses "
                "must use the frozen master analysis "
                "artifact and may cite these frozen "
                "overall statistical outputs."
            ),
    }

    write_json(
        MANIFEST_FILE,
        manifest,
    )

    manifest_hash = (
        sha256_file(
            MANIFEST_FILE
        )
    )

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print(
        "\n"
        + "=" * 76
    )

    print(
        "DESCRIPTIVE STATISTICS"
    )

    print(
        "=" * 76
    )

    for metric in METRICS:

        print(
            f"\n{metric.upper()}"
        )

        for record in (
            descriptive_records
        ):

            if (
                record[
                    "metric"
                ]
                !=
                metric
            ):

                continue

            print(
                f"{record['method']:<12} "
                f"mean={record['mean']:.6f} "
                f"median={record['median']:.6f} "
                f"SD={record['std_sample']:.6f} "
                f"95% CI="
                f"[{record['ci95_mean_low']:.6f}, "
                f"{record['ci95_mean_high']:.6f}]"
            )

    print(
        "\n"
        + "=" * 76
    )

    print(
        "FRIEDMAN OMNIBUS TEST"
    )

    print(
        "=" * 76
    )

    for record in (
        friedman_records
    ):

        print(
            f"{record['metric']:<20} "
            f"chi2={record['statistic_chi_square']:.6f} "
            f"p={record['p_value']:.8f} "
            f"W={record['kendalls_w']:.6f} "
            f"| significant="
            f"{record['significant']}"
        )

    print(
        "\n"
        + "=" * 76
    )

    print(
        "PAIRWISE WILCOXON + HOLM"
    )

    print(
        "=" * 76
    )

    current_metric = None

    for record in (
        pairwise_records
    ):

        if (
            record[
                "metric"
            ]
            !=
            current_metric
        ):

            current_metric = (
                record[
                    "metric"
                ]
            )

            print(
                f"\n{current_metric.upper()}"
            )

        print(
            f"{record['method_a']:<11} "
            f"vs "
            f"{record['method_b']:<11} "
            f"| raw p="
            f"{record['raw_p_value']:.8f} "
            f"| Holm p="
            f"{record['holm_adjusted_p']:.8f} "
            f"| r_rb="
            f"{record['rank_biserial']:.6f} "
            f"| wins="
            f"{record['wins_method_a']}:"
            f"{record['wins_method_b']} "
            f"| ties="
            f"{record['ties']} "
            f"| sig="
            f"{record['significant_holm']}"
        )

    print(
        "\n"
        + "=" * 76
    )

    print(
        "STATISTICAL ANALYSIS FROZEN"
    )

    print(
        "=" * 76
    )

    print(
        "\nOUTPUT HASHES"
    )

    for (
        name,
        file_hash,
    ) in output_hashes.items():

        print(
            f"{name:<20}: "
            f"{file_hash}"
        )

    print(
        "\nMANIFEST SHA256:"
    )

    print(
        manifest_hash
    )

    print(
        "\nMANIFEST:"
    )

    print(
        MANIFEST_FILE
    )

    print(
        "\nSTATUS: FROZEN"
    )


if __name__ == "__main__":
    main()