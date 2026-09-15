import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import scipy
from scipy import stats


# ============================================================
# PATHS
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
    "benchmark/results/analysis/"
    "retrieval_statistics"
)

OVERALL_CSV = (
    OUTPUT_DIR
    / "retrieval_overall_statistics_v1.csv"
)

COCHRAN_CSV = (
    OUTPUT_DIR
    / "retrieval_cochran_q_v1.csv"
)

MCNEMAR_CSV = (
    OUTPUT_DIR
    / "retrieval_pairwise_mcnemar_v1.csv"
)

MRR_FRIEDMAN_CSV = (
    OUTPUT_DIR
    / "retrieval_mrr_friedman_v1.csv"
)

MRR_WILCOXON_CSV = (
    OUTPUT_DIR
    / "retrieval_mrr_pairwise_wilcoxon_v1.csv"
)

SUMMARY_JSON = (
    OUTPUT_DIR
    / "retrieval_statistical_analysis_summary_v1.json"
)

MANIFEST_FILE = (
    OUTPUT_DIR
    / "retrieval_statistical_analysis_manifest_v1.json"
)


# ============================================================
# FROZEN HASHES
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
# CONSTANTS
# ============================================================

METHODS = [
    "bm25",
    "e5",
    "hybrid_rrf",
]

QUERY_COUNT = 40

ALPHA = 0.05


HIT_METRICS = [
    "hit_at1",
    "hit_at3",
    "hit_at5",
]


PAIRS = [
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
# HELPERS
# ============================================================

def sha256_file(
    path,
):

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
    path,
):

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(
            file
        )


def load_jsonl(
    path,
):

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

            line = (
                line.strip()
            )

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
                    f"Invalid JSON line "
                    f"{line_number}"
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


def write_csv(
    path,
    records,
):

    if not records:

        raise RuntimeError(
            f"No records: {path}"
        )

    fields = list(
        records[0].keys()
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


def holm_adjust(
    p_values,
):

    n = len(
        p_values
    )

    indexed = sorted(
        enumerate(
            p_values
        ),
        key=lambda item:
            item[1],
    )

    adjusted = [
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

        value = (
            (
                n - position
            )
            *
            p_value
        )

        value = max(
            previous,
            value,
        )

        value = min(
            1.0,
            value,
        )

        adjusted[
            original_index
        ] = value

        previous = value

    return adjusted


# ============================================================
# COCHRAN Q
#
# k related binary samples.
# ============================================================

def cochran_q(
    matrix,
):

    matrix = np.asarray(
        matrix,
        dtype=float,
    )

    if (
        matrix.ndim
        !=
        2
    ):

        raise RuntimeError(
            "Cochran matrix must be 2D."
        )

    n, k = (
        matrix.shape
    )

    column_totals = (
        np.sum(
            matrix,
            axis=0,
        )
    )

    row_totals = (
        np.sum(
            matrix,
            axis=1,
        )
    )

    numerator = (
        (
            k
            *
            (
                k - 1
            )
        )
        *
        (
            np.sum(
                column_totals ** 2
            )
            -
            (
                np.sum(
                    column_totals
                )
                ** 2
                /
                k
            )
        )
    )

    denominator = (
        (
            k
            *
            np.sum(
                row_totals
            )
        )
        -
        np.sum(
            row_totals ** 2
        )
    )

    if (
        denominator
        ==
        0
    ):

        return (
            0.0,
            1.0,
        )

    statistic = (
        numerator
        /
        denominator
    )

    p_value = float(
        stats.chi2.sf(
            statistic,
            df=k - 1,
        )
    )

    return (
        float(
            statistic
        ),
        p_value,
    )


# ============================================================
# EXACT MCNEMAR
#
# b = A success, B fail
# c = A fail, B success
#
# Under H0:
# b ~ Binomial(b+c, 0.5)
# ============================================================

def exact_mcnemar(
    first,
    second,
):

    first = np.asarray(
        first,
        dtype=int,
    )

    second = np.asarray(
        second,
        dtype=int,
    )

    b = int(
        np.sum(
            (
                first == 1
            )
            &
            (
                second == 0
            )
        )
    )

    c = int(
        np.sum(
            (
                first == 0
            )
            &
            (
                second == 1
            )
        )
    )

    discordant = (
        b + c
    )

    if discordant == 0:

        p_value = 1.0

    else:

        result = stats.binomtest(
            min(
                b,
                c,
            ),
            n=discordant,
            p=0.5,
            alternative="two-sided",
        )

        p_value = float(
            result.pvalue
        )

    return {
        "a_success_b_fail":
            b,

        "a_fail_b_success":
            c,

        "discordant":
            discordant,

        "exact_p_value":
            p_value,
    }


# ============================================================
# RANK-BISERIAL
# ============================================================

def rank_biserial(
    first,
    second,
):

    first = np.asarray(
        first,
        dtype=float,
    )

    second = np.asarray(
        second,
        dtype=float,
    )

    difference = (
        first
        -
        second
    )

    difference = (
        difference[
            difference != 0
        ]
    )

    if (
        len(
            difference
        )
        ==
        0
    ):

        return {
            "rank_biserial":
                0.0,

            "n_nonzero":
                0,

            "w_plus":
                0.0,

            "w_minus":
                0.0,
        }

    ranks = stats.rankdata(
        np.abs(
            difference
        ),
        method="average",
    )

    w_plus = float(
        np.sum(
            ranks[
                difference > 0
            ]
        )
    )

    w_minus = float(
        np.sum(
            ranks[
                difference < 0
            ]
        )
    )

    denominator = (
        w_plus
        +
        w_minus
    )

    rb = (
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
                rb
            ),

        "n_nonzero":
            int(
                len(
                    difference
                )
            ),

        "w_plus":
            w_plus,

        "w_minus":
            w_minus,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 78
    )

    print(
        "RETRIEVAL STATISTICAL ANALYSIS V1"
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
            "Master JSONL changed."
        )

    if (
        csv_hash
        !=
        EXPECTED_MASTER_CSV_SHA256
    ):

        raise RuntimeError(
            "Master CSV changed."
        )

    if (
        manifest_hash
        !=
        EXPECTED_MASTER_MANIFEST_SHA256
    ):

        raise RuntimeError(
            "Master manifest changed."
        )

    manifest = load_json(
        MASTER_MANIFEST_FILE
    )

    if (
        manifest.get(
            "status"
        )
        !=
        "FROZEN"
    ):

        raise RuntimeError(
            "Master not FROZEN."
        )

    rows = load_jsonl(
        MASTER_JSONL_FILE
    )

    if (
        len(
            rows
        )
        !=
        120
    ):

        raise RuntimeError(
            "Master must have 120 rows."
        )

    print(
        "\n✅ Frozen master valid."
    )

    # ========================================================
    # QUERY INDEX
    # ========================================================

    query_index = {}

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

        query_index.setdefault(
            query_id,
            {}
        )

        if (
            method
            in
            query_index[
                query_id
            ]
        ):

            raise RuntimeError(
                f"Duplicate "
                f"{query_id}/{method}"
            )

        query_index[
            query_id
        ][
            method
        ] = row

    query_ids = sorted(
        query_index.keys()
    )

    if (
        len(
            query_ids
        )
        !=
        QUERY_COUNT
    ):

        raise RuntimeError(
            "Expected 40 queries."
        )

    for query_id in query_ids:

        if (
            set(
                query_index[
                    query_id
                ].keys()
            )
            !=
            set(
                METHODS
            )
        ):

            raise RuntimeError(
                f"{query_id} incomplete."
            )

    # ========================================================
    # ARRAYS
    # ========================================================

    data = {
        method: {
            metric: []
            for metric
            in (
                HIT_METRICS
                +
                [
                    "reciprocal_rank_at5"
                ]
            )
        }

        for method
        in METHODS
    }

    for query_id in query_ids:

        for method in METHODS:

            row = (
                query_index[
                    query_id
                ][
                    method
                ]
            )

            for metric in HIT_METRICS:

                value = int(
                    row[
                        metric
                    ]
                )

                if value not in {
                    0,
                    1,
                }:

                    raise RuntimeError(
                        f"Invalid binary "
                        f"{query_id}/{method}/{metric}"
                    )

                data[
                    method
                ][
                    metric
                ].append(
                    value
                )

            rr = float(
                row[
                    "reciprocal_rank_at5"
                ]
            )

            if not math.isfinite(
                rr
            ):

                raise RuntimeError(
                    "Non-finite RR."
                )

            data[
                method
            ][
                "reciprocal_rank_at5"
            ].append(
                rr
            )

    # ========================================================
    # OVERALL DESCRIPTIVE
    # ========================================================

    overall_records = []

    for method in METHODS:

        hit1 = float(
            np.mean(
                data[
                    method
                ][
                    "hit_at1"
                ]
            )
        )

        hit3 = float(
            np.mean(
                data[
                    method
                ][
                    "hit_at3"
                ]
            )
        )

        hit5 = float(
            np.mean(
                data[
                    method
                ][
                    "hit_at5"
                ]
            )
        )

        mrr5 = float(
            np.mean(
                data[
                    method
                ][
                    "reciprocal_rank_at5"
                ]
            )
        )

        overall_records.append(
            {
                "method":
                    method,

                "n_queries":
                    QUERY_COUNT,

                "hit_at1":
                    hit1,

                "hit_at3":
                    hit3,

                "hit_at5":
                    hit5,

                "recall_at1":
                    hit1,

                "recall_at3":
                    hit3,

                "recall_at5":
                    hit5,

                "mrr_at5":
                    mrr5,
            }
        )

    # ========================================================
    # COCHRAN Q
    # ========================================================

    cochran_records = []

    for metric in HIT_METRICS:

        matrix = np.column_stack(
            [
                data[
                    method
                ][
                    metric
                ]

                for method
                in METHODS
            ]
        )

        (
            statistic,
            p_value,
        ) = cochran_q(
            matrix
        )

        cochran_records.append(
            {
                "metric":
                    metric,

                "n_queries":
                    QUERY_COUNT,

                "k_methods":
                    len(
                        METHODS
                    ),

                "q_statistic":
                    statistic,

                "df":
                    len(
                        METHODS
                    )
                    - 1,

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
            }
        )

    # ========================================================
    # PAIRWISE MCNEMAR + HOLM
    # ========================================================

    mcnemar_records = []

    for metric in HIT_METRICS:

        metric_records = []

        raw_p = []

        for (
            method_a,
            method_b,
        ) in PAIRS:

            values_a = (
                data[
                    method_a
                ][
                    metric
                ]
            )

            values_b = (
                data[
                    method_b
                ][
                    metric
                ]
            )

            result = exact_mcnemar(
                values_a,
                values_b,
            )

            record = {
                "metric":
                    metric,

                "method_a":
                    method_a,

                "method_b":
                    method_b,

                "rate_method_a":
                    float(
                        np.mean(
                            values_a
                        )
                    ),

                "rate_method_b":
                    float(
                        np.mean(
                            values_b
                        )
                    ),

                **result,
            }

            raw_p.append(
                result[
                    "exact_p_value"
                ]
            )

            metric_records.append(
                record
            )

        adjusted = holm_adjust(
            raw_p
        )

        for (
            record,
            adjusted_p,
        ) in zip(
            metric_records,
            adjusted,
        ):

            record[
                "holm_p"
            ] = float(
                adjusted_p
            )

            record[
                "significant_holm"
            ] = bool(
                adjusted_p
                <
                ALPHA
            )

            mcnemar_records.append(
                record
            )

    # ========================================================
    # MRR@5 FRIEDMAN
    # ========================================================

    rr_arrays = [
        np.asarray(
            data[
                method
            ][
                "reciprocal_rank_at5"
            ],
            dtype=float,
        )

        for method
        in METHODS
    ]

    friedman_result = (
        stats.friedmanchisquare(
            *rr_arrays
        )
    )

    friedman_statistic = float(
        friedman_result.statistic
    )

    friedman_p = float(
        friedman_result.pvalue
    )

    kendalls_w = (
        friedman_statistic
        /
        (
            QUERY_COUNT
            *
            (
                len(
                    METHODS
                )
                -
                1
            )
        )
    )

    mrr_friedman_records = [
        {
            "metric":
                "mrr_at5",

            "n_queries":
                QUERY_COUNT,

            "k_methods":
                len(
                    METHODS
                ),

            "chi_square":
                friedman_statistic,

            "df":
                len(
                    METHODS
                )
                - 1,

            "p_value":
                friedman_p,

            "alpha":
                ALPHA,

            "significant":
                bool(
                    friedman_p
                    <
                    ALPHA
                ),

            "kendalls_w":
                float(
                    kendalls_w
                ),
        }
    ]

    # ========================================================
    # MRR PAIRWISE WILCOXON
    # ========================================================

    wilcoxon_records = []

    raw_p_values = []

    temporary_records = []

    for (
        method_a,
        method_b,
    ) in PAIRS:

        values_a = np.asarray(
            data[
                method_a
            ][
                "reciprocal_rank_at5"
            ],
            dtype=float,
        )

        values_b = np.asarray(
            data[
                method_b
            ][
                "reciprocal_rank_at5"
            ],
            dtype=float,
        )

        effect = rank_biserial(
            values_a,
            values_b,
        )

        if (
            effect[
                "n_nonzero"
            ]
            ==
            0
        ):

            statistic = 0.0
            p_value = 1.0

        else:

            result = stats.wilcoxon(
                values_a,
                values_b,
                zero_method="wilcox",
                correction=False,
                alternative="two-sided",
                method="approx",
            )

            statistic = float(
                result.statistic
            )

            p_value = float(
                result.pvalue
            )

        record = {
            "metric":
                "mrr_at5",

            "method_a":
                method_a,

            "method_b":
                method_b,

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
                float(
                    np.mean(
                        values_a
                        -
                        values_b
                    )
                ),

            "n_nonzero":
                effect[
                    "n_nonzero"
                ],

            "wilcoxon_statistic":
                statistic,

            "raw_p":
                p_value,

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

        raw_p_values.append(
            p_value
        )

        temporary_records.append(
            record
        )

    adjusted = holm_adjust(
        raw_p_values
    )

    for (
        record,
        adjusted_p,
    ) in zip(
        temporary_records,
        adjusted,
    ):

        record[
            "holm_p"
        ] = float(
            adjusted_p
        )

        record[
            "significant_holm"
        ] = bool(
            adjusted_p
            <
            ALPHA
        )

        wilcoxon_records.append(
            record
        )

    # ========================================================
    # WRITE OUTPUTS
    # ========================================================

    write_csv(
        OVERALL_CSV,
        overall_records,
    )

    write_csv(
        COCHRAN_CSV,
        cochran_records,
    )

    write_csv(
        MCNEMAR_CSV,
        mcnemar_records,
    )

    write_csv(
        MRR_FRIEDMAN_CSV,
        mrr_friedman_records,
    )

    write_csv(
        MRR_WILCOXON_CSV,
        wilcoxon_records,
    )

    summary = {
        "artifact":
            (
                "Aire Optima Retrieval "
                "Statistical Analysis v1"
            ),

        "status":
            "COMPLETE_NOT_YET_FROZEN",

        "source_master_sha256":
            jsonl_hash,

        "alpha":
            ALPHA,

        "overall":
            overall_records,

        "cochran_q":
            cochran_records,

        "pairwise_mcnemar":
            mcnemar_records,

        "mrr_friedman":
            mrr_friedman_records,

        "mrr_pairwise_wilcoxon":
            wilcoxon_records,

        "methodological_notes": {
            "hit_recall":
                (
                    "Hit@K and Recall@K are "
                    "numerically identical because "
                    "each query has exactly one "
                    "binary relevant document."
                ),

            "binary_omnibus":
                (
                    "Cochran's Q is used for "
                    "three related binary retrieval "
                    "outcomes."
                ),

            "binary_pairwise":
                (
                    "Exact McNemar tests with "
                    "Holm correction are used "
                    "for paired Hit@K comparisons."
                ),

            "mrr":
                (
                    "Friedman omnibus and paired "
                    "Wilcoxon signed-rank tests "
                    "with Holm correction are used "
                    "for reciprocal rank at 5."
                ),
        },
    }

    write_json(
        SUMMARY_JSON,
        summary,
    )

    output_files = {
        "overall_csv":
            OVERALL_CSV,

        "cochran_csv":
            COCHRAN_CSV,

        "mcnemar_csv":
            MCNEMAR_CSV,

        "mrr_friedman_csv":
            MRR_FRIEDMAN_CSV,

        "mrr_wilcoxon_csv":
            MRR_WILCOXON_CSV,

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

    freeze_manifest = {
        "artifact":
            (
                "Aire Optima Retrieval "
                "Statistical Analysis v1"
            ),

        "status":
            "FROZEN",

        "source": {
            "master_jsonl":
                jsonl_hash,

            "master_csv":
                csv_hash,

            "master_manifest":
                manifest_hash,
        },

        "environment": {
            "numpy":
                np.__version__,

            "scipy":
                scipy.__version__,
        },

        "analysis_policy": {
            "alpha":
                ALPHA,

            "hit_at_k_omnibus":
                "Cochran Q",

            "hit_at_k_pairwise":
                "Exact McNemar + Holm",

            "mrr_at5_omnibus":
                "Friedman",

            "mrr_at5_pairwise":
                "Wilcoxon + Holm",

            "effect_size_mrr":
                "Matched-pairs rank-biserial",
        },

        "outputs":
            output_hashes,

        "freeze_rule":
            (
                "Do not modify these retrieval "
                "statistical outputs after freeze."
            ),
    }

    write_json(
        MANIFEST_FILE,
        freeze_manifest,
    )

    freeze_manifest_hash = (
        sha256_file(
            MANIFEST_FILE
        )
    )

    # ========================================================
    # PRINT
    # ========================================================

    print(
        "\n"
        + "=" * 78
    )

    print(
        "OVERALL RETRIEVAL"
    )

    print(
        "=" * 78
    )

    for record in overall_records:

        print(
            f"{record['method']:<12} "
            f"H1={record['hit_at1']:.3f} "
            f"H3={record['hit_at3']:.3f} "
            f"H5={record['hit_at5']:.3f} "
            f"MRR5={record['mrr_at5']:.4f}"
        )

    print(
        "\n"
        + "=" * 78
    )

    print(
        "COCHRAN Q - HIT@K"
    )

    print(
        "=" * 78
    )

    for record in cochran_records:

        print(
            f"{record['metric']:<10} "
            f"Q={record['q_statistic']:.6f} "
            f"p={record['p_value']:.8f} "
            f"sig={record['significant']}"
        )

    print(
        "\nPAIRWISE EXACT MCNEMAR + HOLM"
    )

    current_metric = None

    for record in mcnemar_records:

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
            f"| {record['rate_method_a']:.3f} "
            f"vs {record['rate_method_b']:.3f} "
            f"| discordant="
            f"{record['discordant']} "
            f"| raw p="
            f"{record['exact_p_value']:.8f} "
            f"| Holm="
            f"{record['holm_p']:.8f} "
            f"| sig="
            f"{record['significant_holm']}"
        )

    print(
        "\n"
        + "=" * 78
    )

    print(
        "MRR@5 FRIEDMAN"
    )

    print(
        "=" * 78
    )

    record = (
        mrr_friedman_records[0]
    )

    print(
        f"chi2="
        f"{record['chi_square']:.6f} "
        f"p="
        f"{record['p_value']:.8f} "
        f"W="
        f"{record['kendalls_w']:.6f} "
        f"sig="
        f"{record['significant']}"
    )

    print(
        "\nMRR@5 PAIRWISE WILCOXON + HOLM"
    )

    for record in wilcoxon_records:

        print(
            f"{record['method_a']:<11} "
            f"vs "
            f"{record['method_b']:<11} "
            f"| mean="
            f"{record['mean_method_a']:.4f}:"
            f"{record['mean_method_b']:.4f} "
            f"| raw p="
            f"{record['raw_p']:.8f} "
            f"| Holm="
            f"{record['holm_p']:.8f} "
            f"| r_rb="
            f"{record['rank_biserial']:+.4f} "
            f"| sig="
            f"{record['significant_holm']}"
        )

    print(
        "\n"
        + "=" * 78
    )

    print(
        "RETRIEVAL STATISTICS FROZEN"
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
            f"{name:<22}: "
            f"{file_hash}"
        )

    print(
        "\nMANIFEST SHA256:"
    )

    print(
        freeze_manifest_hash
    )

    print(
        "\nSTATUS: FROZEN"
    )


if __name__ == "__main__":
    main()