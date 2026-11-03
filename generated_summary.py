import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from google import genai
from google.genai import types

import config


# ============================================================
# CONFIG
# ============================================================

PROJECT_ID = config.PROJECT_ID

GEMINI_MODEL = getattr(
    config,
    "GEMINI_MODEL",
    "gemini-3.8-flash",
)

VERTEX_LOCATION = getattr(
    config,
    "VERTEX_LOCATION",
    "global",
)


# ============================================================
# FILES
# ============================================================

INSIGHT_FILE = Path(
    "output/page1_insight_points.csv"
)

TREND_FILE = Path(
    "output/page1_trend_points.csv"
)

OUTPUT_FILE = Path(
    "output/page1_ai_summaries_iteration2.csv"
)

LOG_FILE = Path(
    "output/page1_ai_generation_log_iteration2.csv"
)

RAW_FAILURE_DIR = Path(
    "output/iteration2_failed_responses"
)


# ============================================================
# GENERATION SETTINGS
# ============================================================

MAX_ATTEMPTS = 3

RETRY_SLEEP_SECONDS = 2


# ============================================================
# STRUCTURED OUTPUT SCHEMA
# ============================================================

SUMMARY_SCHEMA = {

    "type": "object",

    "properties": {

        "executive_point_1": {
            "type": "string"
        },

        "executive_point_2": {
            "type": "string"
        },

        "key_movement_1": {
            "type": "string"
        },

        "key_movement_2": {
            "type": "string"
        },

        "key_movement_3": {
            "type": "string"
        },

        "key_movement_4": {
            "type": "string"
        },

        "trend_point_1": {
            "type": "string"
        },

        "trend_point_2": {
            "type": "string"
        },

        "budget_ytd_point_1": {
            "type": "string"
        },

        "budget_ytd_point_2": {
            "type": "string"
        },
    },

    "required": [

        "executive_point_1",
        "executive_point_2",

        "key_movement_1",
        "key_movement_2",
        "key_movement_3",
        "key_movement_4",

        "trend_point_1",
        "trend_point_2",

        "budget_ytd_point_1",
        "budget_ytd_point_2",
    ],

    "additionalProperties": False,

    "propertyOrdering": [

        "executive_point_1",
        "executive_point_2",

        "key_movement_1",
        "key_movement_2",
        "key_movement_3",
        "key_movement_4",

        "trend_point_1",
        "trend_point_2",

        "budget_ytd_point_1",
        "budget_ytd_point_2",
    ],
}


OUTPUT_FIELDS = [

    "executive_point_1",
    "executive_point_2",

    "key_movement_1",
    "key_movement_2",
    "key_movement_3",
    "key_movement_4",

    "trend_point_1",
    "trend_point_2",

    "budget_ytd_point_1",
    "budget_ytd_point_2",
]


# ============================================================
# WORDING THAT SHOULD NOT APPEAR
#
# These terms either imply unsupported business judgement
# or unsupported causality.
# ============================================================

BANNED_PHRASES = [

    "outperformed",

    "underperformed",

    "favorable",

    "favourable",

    "unfavorable",

    "unfavourable",

    "supported by",

    "driven by",

    "due to",

    "caused by",

    "helped by",

    "benefited from",

    "improvement of",

    "deterioration of",

    "strong performance",

    "weak performance",

    "healthy",

    "concerning",

    "revERSAL_UP",

    "revERSAL_DOWN",

    "CONSISTENT_INCREASE",

    "CONSISTENT_DECREASE",
]


# ============================================================
# MAX WORDS PER POINT
#
# Rich enough to explain the number, but still popup-friendly.
# ============================================================

MAX_WORDS = {

    "executive_point_1": 42,
    "executive_point_2": 42,

    "key_movement_1": 34,
    "key_movement_2": 34,
    "key_movement_3": 34,
    "key_movement_4": 34,

    "trend_point_1": 38,
    "trend_point_2": 38,

    "budget_ytd_point_1": 38,
    "budget_ytd_point_2": 38,
}


# ============================================================
# GENAI CLIENT
# ============================================================

def create_client():

    return genai.Client(
        vertexai=True,
        project=PROJECT_ID,
        location=VERTEX_LOCATION,
    )


# ============================================================
# GENERAL UTILITIES
# ============================================================

def is_missing(value):

    try:
        return pd.isna(value)

    except Exception:
        return value is None


def clean_string(value):

    if value is None:
        return ""

    if is_missing(value):
        return ""

    return str(value).strip()


def safe_float(value):

    if value is None:
        return None

    if is_missing(value):
        return None

    try:
        return float(value)

    except Exception:
        return None


def rounded_rank(value):

    value = safe_float(
        value
    )

    if value is None:
        return None

    return int(
        round(value)
    )


# ============================================================
# MONEY
# ============================================================

def to_millions(value):

    value = safe_float(
        value
    )

    if value is None:
        return None

    return round(
        value / 1_000_000,
        2,
    )


# ============================================================
# SAFE PERCENTAGE
#
# IMPORTANT ITERATION-2 RULE
#
# Percentages on negative bases or sign-crossing movements can
# be highly misleading.
#
# Example:
#
# -100 → +50
#
# should NOT be described as +150%.
#
# Instead:
#
# "moved from -100m to +50m, a 150m swing."
#
# Therefore MoM percentage is only supplied to Gemini when:
#
# previous > 0
# AND
# current > 0
#
# ============================================================

def safe_positive_base_pct(
    percentage,
    current,
    previous,
):

    percentage = safe_float(
        percentage
    )

    current = safe_float(
        current
    )

    previous = safe_float(
        previous
    )


    if percentage is None:
        return None

    if current is None:
        return None

    if previous is None:
        return None


    if previous <= 0:
        return None


    if current <= 0:
        return None


    if abs(
        percentage
    ) > 500:

        return None


    return round(
        percentage,
        2,
    )


# ============================================================
# LOAD INPUTS
# ============================================================

def load_inputs():

    if not INSIGHT_FILE.exists():

        raise FileNotFoundError(
            f"Missing file: "
            f"{INSIGHT_FILE.resolve()}"
        )


    if not TREND_FILE.exists():

        raise FileNotFoundError(
            f"Missing file: "
            f"{TREND_FILE.resolve()}"
        )


    insight_df = pd.read_csv(
        INSIGHT_FILE,
        dtype={
            "Entity_CD": str,
            "Account_CD": str,
        },
    )


    trend_df = pd.read_csv(
        TREND_FILE,
        dtype={
            "Entity_CD": str,
            "Account_CD": str,
        },
    )


    insight_strings = [

        "Entity_CD",
        "Entity_Name",
        "Currency",
        "Selected_Month",

        "Section",
        "Metric",
        "Account_CD",
        "Account_Name",
    ]


    for column in insight_strings:

        if column in insight_df.columns:

            insight_df[column] = (
                insight_df[column]
                .fillna("")
                .astype(str)
                .str.strip()
            )


    trend_strings = [

        "Entity_CD",
        "Entity_Name",
        "Currency",
        "Selected_Month",

        "Metric_Type",
        "Metric",
        "Account_CD",
        "Account_Name",
        "Trend_Month",
    ]


    for column in trend_strings:

        if column in trend_df.columns:

            trend_df[column] = (
                trend_df[column]
                .fillna("")
                .astype(str)
                .str.strip()
            )


    return (
        insight_df,
        trend_df,
    )


# ============================================================
# BUILD CURRENT FACT
# ============================================================

def build_current_fact(row):

    actual_raw = safe_float(
        row.get(
            "Actual"
        )
    )

    previous_raw = safe_float(
        row.get(
            "Previous_Actual"
        )
    )


    mom_pct = safe_positive_base_pct(

        percentage=
            row.get(
                "MoM_Pct_Safe"
            ),

        current=
            actual_raw,

        previous=
            previous_raw,
    )


    # ========================================================
    # CHANGE FROM FIRST MONTH %
    #
    # Same conservative percentage rule.
    # ========================================================

    first_actual_raw = safe_float(
        row.get(
            "First_Actual"
        )
    )


    change_from_first_pct = (
        safe_positive_base_pct(

            percentage=
                row.get(
                    "Change_From_First_Pct"
                ),

            current=
                actual_raw,

            previous=
                first_actual_raw,
        )
    )


    fact = {

        "section":
            clean_string(
                row.get(
                    "Section"
                )
            ),

        "rank":
            rounded_rank(
                row.get(
                    "Rank"
                )
            ),

        "metric":
            clean_string(
                row.get(
                    "Metric"
                )
            ),

        "account":
            clean_string(
                row.get(
                    "Account_Name"
                )
            ),


        # ====================================================
        # CURRENT MONTH
        # ====================================================

        "actual_m":
            to_millions(
                actual_raw
            ),

        "previous_month":
            clean_string(
                row.get(
                    "Previous_Month"
                )
            ),

        "previous_actual_m":
            to_millions(
                previous_raw
            ),

        "mom_change_m":
            to_millions(
                row.get(
                    "MoM_Change"
                )
            ),

        "mom_pct":
            mom_pct,

        "mom_direction":
            clean_string(
                row.get(
                    "MoM_Direction"
                )
            ),


        # ====================================================
        # BUDGET
        # ====================================================

        "budget_m":
            to_millions(
                row.get(
                    "Budget"
                )
            ),

        "vs_budget_m":
            to_millions(
                row.get(
                    "Vs_Budget"
                )
            ),

        "budget_position":
            clean_string(
                row.get(
                    "Budget_Position"
                )
            ),


        # ====================================================
        # YOY
        # ====================================================

        "yoy_change_m":
            to_millions(
                row.get(
                    "YoY_Abs"
                )
            ),


        # ====================================================
        # YTD
        # ====================================================

        "ytd_actual_m":
            to_millions(
                row.get(
                    "YTD_Actual"
                )
            ),

        "ytd_budget_m":
            to_millions(
                row.get(
                    "YTD_Budget"
                )
            ),

        "ytd_vs_budget_m":
            to_millions(
                row.get(
                    "YTD_Vs_Budget"
                )
            ),

        "ytd_budget_position":
            clean_string(
                row.get(
                    "YTD_Budget_Position"
                )
            ),

        "ytd_yoy_change_m":
            to_millions(
                row.get(
                    "YTD_YoY_Abs"
                )
            ),


        # ====================================================
        # MULTI-MONTH COMPARISONS
        # ====================================================

        "two_months_ago_actual_m":
            to_millions(
                row.get(
                    "Two_Months_Ago_Actual"
                )
            ),

        "two_month_change_m":
            to_millions(
                row.get(
                    "Two_Month_Change"
                )
            ),

        "three_months_ago_actual_m":
            to_millions(
                row.get(
                    "Three_Months_Ago_Actual"
                )
            ),

        "three_month_change_m":
            to_millions(
                row.get(
                    "Three_Month_Change"
                )
            ),


        # ====================================================
        # HISTORY
        # ====================================================

        "history_months":
            rounded_rank(
                row.get(
                    "History_Months"
                )
            ),

        "first_month":
            clean_string(
                row.get(
                    "First_Month"
                )
            ),

        "first_actual_m":
            to_millions(
                first_actual_raw
            ),

        "change_from_first_m":
            to_millions(
                row.get(
                    "Change_From_First"
                )
            ),

        "change_from_first_pct":
            change_from_first_pct,

        "peak_month":
            clean_string(
                row.get(
                    "Peak_Month"
                )
            ),

        "peak_actual_m":
            to_millions(
                row.get(
                    "Peak_Actual"
                )
            ),

        "lowest_month":
            clean_string(
                row.get(
                    "Lowest_Month"
                )
            ),

        "lowest_actual_m":
            to_millions(
                row.get(
                    "Lowest_Actual"
                )
            ),

        "recent_trend":
            clean_string(
                row.get(
                    "Recent_Trend"
                )
            ),

        "consecutive_direction":
            clean_string(
                row.get(
                    "Consecutive_Direction"
                )
            ),

        "consecutive_count":
            rounded_rank(
                row.get(
                    "Consecutive_Count"
                )
            ),


        # ====================================================
        # RANKS
        # ====================================================

        "materiality_rank":
            rounded_rank(
                row.get(
                    "Materiality_Rank"
                )
            ),

        "movement_rank":
            rounded_rank(
                row.get(
                    "Movement_Rank"
                )
            ),

        "budget_variance_rank":
            rounded_rank(
                row.get(
                    "Budget_Variance_Rank"
                )
            ),
    }


    # ========================================================
    # REMOVE NONE
    #
    # Zero stays because zero is meaningful.
    # ========================================================

    return {

        key: value

        for key, value
        in fact.items()

        if value is not None
    }


# ============================================================
# BUILD TREND HISTORY
# ============================================================

def build_trend_facts(
    trend_subset,
):

    output = []


    grouped = trend_subset.groupby(
        [
            "Account_CD",
            "Metric",
            "Metric_Type",
        ],
        dropna=False,
    )


    for (
        account_cd,
        metric,
        metric_type,
    ), group in grouped:

        group = group.sort_values(
            "Trend_Month_Order"
        )


        history = []


        for _, row in group.iterrows():

            history.append(
                {

                    "month":
                        clean_string(
                            row.get(
                                "Trend_Month"
                            )
                        ),

                    "actual_m":
                        to_millions(
                            row.get(
                                "Actual"
                            )
                        ),

                    "budget_m":
                        to_millions(
                            row.get(
                                "Budget"
                            )
                        ),

                    "vs_budget_m":
                        to_millions(
                            row.get(
                                "Vs_Budget"
                            )
                        ),
                }
            )


        output.append(
            {

                "metric_type":
                    clean_string(
                        metric_type
                    ),

                "metric":
                    clean_string(
                        metric
                    ),

                "history":
                    history,
            }
        )


    return output


# ============================================================
# BUILD FACT PACKAGE
# ============================================================

def build_fact_package(
    insight_subset,
    trend_subset,
):

    first = insight_subset.iloc[0]


    # ========================================================
    # HEADLINES
    # ========================================================

    headline_df = (
        insight_subset[
            insight_subset[
                "Section"
            ]
            ==
            "HEADLINE"
        ]
        .sort_values(
            "Rank"
        )
        .copy()
    )


    # ========================================================
    # DRIVERS
    #
    # Exactly four from Step 2.
    # Their order is materiality rank 1 → 4.
    # ========================================================

    driver_df = (
        insight_subset[
            insight_subset[
                "Section"
            ]
            ==
            "DRIVER"
        ]
        .sort_values(
            "Rank"
        )
        .copy()
    )


    headline_facts = [

        build_current_fact(
            row
        )

        for _, row
        in headline_df.iterrows()
    ]


    driver_facts = [

        build_current_fact(
            row
        )

        for _, row
        in driver_df.iterrows()
    ]


    return {

        "context": {

            "fiscal_year":
                int(
                    first[
                        "Fiscal_Year"
                    ]
                ),

            "month":
                clean_string(
                    first[
                        "Selected_Month"
                    ]
                ),

            "entity":
                clean_string(
                    first[
                        "Entity_Name"
                    ]
                ),

            "entity_cd":
                clean_string(
                    first[
                        "Entity_CD"
                    ]
                ),

            "currency":
                clean_string(
                    first[
                        "Currency"
                    ]
                ),

            "amount_unit":
                "millions",
        },


        "headline_kpis":
            headline_facts,


        "top_4_material_drivers":
            driver_facts,


        "historical_monthly_values":
            build_trend_facts(
                trend_subset
            ),
    }


# ============================================================
# PROMPT
# ============================================================

def build_prompt(
    fact_package,
    retry_note="",
):

    context = (
        fact_package[
            "context"
        ]
    )


    retry_section = ""


    if retry_note:

        retry_section = f"""

============================================================
PREVIOUS ATTEMPT DID NOT PASS QUALITY VALIDATION
============================================================

{retry_note}

Correct those issues in this attempt.
"""


    return f"""
You are an executive cash-flow analyst writing commentary for an
SAP Analytics Cloud cash-flow dashboard.

Selected context:

Fiscal Year: {context['fiscal_year']}
Month: {context['month']}
Entity: {context['entity']}
Currency: {context['currency']}

All monetary values supplied below are already in MILLIONS of the
selected dashboard currency.

For EUR:
write amounts such as EUR 125.4m.

For LC:
write amounts such as LC 125.4m.

Never convert between EUR and LC.


============================================================
THE PURPOSE OF THIS COMMENTARY
============================================================

When an executive opens the popup, they should understand the
financial story of the selected month without having to interpret
every dashboard chart manually.

The commentary must explain:

1. What materially changed.
2. Which headline cash-flow measures matter most.
3. Which underlying cash-flow drivers moved materially.
4. Whether the latest result continues or reverses recent history.
5. How the selected month compares with budget.
6. How the cumulative YTD position compares with budget.
7. Relevant YoY context when available.


============================================================
WHAT "INSIGHTFUL" MEANS
============================================================

Do not simply write:

"EBITDA increased 3.5%."

Instead connect supported facts:

"EBITDA increased 3.5% MoM to EUR 145.2m, but remained
EUR 11.6m below the September budget."

Another good form:

"Free Cash Flow fell to EUR 87.3m from EUR 163.5m in August,
reversing the previous month's increase while remaining above its
YTD budget position."

Every point should ideally connect two or more relevant facts when
the data allows it.


============================================================
IMPORTANT PERCENTAGE RULE
============================================================

Use MoM percentage ONLY when the field "mom_pct" is supplied.

If "mom_pct" is missing, DO NOT calculate one yourself.

It has deliberately been removed when:

- the previous value was zero
- the previous value was negative
- the current value was negative
- the movement crossed zero
- the percentage would otherwise be misleading

In those situations describe the movement using values and
absolute change.

Example:

Previous = LC -866.7m
Current  = LC 1,803.8m

Correct:

"WC Receivables moved from LC -866.7m to LC 1,803.8m,
a LC 2,670.5m swing."

Incorrect:

"WC Receivables increased 308%."


============================================================
EXECUTIVE SUMMARY — EXACTLY 2 POINTS
============================================================

These should be the two most important overall conclusions.

Use HEADLINE KPIs.

Each point should normally combine multiple supported facts such as:

- MoM direction
- actual value
- budget position
- recent trend
- another headline KPI where useful

Do not simply copy one Key Movement.

Aim for approximately 25–38 words per point.


============================================================
KEY MOVEMENTS — EXACTLY 4 POINTS
============================================================

THIS RULE IS STRICT:

There are exactly four objects under:

"top_4_material_drivers"

Key Movement 1 must discuss DRIVER rank 1.
Key Movement 2 must discuss DRIVER rank 2.
Key Movement 3 must discuss DRIVER rank 3.
Key Movement 4 must discuss DRIVER rank 4.

Do NOT replace a driver with EBITDA, FCF, OFCF, Working Capital,
Capital Additions, or another headline KPI.

The purpose of this section is to explain the underlying movements
that sit beneath the headline view.

For each driver, explain why it stands out using available evidence:

- absolute MoM movement
- current versus previous value
- budget position
- historical high/low
- reversal
- consecutive movement
- materiality ranking

Do not claim that a driver CAUSED a headline movement.

Aim for approximately 18–30 words per point.


============================================================
TREND INSIGHT — EXACTLY 2 POINTS
============================================================

Translate technical trend labels into natural financial language.

Examples:

REVERSAL_UP

should become:

"reversed upward in September after declining in August"

NOT:

"REVERSAL_UP"


REVERSAL_DOWN

should become:

"reversed lower in September after increasing in August"


CONSISTENT_INCREASE

should become:

"has increased across each available monthly observation"


CONSISTENT_DECREASE

should become:

"has declined across each available monthly observation"


Use:

- multi-month history
- consecutive movement count
- peak month
- lowest month
- change from first available month
- reversals

Do not predict future performance.

If this is the first available month, state that there is insufficient
history rather than inventing a trend.

Aim for approximately 20–32 words per point.


============================================================
BUDGET & YTD — EXACTLY 2 POINTS
============================================================

Explain monthly budget position and cumulative YTD position.

Look especially for contrasts such as:

- current month below budget but YTD above budget
- current month above budget but YTD below budget
- both monthly and YTD below budget
- both monthly and YTD above budget

Use neutral mathematical wording:

"above budget"
"below budget"
"on budget"

Do NOT use:

"outperformed"
"underperformed"
"favorable"
"unfavorable"
"strong"
"weak"

Aim for approximately 20–32 words per point.


============================================================
CAUSALITY RULE — VERY IMPORTANT
============================================================

You have numerical relationships, not operational causal evidence.

Never write:

"X was driven by Y"
"X was supported by Y"
"X improved due to Y"
"Y caused X"
"X benefited from Y"

unless explicit causal information is supplied.

Instead use:

"while"
"alongside"
"at the same time"
"the same month also saw"
"coincided with"

Example:

Correct:

"Working Capital moved lower while WC Receivables also recorded
a material decline."

Incorrect:

"Working Capital declined due to WC Receivables."


============================================================
BUSINESS-JUDGEMENT RULE
============================================================

Do not assume that increasing or decreasing is automatically good
or bad.

Do not use terms such as:

favorable
unfavorable
healthy
poor
concerning
strong performance
weak performance
improvement
deterioration

unless the supplied data explicitly establishes that interpretation.

Prefer descriptive language:

increased
decreased
moved higher
moved lower
reversed upward
reversed lower
above budget
below budget
widened
narrowed
remained stable


============================================================
ANTI-REPETITION RULE
============================================================

The popup contains 10 points.

Each point must add information.

Do not use all sections to repeat:

"FCF decreased by X."

Executive Summary:
overall conclusion.

Key Movements:
underlying four drivers.

Trend:
historical pattern.

Budget & YTD:
plan/cumulative position.

These sections should complement each other.


============================================================
FACTUAL RULES
============================================================

Use ONLY facts provided below.

Never invent:

- operational causes
- market explanations
- management actions
- future forecasts
- missing numbers
- missing percentages
- business events

Do not mix entities.

Do not mix months.

Do not mix EUR and LC.

All commentary must refer only to:

{context['entity']}
{context['month']}
{context['currency']}


============================================================
FACT PACKAGE
============================================================

{json.dumps(
    fact_package,
    indent=2,
    ensure_ascii=False,
    allow_nan=False,
)}

{retry_section}

Return only the structured response required by the supplied schema.
""".strip()


# ============================================================
# PARSE RESPONSE
# ============================================================

def parse_response(
    response,
):

    parsed = getattr(
        response,
        "parsed",
        None,
    )


    if parsed is not None:

        if isinstance(
            parsed,
            dict,
        ):

            return parsed


        if hasattr(
            parsed,
            "model_dump",
        ):

            return parsed.model_dump()


    text = clean_string(
        getattr(
            response,
            "text",
            ""
        )
    )


    if not text:

        raise ValueError(
            "Gemini returned empty response."
        )


    if text.startswith(
        "```"
    ):

        text = text.strip(
            "`"
        )


        if text.lower().startswith(
            "json"
        ):

            text = text[
                4:
            ].strip()


    return json.loads(
        text
    )


# ============================================================
# WORD COUNT
# ============================================================

def word_count(text):

    return len(
        re.findall(
            r"\b[\w€.-]+\b",
            text,
        )
    )


# ============================================================
# QUALITY VALIDATION
# ============================================================

def validate_result(
    result,
    selected_month_order,
):

    if not isinstance(
        result,
        dict,
    ):

        raise ValueError(
            "Response is not a JSON object."
        )


    # ========================================================
    # REQUIRED FIELDS
    # ========================================================

    missing = [

        field

        for field in OUTPUT_FIELDS

        if field not in result
    ]


    if missing:

        raise ValueError(
            "Missing fields: "
            +
            ", ".join(
                missing
            )
        )


    cleaned = {

        field:
            clean_string(
                result.get(
                    field
                )
            )

        for field in OUTPUT_FIELDS
    }


    # ========================================================
    # NON-EMPTY
    # ========================================================

    empty_fields = [

        field

        for field in OUTPUT_FIELDS

        if not cleaned[
            field
        ]
    ]


    # April has limited history.
    #
    # Still require the model to explain that rather than
    # leaving blank.

    if empty_fields:

        raise ValueError(
            "Empty commentary fields: "
            +
            ", ".join(
                empty_fields
            )
        )


    # ========================================================
    # BANNED LANGUAGE
    # ========================================================

    combined_text = " ".join(
        cleaned.values()
    ).lower()


    bad_terms = []


    for phrase in BANNED_PHRASES:

        if (
            phrase.lower()
            in combined_text
        ):

            bad_terms.append(
                phrase
            )


    if bad_terms:

        raise ValueError(
            "Unsupported or technical wording found: "
            +
            ", ".join(
                bad_terms
            )
        )


    # ========================================================
    # POINT LENGTH
    # ========================================================

    too_long = []


    for field, text in (
        cleaned.items()
    ):

        count = word_count(
            text
        )


        maximum = MAX_WORDS[
            field
        ]


        if count > maximum:

            too_long.append(
                f"{field}="
                f"{count} words "
                f"(max {maximum})"
            )


    if too_long:

        raise ValueError(
            "Points are too long: "
            +
            "; ".join(
                too_long
            )
        )


    return cleaned


# ============================================================
# GENERATE ONE SUMMARY
# ============================================================

def generate_one_summary(
    client,
    fact_package,
    selected_month_order,
):

    last_error = ""

    last_raw_text = ""


    for attempt in range(
        1,
        MAX_ATTEMPTS + 1,
    ):

        try:

            prompt = build_prompt(

                fact_package=
                    fact_package,

                retry_note=
                    last_error,
            )


            response = (
                client.models.generate_content(

                    model=
                        GEMINI_MODEL,

                    contents=
                        prompt,

                    config=
                        types.GenerateContentConfig(

                            response_mime_type=
                                "application/json",

                            response_json_schema=
                                SUMMARY_SCHEMA,

                            thinking_config=
                                types.ThinkingConfig(

                                    thinking_level=
                                        types.ThinkingLevel.MEDIUM
                                ),
                        ),
                )
            )


            last_raw_text = (
                clean_string(
                    getattr(
                        response,
                        "text",
                        ""
                    )
                )
            )


            parsed = (
                parse_response(
                    response
                )
            )


            validated = (
                validate_result(

                    result=
                        parsed,

                    selected_month_order=
                        selected_month_order,
                )
            )


            return (
                validated,
                attempt,
                last_raw_text,
                "",
            )


        except Exception as exc:

            last_error = (
                f"{type(exc).__name__}: "
                f"{str(exc)}"
            )


            print(
                f"      Attempt "
                f"{attempt} failed: "
                f"{last_error}"
            )


            if attempt < MAX_ATTEMPTS:

                time.sleep(
                    RETRY_SLEEP_SECONDS
                )


    return (
        None,
        MAX_ATTEMPTS,
        last_raw_text,
        last_error,
    )


# ============================================================
# FULL SUMMARY
# ============================================================

def build_full_summary(
    result,
):

    return (
        "EXECUTIVE SUMMARY\n"
        f"• {result['executive_point_1']}\n"
        f"• {result['executive_point_2']}\n\n"

        "KEY MOVEMENTS\n"
        f"• {result['key_movement_1']}\n"
        f"• {result['key_movement_2']}\n"
        f"• {result['key_movement_3']}\n"
        f"• {result['key_movement_4']}\n\n"

        "TREND INSIGHT\n"
        f"• {result['trend_point_1']}\n"
        f"• {result['trend_point_2']}\n\n"

        "BUDGET & YTD\n"
        f"• {result['budget_ytd_point_1']}\n"
        f"• {result['budget_ytd_point_2']}"
    )


# ============================================================
# KEYS
# ============================================================

def make_key(
    fiscal_year,
    month,
    entity_cd,
    currency,
):

    return (

        int(
            fiscal_year
        ),

        clean_string(
            month
        ),

        clean_string(
            entity_cd
        ),

        clean_string(
            currency
        ),
    )


# ============================================================
# LOAD EXISTING OUTPUT
# ============================================================

def load_existing_output():

    if not OUTPUT_FILE.exists():

        return pd.DataFrame()


    try:

        return pd.read_csv(
            OUTPUT_FILE,
            dtype={
                "Entity_CD": str,
            },
        )

    except Exception:

        return pd.DataFrame()


# ============================================================
# REMOVE OLD KEY
# ============================================================

def remove_existing_key(
    df,
    key,
):

    if df.empty:

        return df


    (
        fiscal_year,
        month,
        entity_cd,
        currency,
    ) = key


    mask = ~(

        (
            pd.to_numeric(
                df[
                    "Fiscal_Year"
                ],
                errors="coerce",
            )
            ==
            fiscal_year
        )

        &

        (
            df[
                "Month"
            ].astype(str)
            ==
            month
        )

        &

        (
            df[
                "Entity_CD"
            ].astype(str)
            ==
            entity_cd
        )

        &

        (
            df[
                "Currency"
            ].astype(str)
            ==
            currency
        )
    )


    return (
        df[
            mask
        ]
        .copy()
    )


# ============================================================
# SAVE OUTPUT
# ============================================================

def save_output(
    df,
):

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    df = (
        df
        .sort_values(
            [
                "Fiscal_Year",
                "Month_Order",
                "Entity_Name",
                "Currency",
            ]
        )
        .reset_index(
            drop=True
        )
    )


    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )


# ============================================================
# APPEND LOG
# ============================================================

def append_log(
    log_row,
):

    LOG_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    log_df = pd.DataFrame(
        [
            log_row
        ]
    )


    if LOG_FILE.exists():

        log_df.to_csv(

            LOG_FILE,

            mode="a",

            index=False,

            header=False,

            encoding="utf-8-sig",
        )

    else:

        log_df.to_csv(

            LOG_FILE,

            index=False,

            encoding="utf-8-sig",
        )


# ============================================================
# SAVE FAILED RESPONSE
# ============================================================

def save_failed_response(
    key,
    raw_response,
    error,
):

    RAW_FAILURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    (
        fiscal_year,
        month,
        entity_cd,
        currency,
    ) = key


    safe_entity = (
        entity_cd
        .replace(
            "/",
            "_"
        )
        .replace(
            "\\",
            "_"
        )
        .replace(
            ".",
            "_"
        )
    )


    filename = (
        f"{safe_entity}_"
        f"{fiscal_year}_"
        f"{month}_"
        f"{currency}.txt"
    )


    path = (
        RAW_FAILURE_DIR
        /
        filename
    )


    path.write_text(

        (
            "ERROR\n"
            "=====\n"
            f"{error}\n\n"

            "RAW GEMINI RESPONSE\n"
            "===================\n"
            f"{raw_response}"
        ),

        encoding="utf-8",
    )


# ============================================================
# BUILD FINAL ROW
# ============================================================

def build_output_row(
    first_row,
    result,
    attempts,
):

    generated_at = (
        datetime.now(
            timezone.utc
        )
        .isoformat()
    )


    full_summary = (
        build_full_summary(
            result
        )
    )


    total_words = word_count(
        full_summary
    )


    return {

        "Fiscal_Year":
            int(
                first_row[
                    "Fiscal_Year"
                ]
            ),

        "Month":
            clean_string(
                first_row[
                    "Selected_Month"
                ]
            ),

        "Month_Order":
            int(
                first_row[
                    "Selected_Month_Order"
                ]
            ),

        "Entity_CD":
            clean_string(
                first_row[
                    "Entity_CD"
                ]
            ),

        "Entity_Name":
            clean_string(
                first_row[
                    "Entity_Name"
                ]
            ),

        "Currency":
            clean_string(
                first_row[
                    "Currency"
                ]
            ),


        # ====================================================
        # EXECUTIVE
        # ====================================================

        "Executive_Point_1":
            result[
                "executive_point_1"
            ],

        "Executive_Point_2":
            result[
                "executive_point_2"
            ],


        # ====================================================
        # KEY MOVEMENTS
        # ====================================================

        "Key_Movement_1":
            result[
                "key_movement_1"
            ],

        "Key_Movement_2":
            result[
                "key_movement_2"
            ],

        "Key_Movement_3":
            result[
                "key_movement_3"
            ],

        "Key_Movement_4":
            result[
                "key_movement_4"
            ],


        # ====================================================
        # TREND
        # ====================================================

        "Trend_Point_1":
            result[
                "trend_point_1"
            ],

        "Trend_Point_2":
            result[
                "trend_point_2"
            ],


        # ====================================================
        # BUDGET & YTD
        # ====================================================

        "Budget_YTD_Point_1":
            result[
                "budget_ytd_point_1"
            ],

        "Budget_YTD_Point_2":
            result[
                "budget_ytd_point_2"
            ],


        # ====================================================
        # VALIDATION / DEBUG
        # ====================================================

        "Summary_Full":
            full_summary,

        "Word_Count":
            total_words,

        "Generation_Status":
            "PASS",

        "Generation_Attempts":
            attempts,

        "Generated_At":
            generated_at,

        "Generator_Model":
            GEMINI_MODEL,

        "Generator_Thinking":
            "MEDIUM",
    }


# ============================================================
# MAIN GENERATION
# ============================================================

def run_generation(
    limit=None,
    force=False,
    currency_filter=None,
    entity_filter=None,
):

    print(
        "\n"
        +
        "=" * 90
    )

    print(
        "CASHFLOW AI - ITERATION 2 FINAL"
    )

    print(
        "RICH POPUP COMMENTARY"
    )

    print(
        "=" * 90
    )


    print(
        f"\nModel: "
        f"{GEMINI_MODEL}"
    )

    print(
        "Thinking: MEDIUM"
    )

    print(
        "Currency-aware: YES"
    )

    print(
        "Structured output: YES"
    )

    print(
        "Quality validation: YES"
    )

    print(
        "Safe percentage handling: YES"
    )


    # ========================================================
    # INPUTS
    # ========================================================

    (
        insight_df,
        trend_df,
    ) = load_inputs()


    # ========================================================
    # SUMMARY KEYS
    # ========================================================

    summary_keys = (
        insight_df[
            [
                "Fiscal_Year",

                "Selected_Month",

                "Selected_Month_Order",

                "Entity_CD",

                "Entity_Name",

                "Currency",
            ]
        ]

        .drop_duplicates()

        .sort_values(
            [
                "Selected_Month_Order",
                "Entity_Name",
                "Currency",
            ],
            ascending=[
                False,
                True,
                True,
            ],
        )

        .reset_index(
            drop=True
        )
    )


    # ========================================================
    # OPTIONAL FILTERS
    # ========================================================

    if currency_filter:

        summary_keys = (
            summary_keys[
                summary_keys[
                    "Currency"
                ]
                .str.upper()
                ==
                currency_filter.upper()
            ]
            .reset_index(
                drop=True
            )
        )


    if entity_filter:

        entity_filter_lower = (
            entity_filter.lower()
        )


        summary_keys = (
            summary_keys[
                summary_keys[
                    "Entity_Name"
                ]
                .str.lower()
                .str.contains(
                    entity_filter_lower,
                    regex=False,
                )
            ]
            .reset_index(
                drop=True
            )
        )


    if limit is not None:

        summary_keys = (
            summary_keys
            .head(
                limit
            )
            .copy()
        )


    print(
        f"\nCombinations selected: "
        f"{len(summary_keys)}"
    )


    # ========================================================
    # EXISTING OUTPUT
    # ========================================================

    output_df = (
        load_existing_output()
    )


    existing_keys = set()


    if not output_df.empty:

        for _, row in (
            output_df.iterrows()
        ):

            existing_keys.add(

                make_key(

                    row[
                        "Fiscal_Year"
                    ],

                    row[
                        "Month"
                    ],

                    row[
                        "Entity_CD"
                    ],

                    row[
                        "Currency"
                    ],
                )
            )


    client = create_client()


    generated = 0
    skipped = 0
    failures = 0


    total = len(
        summary_keys
    )


    # ========================================================
    # LOOP
    # ========================================================

    for position, (
        _,
        key_row,
    ) in enumerate(
        summary_keys.iterrows(),
        start=1,
    ):

        fiscal_year = int(
            key_row[
                "Fiscal_Year"
            ]
        )


        month = clean_string(
            key_row[
                "Selected_Month"
            ]
        )


        month_order = int(
            key_row[
                "Selected_Month_Order"
            ]
        )


        entity_cd = clean_string(
            key_row[
                "Entity_CD"
            ]
        )


        entity_name = clean_string(
            key_row[
                "Entity_Name"
            ]
        )


        currency = clean_string(
            key_row[
                "Currency"
            ]
        )


        key = make_key(
            fiscal_year,
            month,
            entity_cd,
            currency,
        )


        print(
            f"\n[{position}/{total}] "
            f"{entity_name} | "
            f"{month} | "
            f"{currency}"
        )


        # ====================================================
        # SKIP
        # ====================================================

        if (
            key in existing_keys
            and
            not force
        ):

            print(
                "    SKIPPED - already generated"
            )

            skipped += 1

            continue


        # ====================================================
        # CURRENT FACTS
        # ====================================================

        insight_subset = (
            insight_df[
                (
                    insight_df[
                        "Fiscal_Year"
                    ]
                    ==
                    fiscal_year
                )

                &

                (
                    insight_df[
                        "Selected_Month"
                    ]
                    ==
                    month
                )

                &

                (
                    insight_df[
                        "Entity_CD"
                    ]
                    ==
                    entity_cd
                )

                &

                (
                    insight_df[
                        "Currency"
                    ]
                    ==
                    currency
                )
            ]
            .copy()
        )


        # ====================================================
        # TREND FACTS
        # ====================================================

        trend_subset = (
            trend_df[
                (
                    trend_df[
                        "Fiscal_Year"
                    ]
                    ==
                    fiscal_year
                )

                &

                (
                    trend_df[
                        "Selected_Month"
                    ]
                    ==
                    month
                )

                &

                (
                    trend_df[
                        "Entity_CD"
                    ]
                    ==
                    entity_cd
                )

                &

                (
                    trend_df[
                        "Currency"
                    ]
                    ==
                    currency
                )
            ]
            .copy()
        )


        if insight_subset.empty:

            print(
                "    FAILED - no insight facts"
            )

            failures += 1

            continue


        # ====================================================
        # FACT PACKAGE
        # ====================================================

        fact_package = (
            build_fact_package(

                insight_subset=
                    insight_subset,

                trend_subset=
                    trend_subset,
            )
        )


        # ====================================================
        # GENERATE
        # ====================================================

        (
            result,
            attempts,
            raw_response,
            error,
        ) = generate_one_summary(

            client=
                client,

            fact_package=
                fact_package,

            selected_month_order=
                month_order,
        )


        generated_at = (
            datetime.now(
                timezone.utc
            )
            .isoformat()
        )


        # ====================================================
        # FAILURE
        # ====================================================

        if result is None:

            print(
                "    FAILED"
            )

            print(
                f"    {error}"
            )


            failures += 1


            save_failed_response(

                key=
                    key,

                raw_response=
                    raw_response,

                error=
                    error,
            )


            append_log(
                {

                    "Fiscal_Year":
                        fiscal_year,

                    "Month":
                        month,

                    "Entity_CD":
                        entity_cd,

                    "Entity_Name":
                        entity_name,

                    "Currency":
                        currency,

                    "Status":
                        "FAILED",

                    "Attempts":
                        attempts,

                    "Issue":
                        error,

                    "Generated_At":
                        generated_at,
                }
            )


            continue


        # ====================================================
        # SUCCESS
        # ====================================================

        output_row = (
            build_output_row(

                first_row=
                    insight_subset.iloc[0],

                result=
                    result,

                attempts=
                    attempts,
            )
        )


        print(
            f"    PASS "
            f"(attempt {attempts}, "
            f"{output_row['Word_Count']} words)"
        )


        output_df = (
            remove_existing_key(
                output_df,
                key,
            )
        )


        output_df = pd.concat(
            [
                output_df,

                pd.DataFrame(
                    [
                        output_row
                    ]
                ),
            ],
            ignore_index=True,
        )


        save_output(
            output_df
        )


        existing_keys.add(
            key
        )


        generated += 1


        append_log(
            {

                "Fiscal_Year":
                    fiscal_year,

                "Month":
                    month,

                "Entity_CD":
                    entity_cd,

                "Entity_Name":
                    entity_name,

                "Currency":
                    currency,

                "Status":
                    "PASS",

                "Attempts":
                    attempts,

                "Issue":
                    "",

                "Word_Count":
                    output_row[
                        "Word_Count"
                    ],

                "Generated_At":
                    generated_at,
            }
        )


    # ========================================================
    # END
    # ========================================================

    print(
        "\n"
        +
        "=" * 90
    )

    print(
        "ITERATION-2 FULL GENERATION COMPLETE"
    )

    print(
        "=" * 90
    )


    print(
        f"\nGenerated: "
        f"{generated}"
    )

    print(
        f"Skipped:   "
        f"{skipped}"
    )

    print(
        f"Failures:  "
        f"{failures}"
    )


    print(
        "\nFinal Output:"
    )

    print(
        OUTPUT_FILE.resolve()
    )


    print(
        "\nGeneration Log:"
    )

    print(
        LOG_FILE.resolve()
    )


    if failures > 0:

        print(
            "\nFailed response details:"
        )

        print(
            RAW_FAILURE_DIR.resolve()
        )


# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Generate final Iteration-2 "
            "Cashflow AI popup commentary."
        )
    )


    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )


    parser.add_argument(
        "--force",
        action="store_true",
    )


    parser.add_argument(
        "--currency",
        type=str,
        default=None,
        choices=[
            "EUR",
            "LC",
        ],
    )


    parser.add_argument(
        "--entity",
        type=str,
        default=None,
    )


    args = parser.parse_args()


    run_generation(

        limit=
            args.limit,

        force=
            args.force,

        currency_filter=
            args.currency,

        entity_filter=
            args.entity,
    )


if __name__ == "__main__":

    main()
