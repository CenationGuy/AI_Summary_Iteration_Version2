import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from google import genai
from google.genai import types

from config import (
    PROJECT_ID,
    GEMINI_MODEL,
    VERTEX_LOCATION,
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

# IMPORTANT:
# Keep Iteration 1 output untouched.
OUTPUT_FILE = Path(
    "output/page1_ai_summaries_iteration2.csv"
)

LOG_FILE = Path(
    "output/page1_ai_generation_log_iteration2.csv"
)


# ============================================================
# MODEL SETTINGS
# ============================================================

THINKING_LEVEL = "MEDIUM"

MAX_ATTEMPTS = 3

MAX_OUTPUT_TOKENS = 2500


# ============================================================
# RESPONSE STRUCTURE
#
# Gemini directly creates the point-wise output.
#
# No paragraph -> point splitting afterwards.
# ============================================================

RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "executive_summary": {
            "type": "ARRAY",
            "items": {
                "type": "STRING"
            },
        },
        "key_movements": {
            "type": "ARRAY",
            "items": {
                "type": "STRING"
            },
        },
        "trend_insight": {
            "type": "ARRAY",
            "items": {
                "type": "STRING"
            },
        },
        "budget_ytd": {
            "type": "ARRAY",
            "items": {
                "type": "STRING"
            },
        },
    },
    "required": [
        "executive_summary",
        "key_movements",
        "trend_insight",
        "budget_ytd",
    ],
}


# ============================================================
# SYSTEM INSTRUCTION
# ============================================================

SYSTEM_INSTRUCTION = """
You are an executive cash-flow analysis assistant.

You analyse structured financial facts for a Cash Flow dashboard and
produce concise but information-rich management commentary.

Your job is NOT to simply repeat numbers.

Your job is to identify what matters in the supplied facts and explain
the current financial picture clearly.

You must work ONLY from the supplied data.

============================================================
CORE RULES
============================================================

1. Never invent a number.

2. Never invent a cause, operational explanation, seasonality,
   management action, external event, or business reason.

3. Never predict future performance.

4. Never calculate a percentage that is not explicitly supplied.

5. If a percentage field is null, use the absolute movement instead.

6. When quoting amounts, use the provided display values rather than
   recalculating or rescaling raw values.

7. Never mix EUR and LC.

8. For EUR, use the supplied € display values.

9. For LC, use the supplied LC display values.
   Never guess the entity's real local currency.

10. Treat "above budget" and "below budget" as numerical positions.
    Do not automatically describe them as favourable or unfavourable.

11. Avoid unsupported causal wording such as:
    "due to"
    "caused by"
    "driven by"
    "because of"
    "resulting from"
    "seasonal"
    "seasonality"

12. Avoid vague corporate filler such as:
    "softened"
    "helping absorb"
    "supported by"
    "favourable"
    "favorable"
    "unfavourable"
    "unfavorable"
    "outperformed"
    "underperformed"

Use direct language:
rose, fell, increased, decreased, remained, reversed,
widened, narrowed, moved, recovered, declined.

============================================================
WHAT MAKES AN INSIGHT GOOD
============================================================

A strong point normally combines at least TWO useful facts when they
are available.

Examples:

- current result + month-on-month movement
- current result + budget position
- month-on-month movement + multi-month trend
- current result + YTD context
- monthly result + contrasting YTD position
- material driver + its recent historical pattern
- latest month + peak/low/reversal information

Do not create ten isolated single-number statements.

Connect the facts intelligently while remaining factual.

============================================================
EXECUTIVE SUMMARY
============================================================

Return EXACTLY 2 points.

These two points should explain the overall monthly cash-flow story.

Use the headline KPIs.

Do not list all six headline KPIs mechanically.

Prioritise the metrics that best explain the selected month's position.

Where useful, contrast two signals.

For example:
- FCF weakened while EBITDA improved.
- Monthly performance declined while YTD remained above budget.

Each executive point should normally be around 20-40 words.

============================================================
KEY MOVEMENTS
============================================================

Return EXACTLY 4 points.

Four DRIVER facts are supplied, ranked 1 to 4.

Create one useful point for EACH driver.

Follow driver rank order.

Do more than say:
"X increased by Y."

Where data supports it, add:
- previous-month context,
- budget position,
- multi-month movement,
- peak/low,
- consecutive movement,
- or relationship to the broader monthly picture.

Do not claim a driver CAUSED a headline KPI unless causality is
explicitly supplied. Use wording such as "alongside", "while", or
"the largest underlying movement was" when appropriate.

Each point should normally be around 18-40 words.

============================================================
TREND INSIGHT
============================================================

Return 1 or 2 points.

Use actual historical months and the supplied trend classification.

Look for meaningful patterns such as:
- consistent increase,
- consistent decrease,
- reversal,
- peak,
- low,
- consecutive increases/decreases,
- movement from the first available month.

Do not call something a trend if there is insufficient history.

For the first available month, it is acceptable to return only one
trend point explaining that an intra-year trend is not yet established.

Do not repeat the Executive Summary.

============================================================
BUDGET & YTD
============================================================

Return 1 or 2 points.

This section should answer:
"What does the current month look like against plan, and does the
cumulative YTD position tell the same story?"

Prioritise differences between:
- monthly vs budget
- YTD vs budget
- monthly direction vs cumulative direction

This section should add interpretation, not repeat earlier sentences.

============================================================
STYLE
============================================================

Write for a senior finance/dashboard audience.

Be concise, clear, factual and numerically specific.

Do not use headings inside the individual points.

Do not use bullet symbols inside the strings.

Do not use Markdown.

Do not begin every point with the same metric name.

Return only the JSON structure requested by the schema.
"""


# ============================================================
# WORDING WE DO NOT WANT
# ============================================================

BANNED_PHRASES = [
    "due to",
    "caused by",
    "driven by",
    "because of",
    "resulting from",
    "seasonality",
    "seasonal",
    "helping absorb",
    "supported by",
    "favorable",
    "favourable",
    "unfavorable",
    "unfavourable",
    "outperformed",
    "underperformed",
]


# ============================================================
# GENERIC HELPERS
# ============================================================

def is_missing(value):

    if value is None:
        return True

    try:
        return bool(
            pd.isna(value)
        )

    except Exception:
        return False


def safe_float(value):

    if is_missing(value):
        return None

    try:
        return float(value)

    except Exception:
        return None


def safe_int(value):

    if is_missing(value):
        return None

    try:
        return int(float(value))

    except Exception:
        return None


def clean_string(value):

    if is_missing(value):
        return ""

    return str(value).strip()


# ============================================================
# NUMBER FORMATTING
#
# We format the financial numbers BEFORE Gemini.
#
# Gemini therefore does not need to decide:
# 638876900 = 638.88m
#
# Python does it.
# ============================================================

def compact_number(value):

    value = safe_float(
        value
    )

    if value is None:
        return None


    absolute = abs(value)


    if absolute >= 1_000_000_000:

        scaled = (
            absolute
            /
            1_000_000_000
        )

        suffix = "bn"


    elif absolute >= 1_000_000:

        scaled = (
            absolute
            /
            1_000_000
        )

        suffix = "m"


    elif absolute >= 1_000:

        scaled = (
            absolute
            /
            1_000
        )

        suffix = "k"


    else:

        scaled = absolute

        suffix = ""


    text = (
        f"{scaled:.2f}"
        .rstrip("0")
        .rstrip(".")
    )


    return (
        text
        +
        suffix
    )


def format_amount(
    value,
    currency,
):

    value = safe_float(
        value
    )

    if value is None:
        return None


    number = compact_number(
        value
    )


    negative = (
        value < 0
    )


    if currency == "EUR":

        if negative:
            return f"-€{number}"

        return f"€{number}"


    # Local Currency.
    # We deliberately do NOT guess GBP/ZAR/etc.
    if negative:
        return f"-LC {number}"

    return f"LC {number}"


def format_absolute_amount(
    value,
    currency,
):

    value = safe_float(
        value
    )

    if value is None:
        return None

    return format_amount(
        abs(value),
        currency,
    )


def format_pct(
    value,
):

    value = safe_float(
        value
    )

    if value is None:
        return None


    text = (
        f"{abs(value):.2f}"
        .rstrip("0")
        .rstrip(".")
    )


    if value < 0:
        return f"-{text}%"

    return f"{text}%"


def format_absolute_pct(
    value,
):

    value = safe_float(
        value
    )

    if value is None:
        return None


    text = (
        f"{abs(value):.2f}"
        .rstrip("0")
        .rstrip(".")
    )


    return f"{text}%"


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


    string_columns = [
        "Selected_Month",
        "Entity_CD",
        "Entity_Name",
        "Currency",
        "Section",
        "Metric",
        "Account_CD",
        "Account_Name",
    ]


    for column in string_columns:

        if column in insight_df.columns:

            insight_df[column] = (
                insight_df[column]
                .fillna("")
                .astype(str)
                .str.strip()
            )


    trend_string_columns = [
        "Selected_Month",
        "Entity_CD",
        "Entity_Name",
        "Currency",
        "Metric_Type",
        "Metric",
        "Account_CD",
        "Account_Name",
        "Trend_Month",
    ]


    for column in trend_string_columns:

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
# BUILD ONE FACT OBJECT
# ============================================================

def build_fact(
    row,
):

    currency = clean_string(
        row.get(
            "Currency"
        )
    )


    actual = safe_float(
        row.get(
            "Actual"
        )
    )


    previous_actual = safe_float(
        row.get(
            "Previous_Actual"
        )
    )


    mom_change = safe_float(
        row.get(
            "MoM_Change"
        )
    )


    mom_pct = safe_float(
        row.get(
            "MoM_Pct_Safe"
        )
    )


    budget = safe_float(
        row.get(
            "Budget"
        )
    )


    vs_budget = safe_float(
        row.get(
            "Vs_Budget"
        )
    )


    yoy = safe_float(
        row.get(
            "YoY_Abs"
        )
    )


    ytd_actual = safe_float(
        row.get(
            "YTD_Actual"
        )
    )


    ytd_budget = safe_float(
        row.get(
            "YTD_Budget"
        )
    )


    ytd_vs_budget = safe_float(
        row.get(
            "YTD_Vs_Budget"
        )
    )


    ytd_yoy = safe_float(
        row.get(
            "YTD_YoY_Abs"
        )
    )


    first_actual = safe_float(
        row.get(
            "First_Actual"
        )
    )


    change_from_first = safe_float(
        row.get(
            "Change_From_First"
        )
    )


    change_from_first_pct = safe_float(
        row.get(
            "Change_From_First_Pct"
        )
    )


    peak_actual = safe_float(
        row.get(
            "Peak_Actual"
        )
    )


    lowest_actual = safe_float(
        row.get(
            "Lowest_Actual"
        )
    )


    return {

        "section":
            clean_string(
                row.get(
                    "Section"
                )
            ),

        "rank":
            safe_int(
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

        "account_cd":
            clean_string(
                row.get(
                    "Account_CD"
                )
            ),

        "account_name":
            clean_string(
                row.get(
                    "Account_Name"
                )
            ),


        # ====================================================
        # CURRENT MONTH
        # ====================================================

        "current": {

            "actual_raw":
                actual,

            "actual_display":
                format_amount(
                    actual,
                    currency,
                ),
        },


        # ====================================================
        # MONTH ON MONTH
        # ====================================================

        "month_on_month": {

            "previous_month":
                clean_string(
                    row.get(
                        "Previous_Month"
                    )
                ),

            "previous_actual_raw":
                previous_actual,

            "previous_actual_display":
                format_amount(
                    previous_actual,
                    currency,
                ),

            "change_raw":
                mom_change,

            "change_display":
                format_amount(
                    mom_change,
                    currency,
                ),

            "change_absolute_display":
                format_absolute_amount(
                    mom_change,
                    currency,
                ),

            "pct_raw":
                mom_pct,

            "pct_display":
                format_pct(
                    mom_pct
                ),

            "pct_absolute_display":
                format_absolute_pct(
                    mom_pct
                ),

            "direction":
                clean_string(
                    row.get(
                        "MoM_Direction"
                    )
                ),
        },


        # ====================================================
        # MONTHLY BUDGET
        # ====================================================

        "monthly_budget": {

            "budget_raw":
                budget,

            "budget_display":
                format_amount(
                    budget,
                    currency,
                ),

            "variance_raw":
                vs_budget,

            "variance_display":
                format_amount(
                    vs_budget,
                    currency,
                ),

            "variance_absolute_display":
                format_absolute_amount(
                    vs_budget,
                    currency,
                ),

            "position":
                clean_string(
                    row.get(
                        "Budget_Position"
                    )
                ),
        },


        # ====================================================
        # YOY
        # ====================================================

        "year_on_year": {

            "absolute_change_raw":
                yoy,

            "absolute_change_display":
                format_amount(
                    yoy,
                    currency,
                ),

            "absolute_magnitude_display":
                format_absolute_amount(
                    yoy,
                    currency,
                ),
        },


        # ====================================================
        # YTD
        # ====================================================

        "ytd": {

            "actual_raw":
                ytd_actual,

            "actual_display":
                format_amount(
                    ytd_actual,
                    currency,
                ),

            "budget_raw":
                ytd_budget,

            "budget_display":
                format_amount(
                    ytd_budget,
                    currency,
                ),

            "variance_raw":
                ytd_vs_budget,

            "variance_display":
                format_amount(
                    ytd_vs_budget,
                    currency,
                ),

            "variance_absolute_display":
                format_absolute_amount(
                    ytd_vs_budget,
                    currency,
                ),

            "budget_position":
                clean_string(
                    row.get(
                        "YTD_Budget_Position"
                    )
                ),

            "yoy_absolute_raw":
                ytd_yoy,

            "yoy_absolute_display":
                format_amount(
                    ytd_yoy,
                    currency,
                ),
        },


        # ====================================================
        # MULTI-MONTH HISTORY CONTEXT
        # ====================================================

        "history_context": {

            "history_months":
                safe_int(
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

            "first_actual_display":
                format_amount(
                    first_actual,
                    currency,
                ),

            "change_from_first_display":
                format_amount(
                    change_from_first,
                    currency,
                ),

            "change_from_first_pct":
                format_pct(
                    change_from_first_pct
                ),

            "peak_month":
                clean_string(
                    row.get(
                        "Peak_Month"
                    )
                ),

            "peak_actual_display":
                format_amount(
                    peak_actual,
                    currency,
                ),

            "lowest_month":
                clean_string(
                    row.get(
                        "Lowest_Month"
                    )
                ),

            "lowest_actual_display":
                format_amount(
                    lowest_actual,
                    currency,
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
                safe_int(
                    row.get(
                        "Consecutive_Count"
                    )
                ),
        },


        # ====================================================
        # RANKINGS
        # ====================================================

        "importance": {

            "materiality_rank":
                safe_int(
                    row.get(
                        "Materiality_Rank"
                    )
                ),

            "movement_rank":
                safe_int(
                    row.get(
                        "Movement_Rank"
                    )
                ),

            "budget_variance_rank":
                safe_int(
                    row.get(
                        "Budget_Variance_Rank"
                    )
                ),
        },
    }


# ============================================================
# TREND HISTORY FOR ONE ACCOUNT
# ============================================================

def build_monthly_history(
    trend_subset,
    account_cd,
    currency,
):

    history = (
        trend_subset[
            trend_subset[
                "Account_CD"
            ]
            ==
            account_cd
        ]
        .sort_values(
            "Trend_Month_Order"
        )
    )


    output = []


    for _, row in history.iterrows():

        output.append(
            {

                "month":
                    clean_string(
                        row.get(
                            "Trend_Month"
                        )
                    ),

                "month_order":
                    safe_int(
                        row.get(
                            "Trend_Month_Order"
                        )
                    ),

                "actual":
                    format_amount(
                        row.get(
                            "Actual"
                        ),
                        currency,
                    ),

                "budget":
                    format_amount(
                        row.get(
                            "Budget"
                        ),
                        currency,
                    ),

                "vs_budget":
                    format_amount(
                        row.get(
                            "Vs_Budget"
                        ),
                        currency,
                    ),

                "yoy_absolute":
                    format_amount(
                        row.get(
                            "YoY_Abs"
                        ),
                        currency,
                    ),

                "ytd_actual":
                    format_amount(
                        row.get(
                            "YTD_Actual"
                        ),
                        currency,
                    ),

                "ytd_budget":
                    format_amount(
                        row.get(
                            "YTD_Budget"
                        ),
                        currency,
                    ),

                "ytd_vs_budget":
                    format_amount(
                        row.get(
                            "YTD_Vs_Budget"
                        ),
                        currency,
                    ),
            }
        )


    return output


# ============================================================
# BUILD COMPLETE GEMINI FACT PACKAGE
# ============================================================

def build_fact_package(
    insight_subset,
    trend_subset,
):

    insight_subset = (
        insight_subset
        .sort_values(
            [
                "Section",
                "Rank",
            ]
        )
        .copy()
    )


    first = insight_subset.iloc[0]


    currency = clean_string(
        first[
            "Currency"
        ]
    )


    headline_facts = []

    driver_facts = []


    for _, row in insight_subset.iterrows():

        fact = build_fact(
            row
        )


        account_cd = clean_string(
            row[
                "Account_CD"
            ]
        )


        fact[
            "monthly_history"
        ] = build_monthly_history(

            trend_subset=

                trend_subset,

            account_cd=

                account_cd,

            currency=

                currency,
        )


        if (
            row[
                "Section"
            ]
            ==
            "HEADLINE"
        ):

            headline_facts.append(
                fact
            )


        elif (
            row[
                "Section"
            ]
            ==
            "DRIVER"
        ):

            driver_facts.append(
                fact
            )


    headline_facts = sorted(
        headline_facts,
        key=lambda x:
            x[
                "rank"
            ]
            if x[
                "rank"
            ]
            is not None
            else 999,
    )


    driver_facts = sorted(
        driver_facts,
        key=lambda x:
            x[
                "rank"
            ]
            if x[
                "rank"
            ]
            is not None
            else 999,
    )


    return {

        "context": {

            "fiscal_year":
                safe_int(
                    first[
                        "Fiscal_Year"
                    ]
                ),

            "selected_month":
                clean_string(
                    first[
                        "Selected_Month"
                    ]
                ),

            "selected_month_order":
                safe_int(
                    first[
                        "Selected_Month_Order"
                    ]
                ),

            "entity_cd":
                clean_string(
                    first[
                        "Entity_CD"
                    ]
                ),

            "entity_name":
                clean_string(
                    first[
                        "Entity_Name"
                    ]
                ),

            "currency":
                currency,

            "currency_display_rule":
                (
                    "Use € for supplied EUR values."
                    if currency == "EUR"
                    else
                    "Use LC exactly as supplied. "
                    "Do not guess the local currency."
                ),
        },

        "headline_kpis":
            headline_facts,

        "top_4_drivers":
            driver_facts,
    }


# ============================================================
# PROMPT
# ============================================================

def build_prompt(
    fact_package,
    previous_issues=None,
):

    prompt = f"""
Analyse the following structured Cash Flow facts and create the
Iteration-2 popup commentary.

The output must follow these section counts:

EXECUTIVE SUMMARY:
- exactly 2 points

KEY MOVEMENTS:
- exactly 4 points
- one point per supplied driver
- follow driver rank 1, 2, 3, 4

TREND INSIGHT:
- 1 or 2 points

BUDGET & YTD:
- 1 or 2 points

The purpose of this popup is that a finance user should be able to
open it and understand the selected month's financial story without
having to interpret every dashboard number manually.

Do NOT merely repeat values.

Use the supplied comparisons and historical context to determine:
- what changed materially,
- whether the movement continued or reversed,
- how the result sits against budget,
- whether the YTD picture differs from the monthly result,
- and which underlying movements deserve attention.

IMPORTANT:
`raw` values exist only for factual grounding.
When quoting amounts, use the corresponding `display` values.

Here are the facts:

{json.dumps(
    fact_package,
    indent=2,
    ensure_ascii=False
)}
"""


    if previous_issues:

        prompt += f"""

Your previous response had these quality/structure problems:

{previous_issues}

Correct those problems in this response.
"""


    return prompt


# ============================================================
# LOCAL OUTPUT VALIDATION
# ============================================================

def normalize_point(
    point,
):

    point = clean_string(
        point
    )


    # Remove Gemini-created bullet markers
    point = re.sub(
        r"^[•\-\*\s]+",
        "",
        point,
    )


    # Collapse whitespace
    point = re.sub(
        r"\s+",
        " ",
        point,
    )


    return point.strip()


def normalize_generated_output(
    result,
):

    normalized = {}


    for section in [
        "executive_summary",
        "key_movements",
        "trend_insight",
        "budget_ytd",
    ]:

        values = result.get(
            section,
            []
        )


        if not isinstance(
            values,
            list,
        ):

            values = []


        normalized[
            section
        ] = [

            normalize_point(
                value
            )

            for value in values

            if normalize_point(
                value
            )
        ]


    return normalized


def validate_generated_output(
    result,
):

    issues = []


    # ========================================================
    # STRUCTURE
    # ========================================================

    if len(
        result[
            "executive_summary"
        ]
    ) != 2:

        issues.append(
            "Executive Summary must contain exactly 2 points."
        )


    if len(
        result[
            "key_movements"
        ]
    ) != 4:

        issues.append(
            "Key Movements must contain exactly 4 points."
        )


    if not (
        1
        <=
        len(
            result[
                "trend_insight"
            ]
        )
        <=
        2
    ):

        issues.append(
            "Trend Insight must contain 1 or 2 points."
        )


    if not (
        1
        <=
        len(
            result[
                "budget_ytd"
            ]
        )
        <=
        2
    ):

        issues.append(
            "Budget & YTD must contain 1 or 2 points."
        )


    # ========================================================
    # DUPLICATE POINTS
    # ========================================================

    all_points = (
        result[
            "executive_summary"
        ]
        +
        result[
            "key_movements"
        ]
        +
        result[
            "trend_insight"
        ]
        +
        result[
            "budget_ytd"
        ]
    )


    normalized_points = [
        re.sub(
            r"[^a-z0-9]+",
            " ",
            point.lower(),
        ).strip()

        for point in all_points
    ]


    if (
        len(
            normalized_points
        )
        !=
        len(
            set(
                normalized_points
            )
        )
    ):

        issues.append(
            "Duplicate commentary points were produced."
        )


    # ========================================================
    # BANNED / UNSUPPORTED WORDING
    # ========================================================

    joined_text = (
        " "
        .join(
            all_points
        )
        .lower()
    )


    for phrase in BANNED_PHRASES:

        if phrase.lower() in joined_text:

            issues.append(
                f"Unsupported/vague wording detected: '{phrase}'."
            )


    # ========================================================
    # VERY SHORT POINTS
    #
    # This stops us getting:
    # "EBITDA increased 3.5%."
    # ========================================================

    for index, point in enumerate(
        all_points,
        start=1,
    ):

        word_count = len(
            point.split()
        )


        if word_count < 9:

            issues.append(
                f"Point {index} is too short to be insightful."
            )


        if word_count > 55:

            issues.append(
                f"Point {index} is too long for the popup."
            )


    return issues


# ============================================================
# GENERATE ONE SUMMARY
# ============================================================

def generate_summary(
    client,
    fact_package,
):

    previous_issues = None

    last_result = None

    last_issues = []


    for attempt in range(
        1,
        MAX_ATTEMPTS + 1,
    ):

        prompt = build_prompt(
            fact_package=
                fact_package,

            previous_issues=
                previous_issues,
        )


        response = (
            client.models.generate_content(

                model=
                    GEMINI_MODEL,

                contents=
                    prompt,

                config=
                    types.GenerateContentConfig(

                        system_instruction=
                            SYSTEM_INSTRUCTION,

                        thinking_config=
                            types.ThinkingConfig(
                                thinking_level=
                                    THINKING_LEVEL
                            ),

                        response_mime_type=
                            "application/json",

                        response_schema=
                            RESPONSE_SCHEMA,

                        max_output_tokens=
                            MAX_OUTPUT_TOKENS,
                    ),
            )
        )


        if not response.text:

            previous_issues = (
                "The model returned an empty response."
            )

            continue


        try:

            result = json.loads(
                response.text
            )


        except json.JSONDecodeError as error:

            previous_issues = (
                "The previous response was not valid JSON: "
                f"{error}"
            )

            continue


        result = normalize_generated_output(
            result
        )


        issues = validate_generated_output(
            result
        )


        last_result = result

        last_issues = issues


        if not issues:

            return (
                result,
                "PASS",
                "",
                attempt,
            )


        previous_issues = (
            "\n".join(
                f"- {issue}"
                for issue in issues
            )
        )


    # ========================================================
    # If the third attempt is usable but has a softer quality
    # warning, keep it for POC review instead of losing it.
    # ========================================================

    if last_result is not None:

        return (
            last_result,
            "WARN",
            " | ".join(
                last_issues
            ),
            MAX_ATTEMPTS,
        )


    raise RuntimeError(
        "Gemini failed to return a usable structured response "
        f"after {MAX_ATTEMPTS} attempts."
    )


# ============================================================
# POPUP COLUMNS
# ============================================================

def point_at(
    points,
    index,
):

    if index < len(points):
        return points[index]

    return ""


def build_summary_full(
    result,
):

    sections = []


    sections.append(
        "EXECUTIVE SUMMARY\n"
        +
        "\n".join(
            f"• {point}"
            for point in result[
                "executive_summary"
            ]
        )
    )


    sections.append(
        "KEY MOVEMENTS\n"
        +
        "\n".join(
            f"• {point}"
            for point in result[
                "key_movements"
            ]
        )
    )


    sections.append(
        "TREND INSIGHT\n"
        +
        "\n".join(
            f"• {point}"
            for point in result[
                "trend_insight"
            ]
        )
    )


    sections.append(
        "BUDGET & YTD\n"
        +
        "\n".join(
            f"• {point}"
            for point in result[
                "budget_ytd"
            ]
        )
    )


    return (
        "\n\n"
        .join(
            sections
        )
    )


# ============================================================
# KEY
# ============================================================

def row_key(
    fiscal_year,
    month,
    entity_cd,
    currency,
):

    return (
        safe_int(
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
# EXISTING OUTPUT
# ============================================================

def load_existing_output():

    if not OUTPUT_FILE.exists():

        return pd.DataFrame()


    return pd.read_csv(
        OUTPUT_FILE,
        dtype={
            "Entity_CD": str,
        },
    )


def existing_keys(
    df,
):

    if df.empty:

        return set()


    keys = set()


    for _, row in df.iterrows():

        keys.add(
            row_key(
                row.get(
                    "Fiscal_Year"
                ),
                row.get(
                    "Month"
                ),
                row.get(
                    "Entity_CD"
                ),
                row.get(
                    "Currency"
                ),
            )
        )


    return keys


# ============================================================
# SAVE
# ============================================================

def save_output(
    output_df,
):

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    if not output_df.empty:

        output_df = (
            output_df
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


    output_df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )


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
# ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Regenerate selected combinations even if "
            "they already exist."
        ),
    )


    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Generate only N combinations. "
            "Useful for quality testing."
        ),
    )


    parser.add_argument(
        "--entity",
        type=str,
        default=None,
        help=(
            "Optional Entity_CD or exact Entity_Name filter."
        ),
    )


    parser.add_argument(
        "--month",
        type=str,
        default=None,
        help=(
            "Optional month filter, e.g. Sep."
        ),
    )


    parser.add_argument(
        "--currency",
        type=str,
        choices=[
            "EUR",
            "LC",
        ],
        default=None,
    )


    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()


    print()
    print("=" * 90)
    print("CASHFLOW AI - ITERATION 2")
    print("RICH POPUP COMMENTARY GENERATOR")
    print("=" * 90)


    # ========================================================
    # INPUT
    # ========================================================

    insight_df, trend_df = (
        load_inputs()
    )


    print(
        f"\nInsight rows: "
        f"{len(insight_df)}"
    )

    print(
        f"Trend rows: "
        f"{len(trend_df)}"
    )


    # ========================================================
    # SUMMARY KEYS
    #
    # Grain:
    # FY × Month × Entity × Currency
    # ========================================================

    key_columns = [
        "Fiscal_Year",
        "Selected_Month",
        "Selected_Month_Order",
        "Entity_CD",
        "Entity_Name",
        "Currency",
    ]


    summary_keys = (
        insight_df[
            key_columns
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

    if args.entity:

        entity_filter = (
            args.entity
            .strip()
            .lower()
        )


        summary_keys = (
            summary_keys[
                (
                    summary_keys[
                        "Entity_CD"
                    ]
                    .str.lower()
                    ==
                    entity_filter
                )
                |
                (
                    summary_keys[
                        "Entity_Name"
                    ]
                    .str.lower()
                    ==
                    entity_filter
                )
            ]
        )


    if args.month:

        summary_keys = (
            summary_keys[
                summary_keys[
                    "Selected_Month"
                ]
                .str.lower()
                ==
                args.month
                .strip()
                .lower()
            ]
        )


    if args.currency:

        summary_keys = (
            summary_keys[
                summary_keys[
                    "Currency"
                ]
                ==
                args.currency
            ]
        )


    if args.limit is not None:

        summary_keys = (
            summary_keys
            .head(
                args.limit
            )
        )


    print(
        f"\nSelected combinations: "
        f"{len(summary_keys)}"
    )


    if summary_keys.empty:

        print(
            "No matching combinations found."
        )

        return


    # ========================================================
    # GEMINI CLIENT
    #
    # vertexai=True is intentionally used because it works
    # across older/newer google-genai versions with ADC.
    # ========================================================

    client = genai.Client(

        vertexai=True,

        project=
            PROJECT_ID,

        location=
            VERTEX_LOCATION,
    )


    print(
        f"Generator Model: "
        f"{GEMINI_MODEL}"
    )

    print(
        f"Thinking Level: "
        f"{THINKING_LEVEL}"
    )


    # ========================================================
    # EXISTING OUTPUT
    # ========================================================

    output_df = (
        load_existing_output()
    )


    completed_keys = (
        existing_keys(
            output_df
        )
    )


    generated_count = 0

    skipped_count = 0

    warning_count = 0

    failure_count = 0


    total = len(
        summary_keys
    )


    # ========================================================
    # GENERATION LOOP
    # ========================================================

    for position, key_row in (
        summary_keys.iterrows()
    ):

        fiscal_year = (
            key_row[
                "Fiscal_Year"
            ]
        )

        month = (
            key_row[
                "Selected_Month"
            ]
        )

        month_order = (
            key_row[
                "Selected_Month_Order"
            ]
        )

        entity_cd = (
            key_row[
                "Entity_CD"
            ]
        )

        entity_name = (
            key_row[
                "Entity_Name"
            ]
        )

        currency = (
            key_row[
                "Currency"
            ]
        )


        key = row_key(
            fiscal_year,
            month,
            entity_cd,
            currency,
        )


        display_number = (
            generated_count
            +
            skipped_count
            +
            failure_count
            +
            1
        )


        print()
        print(
            f"[{display_number}/{total}] "
            f"{entity_name} | "
            f"{month} | "
            f"{currency}"
        )


        # ====================================================
        # SKIP EXISTING
        # ====================================================

        if (
            key in completed_keys
            and
            not args.force
        ):

            print(
                "  SKIP - already generated"
            )

            skipped_count += 1

            continue


        # ====================================================
        # IF FORCE, REMOVE OLD VERSION OF THIS KEY
        # ====================================================

        if (
            args.force
            and
            not output_df.empty
        ):

            match = (
                (
                    output_df[
                        "Fiscal_Year"
                    ]
                    .apply(
                        safe_int
                    )
                    ==
                    safe_int(
                        fiscal_year
                    )
                )
                &
                (
                    output_df[
                        "Month"
                    ]
                    ==
                    month
                )
                &
                (
                    output_df[
                        "Entity_CD"
                    ]
                    .astype(str)
                    ==
                    str(
                        entity_cd
                    )
                )
                &
                (
                    output_df[
                        "Currency"
                    ]
                    ==
                    currency
                )
            )


            output_df = (
                output_df[
                    ~match
                ]
                .copy()
            )


        # ====================================================
        # INSIGHT SUBSET
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


        try:

            fact_package = (
                build_fact_package(
                    insight_subset=
                        insight_subset,

                    trend_subset=
                        trend_subset,
                )
            )


            result, status, issues, attempts = (
                generate_summary(

                    client=
                        client,

                    fact_package=
                        fact_package,
                )
            )


            full_summary = (
                build_summary_full(
                    result
                )
            )


            all_point_text = (
                result[
                    "executive_summary"
                ]
                +
                result[
                    "key_movements"
                ]
                +
                result[
                    "trend_insight"
                ]
                +
                result[
                    "budget_ytd"
                ]
            )


            word_count = sum(
                len(
                    point.split()
                )

                for point
                in all_point_text
            )


            generated_at = (
                datetime.now(
                    timezone.utc
                )
                .isoformat()
            )


            output_row = {

                "Fiscal_Year":
                    safe_int(
                        fiscal_year
                    ),

                "Month":
                    month,

                "Month_Order":
                    safe_int(
                        month_order
                    ),

                "Entity_CD":
                    entity_cd,

                "Entity_Name":
                    entity_name,

                "Currency":
                    currency,


                # ==========================================
                # EXECUTIVE
                # ==========================================

                "Executive_Point_1":
                    point_at(
                        result[
                            "executive_summary"
                        ],
                        0,
                    ),

                "Executive_Point_2":
                    point_at(
                        result[
                            "executive_summary"
                        ],
                        1,
                    ),


                # ==========================================
                # KEY MOVEMENTS
                # ==========================================

                "Key_Movement_1":
                    point_at(
                        result[
                            "key_movements"
                        ],
                        0,
                    ),

                "Key_Movement_2":
                    point_at(
                        result[
                            "key_movements"
                        ],
                        1,
                    ),

                "Key_Movement_3":
                    point_at(
                        result[
                            "key_movements"
                        ],
                        2,
                    ),

                "Key_Movement_4":
                    point_at(
                        result[
                            "key_movements"
                        ],
                        3,
                    ),


                # ==========================================
                # TREND
                # ==========================================

                "Trend_Point_1":
                    point_at(
                        result[
                            "trend_insight"
                        ],
                        0,
                    ),

                "Trend_Point_2":
                    point_at(
                        result[
                            "trend_insight"
                        ],
                        1,
                    ),


                # ==========================================
                # BUDGET / YTD
                # ==========================================

                "Budget_YTD_Point_1":
                    point_at(
                        result[
                            "budget_ytd"
                        ],
                        0,
                    ),

                "Budget_YTD_Point_2":
                    point_at(
                        result[
                            "budget_ytd"
                        ],
                        1,
                    ),


                # ==========================================
                # FULL TEXT FOR REVIEW / DEBUGGING
                # ==========================================

                "Summary_Full":
                    full_summary,

                "Word_Count":
                    word_count,

                "Review_Status":
                    status,

                "Review_Issues":
                    issues,

                "Generation_Attempts":
                    attempts,

                "Generated_At":
                    generated_at,

                "Generator_Model":
                    GEMINI_MODEL,

                "Generator_Thinking":
                    THINKING_LEVEL,
            }


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


            completed_keys.add(
                key
            )


            save_output(
                output_df
            )


            append_log(
                {

                    "Fiscal_Year":
                        safe_int(
                            fiscal_year
                        ),

                    "Month":
                        month,

                    "Entity_CD":
                        entity_cd,

                    "Entity_Name":
                        entity_name,

                    "Currency":
                        currency,

                    "Status":
                        status,

                    "Issues":
                        issues,

                    "Attempts":
                        attempts,

                    "Generated_At":
                        generated_at,
                }
            )


            if status == "WARN":

                warning_count += 1

                print(
                    f"  WARN - {issues}"
                )


            else:

                print(
                    "  PASS"
                )


            generated_count += 1


        except Exception as error:

            failure_count += 1


            print(
                f"  FAILED - {error}"
            )


            append_log(
                {

                    "Fiscal_Year":
                        safe_int(
                            fiscal_year
                        ),

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

                    "Issues":
                        str(
                            error
                        ),

                    "Attempts":
                        MAX_ATTEMPTS,

                    "Generated_At":
                        datetime.now(
                            timezone.utc
                        ).isoformat(),
                }
            )


    # ========================================================
    # FINAL STATUS
    # ========================================================

    print()
    print("=" * 90)
    print("ITERATION-2 GENERATION COMPLETE")
    print("=" * 90)

    print(
        f"Generated: {generated_count}"
    )

    print(
        f"Skipped:   {skipped_count}"
    )

    print(
        f"Warnings:  {warning_count}"
    )

    print(
        f"Failures:  {failure_count}"
    )

    print(
        f"\nOutput:\n"
        f"{OUTPUT_FILE.resolve()}"
    )

    print(
        f"\nLog:\n"
        f"{LOG_FILE.resolve()}"
    )


if __name__ == "__main__":

    main()
