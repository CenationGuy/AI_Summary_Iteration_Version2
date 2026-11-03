import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
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

MAX_DISPLAY_MOM_PCT = 500


# ============================================================
# STRUCTURED OUTPUT SCHEMA
#
# IMPORTANT:
# We use explicit fields instead of arrays.
#
# This is more reliable than asking Gemini for:
#
# executive_summary: [...]
# key_movements: [...]
#
# Every successful response has the exact same structure.
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


# ============================================================
# REQUIRED OUTPUT FIELDS
# ============================================================

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
# GOOGLE GENAI CLIENT
# ============================================================

def create_client():

    return genai.Client(
        vertexai=True,
        project=PROJECT_ID,
        location=VERTEX_LOCATION,
    )


# ============================================================
# UTILITIES
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


# ============================================================
# CONVERT CASH VALUES TO MILLIONS
#
# The raw CF table contains full currency values.
#
# Example:
# 638,876,900
#
# becomes:
# 638.88
#
# Gemini is explicitly told that these numbers are millions
# of the selected currency.
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


def rounded_pct(value):

    value = safe_float(
        value
    )

    if value is None:
        return None

    if abs(value) > MAX_DISPLAY_MOM_PCT:
        return None

    return round(
        value,
        2,
    )


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
# READ INPUTS
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

        "Entity_CD",
        "Entity_Name",
        "Currency",
        "Selected_Month",
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
# BUILD CURRENT FACT
# ============================================================

def build_current_fact(row):

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

        # ==============================================
        # CURRENT MONTH
        # ==============================================

        "actual_m":
            to_millions(
                row.get(
                    "Actual"
                )
            ),

        "previous_month":
            clean_string(
                row.get(
                    "Previous_Month"
                )
            ),

        "previous_actual_m":
            to_millions(
                row.get(
                    "Previous_Actual"
                )
            ),

        "mom_change_m":
            to_millions(
                row.get(
                    "MoM_Change"
                )
            ),

        "mom_pct":
            rounded_pct(
                row.get(
                    "MoM_Pct_Safe"
                )
            ),

        "mom_direction":
            clean_string(
                row.get(
                    "MoM_Direction"
                )
            ),

        # ==============================================
        # BUDGET
        # ==============================================

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

        # ==============================================
        # YOY
        # ==============================================

        "yoy_change_m":
            to_millions(
                row.get(
                    "YoY_Abs"
                )
            ),

        # ==============================================
        # YTD
        # ==============================================

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

        # ==============================================
        # MULTI-MONTH ANALYSIS
        # ==============================================

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

        # ==============================================
        # TREND CONTEXT
        # ==============================================

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
                row.get(
                    "First_Actual"
                )
            ),

        "change_from_first_m":
            to_millions(
                row.get(
                    "Change_From_First"
                )
            ),

        "change_from_first_pct":
            rounded_pct(
                row.get(
                    "Change_From_First_Pct"
                )
            ),

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

        # ==============================================
        # IMPORTANCE
        # ==============================================

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


    # Remove only None values.
    #
    # Keep zero because zero is meaningful financial data.

    return {
        key: value
        for key, value in fact.items()
        if value is not None
    }


# ============================================================
# BUILD TREND FACTS
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


        monthly_history = []


        for _, row in group.iterrows():

            monthly_history.append(
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
                    monthly_history,
            }
        )


    return output


# ============================================================
# BUILD COMPLETE FACT PACKAGE
# ============================================================

def build_fact_package(
    insight_subset,
    trend_subset,
):

    first = insight_subset.iloc[0]


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
    )


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
    )


    headline_facts = [

        build_current_fact(
            row
        )

        for _, row in (
            headline_df.iterrows()
        )
    ]


    driver_facts = [

        build_current_fact(
            row
        )

        for _, row in (
            driver_df.iterrows()
        )
    ]


    trend_facts = (
        build_trend_facts(
            trend_subset
        )
    )


    package = {

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
                (
                    "millions of the selected "
                    "currency"
                ),
        },

        "headline_kpis":
            headline_facts,

        "top_4_material_drivers":
            driver_facts,

        "historical_monthly_values":
            trend_facts,
    }


    return package


# ============================================================
# PROMPT
# ============================================================

def build_prompt(
    fact_package,
    retry_note="",
):

    context = fact_package[
        "context"
    ]


    currency = context[
        "currency"
    ]


    retry_section = ""


    if retry_note:

        retry_section = f"""

PREVIOUS ATTEMPT ISSUE
----------------------
{retry_note}

Correct that issue in this attempt.
"""


    prompt = f"""
You are an executive cash-flow analyst.

You are writing AI commentary for an SAP Analytics Cloud
cash-flow dashboard.

Selected context:

Fiscal Year: {context['fiscal_year']}
Month: {context['month']}
Entity: {context['entity']}
Currency: {currency}

All monetary numbers supplied below are already expressed in
MILLIONS of the selected currency.

For EUR:
write monetary amounts as "EUR 25.4m".

For LC:
write monetary amounts as "LC 25.4m".

Do not convert between currencies.


============================================================
OBJECTIVE
============================================================

Create concise but genuinely insightful executive commentary.

A reader opening the popup should understand:

1. What materially happened in the selected month.
2. Which headline KPIs changed the most or matter most.
3. Which underlying supplied drivers were significant.
4. Whether the latest month continues or reverses a trend.
5. How performance compares with monthly budget.
6. How the year-to-date position compares with budget.
7. Relevant YoY context where supplied.


============================================================
VERY IMPORTANT WRITING STANDARD
============================================================

Do NOT merely restate one number per sentence.

Weak:
"EBITDA decreased by 3.6%."

Better:
"EBITDA declined 3.6% MoM to EUR 113.7m, extending the recent
downward direction while remaining EUR 8.2m below budget."

The better sentence combines multiple supported facts.

Look for meaningful relationships across the supplied facts.

Examples of useful relationships:

- current month versus previous month
- current value versus budget
- YTD value versus YTD budget
- current month versus available FY history
- reversal after previous increases/decreases
- consecutive monthly movements
- latest value being the highest or lowest in available history
- headline movement occurring alongside a material component movement
- divergence between two headline KPIs
- strongest material driver among the supplied drivers

However, correlation is NOT causation.

You may say:

"Working Capital declined while WC Receivables also moved lower."

Do NOT say:

"WC Receivables caused the Working Capital decline"

unless that causal relationship is explicitly established by the
supplied data.


============================================================
SECTION REQUIREMENTS
============================================================

EXECUTIVE POINTS — 2

These are the two most important takeaways.

Each executive point should usually combine 2 or more relevant facts.

Do not repeat the same fact in both points.


KEY MOVEMENTS — 4

Use the supplied Top 4 material drivers and/or important headline
movements.

Explain why each movement stands out numerically.

Use comparisons such as MoM, budget, ranking, or historical position
when available.

Avoid four repetitive sentences with identical structure.


TREND POINTS — 2

Describe meaningful multi-month patterns.

Use facts such as:

CONSISTENT_INCREASE
CONSISTENT_DECREASE
REVERSAL_UP
REVERSAL_DOWN
MIXED

Also use peak/low month and consecutive movements where useful.

If this is the first available month and a trend cannot be established,
state that clearly rather than inventing a trend.

Do not predict future performance.


BUDGET & YTD POINTS — 2

Focus on:

monthly budget position
+
year-to-date budget position

Where useful, contrast monthly performance with cumulative YTD
performance.

Example:

"Although September FCF finished above monthly budget, YTD FCF
remains below plan."

Only make such a statement if the supplied facts support it.


============================================================
STRICT FACTUAL RULES
============================================================

Use ONLY the supplied data.

Never invent:

- causes
- operational explanations
- management actions
- forecasts
- missing numbers
- business events

Do not call a movement favourable, unfavourable, strong, weak,
healthy, poor, or concerning solely because the number increased
or decreased.

For example, a more negative cash-flow line is not automatically
bad unless the supplied business context proves that interpretation.

Use neutral language when business meaning is uncertain:

- increased
- decreased
- moved higher
- moved lower
- above budget
- below budget
- reversed
- remained stable
- widened
- narrowed


============================================================
NUMERIC RULES
============================================================

Round monetary values sensibly, normally to 1 or 2 decimals.

Do not calculate a MoM percentage when "mom_pct" is absent.

When a supplied percentage is absent because the prior-period base
was near zero, describe the absolute movement instead.

Do not invent percentages from the raw values.

Do not mix EUR and LC.

All facts in this request belong only to:
{context['entity']} | {context['month']} | {currency}


============================================================
ANTI-REPETITION RULE
============================================================

Each of the 10 points must add new information.

A number may appear again only when needed to connect a different
insight.

Do not repeat the exact same observation in Executive Summary,
Key Movements, Trend, and Budget/YTD.


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
"""


    return prompt.strip()


# ============================================================
# PARSE GEMINI RESPONSE
# ============================================================

def parse_response(
    response,
):

    # ========================================================
    # FIRST: SDK-parsed structured response
    # ========================================================

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


    # ========================================================
    # FALLBACK: response.text
    # ========================================================

    text = clean_string(
        getattr(
            response,
            "text",
            ""
        )
    )


    if not text:

        raise ValueError(
            "Gemini returned empty response text."
        )


    # Defensive cleanup in case fences appear despite
    # structured-output mode.

    if text.startswith(
        "```"
    ):

        text = text.strip("`")

        if text.lower().startswith(
            "json"
        ):

            text = text[4:].strip()


    return json.loads(
        text
    )


# ============================================================
# VALIDATE STRUCTURED RESPONSE
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
            "Structured response is not a JSON object."
        )


    missing = [

        field

        for field in OUTPUT_FIELDS

        if field not in result
    ]


    if missing:

        raise ValueError(
            "Missing output fields: "
            +
            ", ".join(
                missing
            )
        )


    cleaned = {}


    for field in OUTPUT_FIELDS:

        value = clean_string(
            result.get(
                field
            )
        )

        cleaned[
            field
        ] = value


    # ========================================================
    # CORE POINTS MUST ALWAYS EXIST
    # ========================================================

    mandatory_non_empty = [

        "executive_point_1",
        "executive_point_2",

        "key_movement_1",
        "key_movement_2",
        "key_movement_3",
        "key_movement_4",

        "budget_ytd_point_1",
        "budget_ytd_point_2",
    ]


    empty_mandatory = [

        field

        for field in mandatory_non_empty

        if not cleaned[
            field
        ]
    ]


    if empty_mandatory:

        raise ValueError(
            "Empty required commentary fields: "
            +
            ", ".join(
                empty_mandatory
            )
        )


    # ========================================================
    # TREND
    #
    # From May onward, require both trend points.
    #
    # April is first available month, so one trend point can
    # legitimately explain that history is insufficient.
    # ========================================================

    if selected_month_order > 1:

        if not cleaned[
            "trend_point_1"
        ]:

            raise ValueError(
                "Trend point 1 is empty."
            )


        if not cleaned[
            "trend_point_2"
        ]:

            raise ValueError(
                "Trend point 2 is empty."
            )


    else:

        if not cleaned[
            "trend_point_1"
        ]:

            cleaned[
                "trend_point_1"
            ] = (
                "This is the first available month in the "
                "current fiscal-year history, so a monthly "
                "trend cannot yet be established."
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
                fact_package=fact_package,
                retry_note=last_error,
            )


            response = (
                client.models.generate_content(

                    model=GEMINI_MODEL,

                    contents=prompt,

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


            last_raw_text = clean_string(
                getattr(
                    response,
                    "text",
                    ""
                )
            )


            result = parse_response(
                response
            )


            result = validate_result(
                result=result,
                selected_month_order=
                    selected_month_order,
            )


            return (
                result,
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
                f"      Attempt {attempt} failed: "
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
# BUILD FULL SUMMARY
#
# Kept for easy validation and optional SAC use.
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
# OUTPUT KEY
# ============================================================

KEY_COLUMNS = [

    "Fiscal_Year",
    "Month",
    "Entity_CD",
    "Currency",
]


def make_key(
    fiscal_year,
    month,
    entity_cd,
    currency,
):

    return (
        int(fiscal_year),
        clean_string(month),
        clean_string(entity_cd),
        clean_string(currency),
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
# REMOVE EXISTING KEY BEFORE REPLACEMENT
# ============================================================

def remove_existing_key(
    df,
    key,
):

    if df.empty:

        return df


    fiscal_year, month, entity_cd, currency = key


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
# SAVE FAILED RAW RESPONSE
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


    fiscal_year, month, entity_cd, currency = key


    safe_entity = (
        entity_cd
        .replace("/", "_")
        .replace("\\", "_")
        .replace(".", "_")
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


    content = (
        f"ERROR\n"
        f"=====\n"
        f"{error}\n\n"
        f"RAW GEMINI RESPONSE\n"
        f"===================\n"
        f"{raw_response}"
    )


    path.write_text(
        content,
        encoding="utf-8",
    )


# ============================================================
# BUILD FINAL OUTPUT ROW
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

        # ==============================================
        # EXECUTIVE
        # ==============================================

        "Executive_Point_1":
            result[
                "executive_point_1"
            ],

        "Executive_Point_2":
            result[
                "executive_point_2"
            ],

        # ==============================================
        # KEY MOVEMENTS
        # ==============================================

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

        # ==============================================
        # TREND
        # ==============================================

        "Trend_Point_1":
            result[
                "trend_point_1"
            ],

        "Trend_Point_2":
            result[
                "trend_point_2"
            ],

        # ==============================================
        # BUDGET / YTD
        # ==============================================

        "Budget_YTD_Point_1":
            result[
                "budget_ytd_point_1"
            ],

        "Budget_YTD_Point_2":
            result[
                "budget_ytd_point_2"
            ],

        # ==============================================
        # FULL SUMMARY
        # ==============================================

        "Summary_Full":
            full_summary,

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
        "CASHFLOW AI - ITERATION 2"
    )

    print(
        "RICH STRUCTURED COMMENTARY GENERATION"
    )

    print(
        "=" * 90
    )


    print(
        f"\nModel: {GEMINI_MODEL}"
    )

    print(
        "Thinking: MEDIUM"
    )

    print(
        "Output mode: Schema-enforced JSON"
    )


    # ========================================================
    # INPUTS
    # ========================================================

    (
        insight_df,
        trend_df,
    ) = load_inputs()


    # ========================================================
    # AVAILABLE SUMMARY KEYS
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


    if currency_filter:

        summary_keys = (
            summary_keys[
                summary_keys[
                    "Currency"
                ].str.upper()
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


    # ========================================================
    # CLIENT
    # ========================================================

    client = create_client()


    # ========================================================
    # COUNTERS
    # ========================================================

    generated = 0
    skipped = 0
    failures = 0


    total = len(
        summary_keys
    )


    # ========================================================
    # GENERATE
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
        # SKIP EXISTING
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
        # CURRENT INSIGHTS
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
        # TREND
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
                "    FAILED - no structured insight facts"
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
        # GEMINI
        # ====================================================

        (
            result,
            attempts,
            raw_response,
            error,
        ) = generate_one_summary(
            client=client,
            fact_package=fact_package,
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
                key=key,
                raw_response=
                    raw_response,
                error=error,
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

        print(
            f"    PASS "
            f"(attempt {attempts})"
        )


        output_row = (
            build_output_row(
                first_row=
                    insight_subset.iloc[0],
                result=result,
                attempts=attempts,
            )
        )


        # Remove prior version if --force.

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

                "Generated_At":
                    generated_at,
            }
        )


    # ========================================================
    # FINAL STATUS
    # ========================================================

    print(
        "\n"
        +
        "=" * 90
    )

    print(
        "ITERATION-2 GENERATION COMPLETE"
    )

    print(
        "=" * 90
    )


    print(
        f"\nGenerated: {generated}"
    )

    print(
        f"Skipped:   {skipped}"
    )

    print(
        f"Failures:  {failures}"
    )


    print(
        "\nOutput:"
    )

    print(
        OUTPUT_FILE.resolve()
    )


    print(
        "\nLog:"
    )

    print(
        LOG_FILE.resolve()
    )


    if failures > 0:

        print(
            "\nRaw failed responses:"
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
            "Generate Iteration-2 rich "
            "Cashflow AI commentary."
        )
    )


    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Generate only the first N "
            "summary combinations."
        ),
    )


    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Regenerate combinations that "
            "already exist."
        ),
    )


    parser.add_argument(
        "--currency",
        type=str,
        default=None,
        choices=[
            "EUR",
            "LC",
        ],
        help=(
            "Optional currency filter."
        ),
    )


    parser.add_argument(
        "--entity",
        type=str,
        default=None,
        help=(
            "Optional entity-name search."
        ),
    )


    args = parser.parse_args()


    run_generation(
        limit=args.limit,
        force=args.force,
        currency_filter=
            args.currency,
        entity_filter=
            args.entity,
    )


if __name__ == "__main__":

    main()
