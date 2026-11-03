from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# FILES
# ============================================================

INPUT_FILE = Path(
    "output/page1_enriched_analysis.csv"
)

INSIGHT_OUTPUT_FILE = Path(
    "output/page1_insight_points.csv"
)

TREND_OUTPUT_FILE = Path(
    "output/page1_trend_points.csv"
)


# ============================================================
# ITERATION-2 SETTINGS
# ============================================================

TOP_DRIVER_COUNT = 4

# Avoid ridiculous percentage commentary when previous value
# was extremely close to zero.
MAX_DISPLAY_MOM_PCT = 500

# ============================================================
# HEADLINE PAGE-1 KPIs
#
# These are always supplied to Gemini.
#
# We use account code where known because it is safer than
# relying entirely on descriptions.
# ============================================================

HEADLINE_CONFIG = [

    {
        "metric": "EBITDA",
        "account_codes": ["C8211000"],
        "account_names": ["EBITDA"],
        "rank": 1,
    },

    {
        "metric": "Working Capital",
        "account_codes": ["C8220000"],
        "account_names": ["Working Capital"],
        "rank": 2,
    },

    {
        "metric": "Capital Additions",
        "account_codes": ["C8260000"],
        "account_names": ["Capital Additions"],
        "rank": 3,
    },

    {
        "metric": "Operating Free Cash Flow",
        "account_codes": ["C8200000"],
        "account_names": ["Operating Free Cash Flow"],
        "rank": 4,
    },

    {
        "metric": "Free cash flow pre spectrum and restructuring",
        "account_codes": ["C8100000"],
        "account_names": [
            "Free cash flow pre spectrum and restructuring"
        ],
        "rank": 5,
    },

    {
        "metric": "Free Cash Flow",
        "account_codes": [],
        "account_names": ["Free Cash Flow"],
        "rank": 6,
    },
]


# ============================================================
# PAGE-1 DRIVER ACCOUNTS
#
# These can compete for the Top 4 Key Movements.
#
# Headline KPIs are intentionally excluded because they already
# have their own section.
# ============================================================

DRIVER_ACCOUNT_NAMES = {

    "Licences and Spectrum",

    "Movement in Capital Creditor",

    "Net Dividends",

    "Net Interest",

    "Net Taxation Paid",

    "Restructuring",

    "WC - Inventory",

    "WC - Payables",

    "WC - Receivables",
}


# ============================================================
# REQUIRED COLUMNS
# ============================================================

REQUIRED_COLUMNS = [

    "Fiscal_Year",

    "Month",

    "Month_Order",

    "Entity_CD",

    "Entity_Name",

    "Currency",

    "Account_CD",

    "Account_Name",

    "Periodic_Actual",

    "Periodic_Budget",

    "Periodic_Vs_Budget",

    "Periodic_YoY_Abs",

    "YTD_Actual",

    "YTD_Budget",

    "YTD_Vs_Budget",

    "YTD_YoY_Abs",

    "Previous_Month",

    "Previous_Periodic_Actual",

    "MoM_Change",

    "MoM_Pct",

    "Two_Months_Ago_Actual",

    "Two_Month_Change",

    "Three_Months_Ago_Actual",

    "Three_Month_Change",

    "Materiality_Score",
]


# ============================================================
# UTILITY
# ============================================================

def safe_number(value):

    if pd.isna(value):
        return np.nan

    try:
        return float(value)

    except Exception:
        return np.nan


def safe_pct_change(
    current,
    previous,
):

    current = safe_number(
        current
    )

    previous = safe_number(
        previous
    )

    if pd.isna(current):
        return np.nan

    if pd.isna(previous):
        return np.nan

    if previous == 0:
        return np.nan

    return (
        (
            current
            -
            previous
        )
        /
        abs(previous)
    ) * 100


def clean_text(value):

    if pd.isna(value):
        return ""

    return str(
        value
    ).strip()


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Input file not found: "
            f"{INPUT_FILE.resolve()}"
        )


    df = pd.read_csv(
        INPUT_FILE,
        dtype={
            "Entity_CD": str,
            "Account_CD": str,
        },
    )


    missing_columns = [

        column

        for column in REQUIRED_COLUMNS

        if column not in df.columns
    ]


    if missing_columns:

        raise ValueError(
            "Missing required columns:\n"
            +
            "\n".join(
                missing_columns
            )
        )


    # ========================================================
    # CLEAN STRINGS
    # ========================================================

    string_columns = [

        "Month",

        "Entity_CD",

        "Entity_Name",

        "Currency",

        "Account_CD",

        "Account_Name",
    ]


    for column in string_columns:

        df[column] = (
            df[column]
            .fillna("")
            .astype(str)
            .str.strip()
        )


    # ========================================================
    # NUMERIC COLUMNS
    # ========================================================

    numeric_columns = [

        "Fiscal_Year",

        "Month_Order",

        "Periodic_Actual",

        "Periodic_Budget",

        "Periodic_Vs_Budget",

        "Periodic_YoY_Abs",

        "YTD_Actual",

        "YTD_Budget",

        "YTD_Vs_Budget",

        "YTD_YoY_Abs",

        "Previous_Periodic_Actual",

        "MoM_Change",

        "MoM_Pct",

        "Two_Months_Ago_Actual",

        "Two_Month_Change",

        "Three_Months_Ago_Actual",

        "Three_Month_Change",

        "Materiality_Score",
    ]


    for column in numeric_columns:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )


    return df


# ============================================================
# IDENTIFY HEADLINE ACCOUNT
# ============================================================

def find_headline_config(
    account_cd,
    account_name,
):

    account_cd = clean_text(
        account_cd
    )

    account_name = clean_text(
        account_name
    ).lower()


    for config in HEADLINE_CONFIG:

        if (
            account_cd
            and
            account_cd
            in config[
                "account_codes"
            ]
        ):

            return config


        valid_names = [

            name.lower()

            for name in config[
                "account_names"
            ]
        ]


        if account_name in valid_names:

            return config


    return None


# ============================================================
# NUMERICAL POSITION LABELS
#
# IMPORTANT:
# These are NOT favourable / unfavourable labels.
#
# Example:
# Actual > Budget = ABOVE_BUDGET
#
# We do not tell Gemini that this is automatically "good".
# ============================================================

def get_budget_position(
    actual,
    budget,
):

    actual = safe_number(
        actual
    )

    budget = safe_number(
        budget
    )

    if (
        pd.isna(actual)
        or
        pd.isna(budget)
    ):

        return "NO_BUDGET_COMPARISON"


    difference = (
        actual
        -
        budget
    )


    if difference > 0:

        return "ABOVE_BUDGET"

    if difference < 0:

        return "BELOW_BUDGET"

    return "ON_BUDGET"


def get_movement_direction(
    change,
):

    change = safe_number(
        change
    )

    if pd.isna(change):

        return "NO_PREVIOUS_MONTH"

    if change > 0:

        return "INCREASE"

    if change < 0:

        return "DECREASE"

    return "NO_CHANGE"


# ============================================================
# SAFE MOM %
#
# If percentage is extreme because previous value was almost
# zero, leave it blank for Gemini.
#
# Gemini can then use absolute movement instead.
# ============================================================

def get_safe_mom_pct(
    value,
):

    value = safe_number(
        value
    )

    if pd.isna(value):
        return np.nan

    if abs(value) > MAX_DISPLAY_MOM_PCT:
        return np.nan

    return value


# ============================================================
# TREND CLASSIFICATION
#
# Pure mathematical pattern.
# No business judgement.
# ============================================================

def classify_trend(
    history_df,
):

    history_df = (
        history_df
        .sort_values(
            "Month_Order"
        )
    )


    values = (
        history_df[
            "Periodic_Actual"
        ]
        .dropna()
        .tolist()
    )


    if len(values) < 2:

        return "INSUFFICIENT_HISTORY"


    differences = np.diff(
        values
    )


    if len(differences) == 0:

        return "INSUFFICIENT_HISTORY"


    if np.all(
        differences > 0
    ):

        return "CONSISTENT_INCREASE"


    if np.all(
        differences < 0
    ):

        return "CONSISTENT_DECREASE"


    if np.all(
        differences == 0
    ):

        return "FLAT"


    if len(differences) >= 2:

        previous_direction = (
            differences[-2]
        )

        latest_direction = (
            differences[-1]
        )


        if (
            previous_direction < 0
            and
            latest_direction > 0
        ):

            return "REVERSAL_UP"


        if (
            previous_direction > 0
            and
            latest_direction < 0
        ):

            return "REVERSAL_DOWN"


    return "MIXED"


# ============================================================
# CONSECUTIVE MOVEMENT
#
# Example:
#
# Apr 10
# May 20
# Jun 30
#
# = 2 consecutive increases
# ============================================================

def get_consecutive_movement(
    history_df,
):

    history_df = (
        history_df
        .sort_values(
            "Month_Order"
        )
    )


    values = (
        history_df[
            "Periodic_Actual"
        ]
        .dropna()
        .tolist()
    )


    if len(values) < 2:

        return (
            "NONE",
            0,
        )


    differences = np.diff(
        values
    )


    latest = differences[-1]


    if latest > 0:

        direction = "INCREASE"

    elif latest < 0:

        direction = "DECREASE"

    else:

        return (
            "NO_CHANGE",
            1,
        )


    count = 0


    for difference in reversed(
        differences
    ):

        if (
            direction == "INCREASE"
            and
            difference > 0
        ):

            count += 1

        elif (
            direction == "DECREASE"
            and
            difference < 0
        ):

            count += 1

        else:

            break


    return (
        direction,
        count,
    )


# ============================================================
# HISTORY STATISTICS
#
# Gives Gemini pre-calculated context such as:
#
# - first available month
# - change since first month
# - peak month
# - lowest month
# - recent trend shape
#
# This helps it produce much more insightful wording.
# ============================================================

def get_history_context(
    history_df,
):

    history_df = (
        history_df
        .sort_values(
            "Month_Order"
        )
        .copy()
    )


    valid = (
        history_df[
            history_df[
                "Periodic_Actual"
            ].notna()
        ]
        .copy()
    )


    if valid.empty:

        return {
            "History_Months": 0,
            "First_Month": "",
            "First_Actual": np.nan,
            "Change_From_First": np.nan,
            "Change_From_First_Pct": np.nan,
            "Peak_Month": "",
            "Peak_Actual": np.nan,
            "Lowest_Month": "",
            "Lowest_Actual": np.nan,
            "Recent_Trend": "NO_HISTORY",
            "Consecutive_Direction": "NONE",
            "Consecutive_Count": 0,
        }


    first_row = valid.iloc[0]

    latest_row = valid.iloc[-1]


    peak_index = (
        valid[
            "Periodic_Actual"
        ]
        .idxmax()
    )


    low_index = (
        valid[
            "Periodic_Actual"
        ]
        .idxmin()
    )


    peak_row = valid.loc[
        peak_index
    ]


    low_row = valid.loc[
        low_index
    ]


    first_actual = (
        first_row[
            "Periodic_Actual"
        ]
    )


    latest_actual = (
        latest_row[
            "Periodic_Actual"
        ]
    )


    change_from_first = (
        latest_actual
        -
        first_actual
    )


    pct_from_first = (
        safe_pct_change(
            latest_actual,
            first_actual,
        )
    )


    consecutive_direction, consecutive_count = (
        get_consecutive_movement(
            valid
        )
    )


    return {

        "History_Months":
            len(valid),

        "First_Month":
            first_row[
                "Month"
            ],

        "First_Actual":
            first_actual,

        "Change_From_First":
            change_from_first,

        "Change_From_First_Pct":
            pct_from_first,

        "Peak_Month":
            peak_row[
                "Month"
            ],

        "Peak_Actual":
            peak_row[
                "Periodic_Actual"
            ],

        "Lowest_Month":
            low_row[
                "Month"
            ],

        "Lowest_Actual":
            low_row[
                "Periodic_Actual"
            ],

        "Recent_Trend":
            classify_trend(
                valid
            ),

        "Consecutive_Direction":
            consecutive_direction,

        "Consecutive_Count":
            consecutive_count,
    }


# ============================================================
# BUILD SINGLE INSIGHT ROW
# ============================================================

def build_insight_row(
    row,
    section,
    rank,
    metric,
    materiality_rank,
    movement_rank,
    budget_rank,
    history_context,
):

    mom_pct_safe = (
        get_safe_mom_pct(
            row[
                "MoM_Pct"
            ]
        )
    )


    return {

        "Fiscal_Year":
            row[
                "Fiscal_Year"
            ],

        "Selected_Month":
            row[
                "Month"
            ],

        "Selected_Month_Order":
            row[
                "Month_Order"
            ],

        "Entity_CD":
            row[
                "Entity_CD"
            ],

        "Entity_Name":
            row[
                "Entity_Name"
            ],

        # ==============================================
        # ITERATION-2 CURRENCY
        # ==============================================

        "Currency":
            row[
                "Currency"
            ],


        # ==============================================
        # INSIGHT TYPE
        # ==============================================

        "Section":
            section,

        "Rank":
            rank,

        "Metric":
            metric,

        "Account_CD":
            row[
                "Account_CD"
            ],

        "Account_Name":
            row[
                "Account_Name"
            ],


        # ==============================================
        # CURRENT PERIOD
        # ==============================================

        "Actual":
            row[
                "Periodic_Actual"
            ],

        "Budget":
            row[
                "Periodic_Budget"
            ],

        "Vs_Budget":
            row[
                "Periodic_Vs_Budget"
            ],

        "Budget_Position":
            get_budget_position(
                row[
                    "Periodic_Actual"
                ],
                row[
                    "Periodic_Budget"
                ],
            ),

        "YoY_Abs":
            row[
                "Periodic_YoY_Abs"
            ],


        # ==============================================
        # MONTH-ON-MONTH
        # ==============================================

        "Previous_Month":
            row[
                "Previous_Month"
            ],

        "Previous_Actual":
            row[
                "Previous_Periodic_Actual"
            ],

        "MoM_Change":
            row[
                "MoM_Change"
            ],

        "MoM_Pct":
            row[
                "MoM_Pct"
            ],

        "MoM_Pct_Safe":
            mom_pct_safe,

        "MoM_Direction":
            get_movement_direction(
                row[
                    "MoM_Change"
                ]
            ),


        # ==============================================
        # MULTI-MONTH CONTEXT
        # ==============================================

        "Two_Months_Ago_Actual":
            row[
                "Two_Months_Ago_Actual"
            ],

        "Two_Month_Change":
            row[
                "Two_Month_Change"
            ],

        "Three_Months_Ago_Actual":
            row[
                "Three_Months_Ago_Actual"
            ],

        "Three_Month_Change":
            row[
                "Three_Month_Change"
            ],


        # ==============================================
        # YTD
        # ==============================================

        "YTD_Actual":
            row[
                "YTD_Actual"
            ],

        "YTD_Budget":
            row[
                "YTD_Budget"
            ],

        "YTD_Vs_Budget":
            row[
                "YTD_Vs_Budget"
            ],

        "YTD_Budget_Position":
            get_budget_position(
                row[
                    "YTD_Actual"
                ],
                row[
                    "YTD_Budget"
                ],
            ),

        "YTD_YoY_Abs":
            row[
                "YTD_YoY_Abs"
            ],


        # ==============================================
        # IMPORTANCE
        # ==============================================

        "Materiality_Score":
            row[
                "Materiality_Score"
            ],

        "Materiality_Rank":
            materiality_rank,

        "Movement_Rank":
            movement_rank,

        "Budget_Variance_Rank":
            budget_rank,


        # ==============================================
        # HISTORY / TREND CONTEXT
        # ==============================================

        **history_context,
    }


# ============================================================
# GET HISTORY FOR ONE ACCOUNT
#
# Only:
# same FY
# same entity
# same currency
# same account
# up to selected month
#
# EUR and LC can NEVER mix here.
# ============================================================

def get_account_history(
    full_df,
    fiscal_year,
    entity_cd,
    currency,
    account_cd,
    selected_month_order,
):

    return (
        full_df[
            (
                full_df[
                    "Fiscal_Year"
                ]
                ==
                fiscal_year
            )
            &
            (
                full_df[
                    "Entity_CD"
                ]
                ==
                entity_cd
            )
            &
            (
                full_df[
                    "Currency"
                ]
                ==
                currency
            )
            &
            (
                full_df[
                    "Account_CD"
                ]
                ==
                account_cd
            )
            &
            (
                full_df[
                    "Month_Order"
                ]
                <=
                selected_month_order
            )
        ]
        .sort_values(
            "Month_Order"
        )
        .copy()
    )


# ============================================================
# FIND HEADLINE ROWS
# ============================================================

def get_headline_rows(
    current_df,
):

    output = []


    for config in HEADLINE_CONFIG:

        matches = pd.DataFrame()


        # ==============================================
        # FIRST TRY ACCOUNT CODE
        # ==============================================

        if config[
            "account_codes"
        ]:

            matches = (
                current_df[
                    current_df[
                        "Account_CD"
                    ]
                    .isin(
                        config[
                            "account_codes"
                        ]
                    )
                ]
            )


        # ==============================================
        # FALL BACK TO ACCOUNT NAME
        # ==============================================

        if matches.empty:

            target_names = {

                name.lower()

                for name in config[
                    "account_names"
                ]
            }


            matches = (
                current_df[
                    current_df[
                        "Account_Name"
                    ]
                    .str.lower()
                    .isin(
                        target_names
                    )
                ]
            )


        if matches.empty:

            continue


        row = matches.iloc[0]


        output.append(
            (
                config,
                row,
            )
        )


    return output


# ============================================================
# GET TOP 4 DRIVERS
# ============================================================

def get_driver_rows(
    current_df,
):

    driver_names_lower = {

        name.lower()

        for name in DRIVER_ACCOUNT_NAMES
    }


    driver_df = (
        current_df[
            current_df[
                "Account_Name"
            ]
            .str.lower()
            .isin(
                driver_names_lower
            )
        ]
        .copy()
    )


    if driver_df.empty:

        return driver_df


    # ========================================================
    # Sort by overall materiality.
    #
    # This captures major:
    #
    # MoM movement
    # Budget variance
    # YoY movement
    # ========================================================

    driver_df = (
        driver_df
        .sort_values(
            [
                "Materiality_Score",
                "Abs_MoM_Change",
            ]
            if "Abs_MoM_Change"
            in driver_df.columns
            else [
                "Materiality_Score",
            ],
            ascending=False,
        )
        .head(
            TOP_DRIVER_COUNT
        )
        .copy()
    )


    return driver_df


# ============================================================
# CALCULATE CURRENT-MONTH RANKINGS
#
# Gives Gemini context such as:
#
# "largest movement"
# "second-largest budget deviation"
#
# without asking Gemini to calculate ranks itself.
# ============================================================

def add_current_rankings(
    current_df,
):

    current_df = (
        current_df
        .copy()
    )


    current_df[
        "_abs_mom"
    ] = (
        current_df[
            "MoM_Change"
        ]
        .abs()
    )


    current_df[
        "_abs_budget"
    ] = (
        current_df[
            "Periodic_Vs_Budget"
        ]
        .abs()
    )


    current_df[
        "Movement_Rank_Current"
    ] = (
        current_df[
            "_abs_mom"
        ]
        .rank(
            method="min",
            ascending=False,
            na_option="bottom",
        )
    )


    current_df[
        "Budget_Rank_Current"
    ] = (
        current_df[
            "_abs_budget"
        ]
        .rank(
            method="min",
            ascending=False,
            na_option="bottom",
        )
    )


    current_df[
        "Materiality_Rank_Current"
    ] = (
        current_df[
            "Materiality_Score"
        ]
        .rank(
            method="min",
            ascending=False,
            na_option="bottom",
        )
    )


    return current_df


# ============================================================
# BUILD INSIGHT POINTS
# ============================================================

def build_insight_points(
    df,
):

    output_rows = []


    summary_keys = (
        df[
            [
                "Fiscal_Year",

                "Month",

                "Month_Order",

                "Entity_CD",

                "Entity_Name",

                "Currency",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                "Entity_Name",
                "Currency",
                "Month_Order",
            ]
        )
    )


    total_groups = len(
        summary_keys
    )


    print(
        f"\nSummary combinations: "
        f"{total_groups}"
    )


    for counter, key in enumerate(
        summary_keys.itertuples(
            index=False
        ),
        start=1,
    ):

        fiscal_year = (
            key.Fiscal_Year
        )

        selected_month = (
            key.Month
        )

        selected_month_order = (
            key.Month_Order
        )

        entity_cd = (
            key.Entity_CD
        )

        entity_name = (
            key.Entity_Name
        )

        currency = (
            key.Currency
        )


        current_df = (
            df[
                (
                    df[
                        "Fiscal_Year"
                    ]
                    ==
                    fiscal_year
                )
                &
                (
                    df[
                        "Month"
                    ]
                    ==
                    selected_month
                )
                &
                (
                    df[
                        "Entity_CD"
                    ]
                    ==
                    entity_cd
                )
                &
                (
                    df[
                        "Currency"
                    ]
                    ==
                    currency
                )
            ]
            .copy()
        )


        if current_df.empty:
            continue


        current_df = (
            add_current_rankings(
                current_df
            )
        )


        # ====================================================
        # HEADLINE KPIs
        # ====================================================

        headline_rows = (
            get_headline_rows(
                current_df
            )
        )


        for (
            config,
            row,
        ) in headline_rows:

            history_df = (
                get_account_history(
                    full_df=df,
                    fiscal_year=fiscal_year,
                    entity_cd=entity_cd,
                    currency=currency,
                    account_cd=row[
                        "Account_CD"
                    ],
                    selected_month_order=selected_month_order,
                )
            )


            history_context = (
                get_history_context(
                    history_df
                )
            )


            output_rows.append(

                build_insight_row(

                    row=row,

                    section="HEADLINE",

                    rank=config[
                        "rank"
                    ],

                    metric=config[
                        "metric"
                    ],

                    materiality_rank=row[
                        "Materiality_Rank_Current"
                    ],

                    movement_rank=row[
                        "Movement_Rank_Current"
                    ],

                    budget_rank=row[
                        "Budget_Rank_Current"
                    ],

                    history_context=
                        history_context,
                )
            )


        # ====================================================
        # TOP 4 DRIVERS
        # ====================================================

        drivers = (
            get_driver_rows(
                current_df
            )
        )


        for driver_rank, (
            _,
            row,
        ) in enumerate(
            drivers.iterrows(),
            start=1,
        ):

            history_df = (
                get_account_history(
                    full_df=df,
                    fiscal_year=fiscal_year,
                    entity_cd=entity_cd,
                    currency=currency,
                    account_cd=row[
                        "Account_CD"
                    ],
                    selected_month_order=selected_month_order,
                )
            )


            history_context = (
                get_history_context(
                    history_df
                )
            )


            output_rows.append(

                build_insight_row(

                    row=row,

                    section="DRIVER",

                    rank=driver_rank,

                    metric=row[
                        "Account_Name"
                    ],

                    materiality_rank=row[
                        "Materiality_Rank_Current"
                    ],

                    movement_rank=row[
                        "Movement_Rank_Current"
                    ],

                    budget_rank=row[
                        "Budget_Rank_Current"
                    ],

                    history_context=
                        history_context,
                )
            )


        if (
            counter % 50 == 0
            or
            counter == total_groups
        ):

            print(
                f"Processed "
                f"{counter}/"
                f"{total_groups}"
            )


    insight_df = pd.DataFrame(
        output_rows
    )


    return insight_df


# ============================================================
# BUILD TREND POINTS
#
# IMPORTANT ITERATION-2 CHANGE:
#
# We now provide history not only for headline KPIs,
# but also for the selected Top 4 drivers.
#
# Gemini can therefore explain:
#
# "WC Receivables has deteriorated for three months"
#
# rather than merely:
#
# "WC Receivables changed by X this month."
# ============================================================

def build_trend_points(
    full_df,
    insight_df,
):

    trend_rows = []


    if insight_df.empty:

        return pd.DataFrame()


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
    )


    for key in summary_keys.itertuples(
        index=False
    ):

        selected_points = (
            insight_df[
                (
                    insight_df[
                        "Fiscal_Year"
                    ]
                    ==
                    key.Fiscal_Year
                )
                &
                (
                    insight_df[
                        "Selected_Month"
                    ]
                    ==
                    key.Selected_Month
                )
                &
                (
                    insight_df[
                        "Entity_CD"
                    ]
                    ==
                    key.Entity_CD
                )
                &
                (
                    insight_df[
                        "Currency"
                    ]
                    ==
                    key.Currency
                )
            ]
            .copy()
        )


        # Each account only once.
        selected_points = (
            selected_points
            .drop_duplicates(
                subset=[
                    "Account_CD"
                ]
            )
        )


        for point in selected_points.itertuples(
            index=False
        ):

            history_df = (
                full_df[
                    (
                        full_df[
                            "Fiscal_Year"
                        ]
                        ==
                        key.Fiscal_Year
                    )
                    &
                    (
                        full_df[
                            "Entity_CD"
                        ]
                        ==
                        key.Entity_CD
                    )
                    &
                    (
                        full_df[
                            "Currency"
                        ]
                        ==
                        key.Currency
                    )
                    &
                    (
                        full_df[
                            "Account_CD"
                        ]
                        ==
                        point.Account_CD
                    )
                    &
                    (
                        full_df[
                            "Month_Order"
                        ]
                        <=
                        key.Selected_Month_Order
                    )
                ]
                .sort_values(
                    "Month_Order"
                )
            )


            for history_row in (
                history_df.itertuples(
                    index=False
                )
            ):

                trend_rows.append(
                    {

                        "Fiscal_Year":
                            key.Fiscal_Year,

                        "Selected_Month":
                            key.Selected_Month,

                        "Selected_Month_Order":
                            key.Selected_Month_Order,

                        "Entity_CD":
                            key.Entity_CD,

                        "Entity_Name":
                            key.Entity_Name,

                        "Currency":
                            key.Currency,

                        "Metric_Type":
                            point.Section,

                        "Metric":
                            point.Metric,

                        "Account_CD":
                            point.Account_CD,

                        "Account_Name":
                            point.Account_Name,

                        "Trend_Month":
                            history_row.Month,

                        "Trend_Month_Order":
                            history_row.Month_Order,

                        "Actual":
                            history_row.Periodic_Actual,

                        "Budget":
                            history_row.Periodic_Budget,

                        "Vs_Budget":
                            history_row.Periodic_Vs_Budget,

                        "YoY_Abs":
                            history_row.Periodic_YoY_Abs,

                        "YTD_Actual":
                            history_row.YTD_Actual,

                        "YTD_Budget":
                            history_row.YTD_Budget,

                        "YTD_Vs_Budget":
                            history_row.YTD_Vs_Budget,

                        "Is_Selected_Month":
                            (
                                history_row.Month
                                ==
                                key.Selected_Month
                            ),
                    }
                )


    return pd.DataFrame(
        trend_rows
    )


# ============================================================
# VALIDATE OUTPUT
# ============================================================

def validate_outputs(
    insight_df,
    trend_df,
):

    print(
        "\n"
        +
        "=" * 80
    )

    print(
        "ITERATION-2 STRUCTURED INSIGHT VALIDATION"
    )

    print(
        "=" * 80
    )


    print(
        f"\nInsight rows: "
        f"{len(insight_df)}"
    )


    print(
        f"Trend rows: "
        f"{len(trend_df)}"
    )


    if not insight_df.empty:

        print(
            f"\nEntities: "
            f"{insight_df['Entity_CD'].nunique()}"
        )


        print(
            f"Months: "
            f"{insight_df['Selected_Month'].nunique()}"
        )


        print(
            f"Currencies: "
            f"{insight_df['Currency'].nunique()}"
        )


        combinations = (
            insight_df[
                [
                    "Fiscal_Year",
                    "Selected_Month",
                    "Entity_CD",
                    "Currency",
                ]
            ]
            .drop_duplicates()
        )


        print(
            f"Summary combinations: "
            f"{len(combinations)}"
        )


        print(
            "\nRows by Section:"
        )


        print(
            insight_df[
                "Section"
            ]
            .value_counts()
            .to_string()
        )


        print(
            "\nRows by Currency:"
        )


        print(
            insight_df[
                "Currency"
            ]
            .value_counts()
            .to_string()
        )


        duplicate_count = (
            insight_df
            .duplicated(
                subset=[
                    "Fiscal_Year",
                    "Selected_Month",
                    "Entity_CD",
                    "Currency",
                    "Section",
                    "Rank",
                ]
            )
            .sum()
        )


        print(
            f"\nDuplicate insight positions: "
            f"{duplicate_count}"
        )


        if duplicate_count == 0:

            print(
                "PASS: Insight grain is clean."
            )


# ============================================================
# PRINT SAMPLE
# ============================================================

def print_sample(
    insight_df,
):

    if insight_df.empty:
        return


    sample_key = (
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
        .iloc[0]
    )


    sample = (
        insight_df[
            (
                insight_df[
                    "Fiscal_Year"
                ]
                ==
                sample_key[
                    "Fiscal_Year"
                ]
            )
            &
            (
                insight_df[
                    "Selected_Month"
                ]
                ==
                sample_key[
                    "Selected_Month"
                ]
            )
            &
            (
                insight_df[
                    "Entity_CD"
                ]
                ==
                sample_key[
                    "Entity_CD"
                ]
            )
            &
            (
                insight_df[
                    "Currency"
                ]
                ==
                sample_key[
                    "Currency"
                ]
            )
        ]
        .sort_values(
            [
                "Section",
                "Rank",
            ]
        )
    )


    columns = [

        "Entity_Name",

        "Selected_Month",

        "Currency",

        "Section",

        "Rank",

        "Metric",

        "Actual",

        "Previous_Actual",

        "MoM_Change",

        "MoM_Pct_Safe",

        "Budget",

        "Vs_Budget",

        "YTD_Actual",

        "YTD_Vs_Budget",

        "Recent_Trend",

        "Consecutive_Direction",

        "Consecutive_Count",

        "Materiality_Rank",
    ]


    print(
        "\nSAMPLE STRUCTURED INSIGHT PACKAGE"
    )

    print(
        "=" * 140
    )


    print(
        sample[
            columns
        ]
        .to_string(
            index=False
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\nITERATION 2"
    )

    print(
        "BUILDING RICH PAGE-1 STRUCTURED INSIGHTS"
    )

    print(
        "=" * 80
    )


    # ========================================================
    # LOAD ENRICHED DATA
    # ========================================================

    df = load_data()


    print(
        f"\nLoaded enriched rows: "
        f"{len(df)}"
    )


    print(
        f"Entities: "
        f"{df['Entity_CD'].nunique()}"
    )


    print(
        f"Months: "
        f"{df['Month'].nunique()}"
    )


    print(
        f"Currencies: "
        f"{df['Currency'].nunique()}"
    )


    # ========================================================
    # INSIGHT POINTS
    # ========================================================

    print(
        "\nBuilding richer current-period insight points..."
    )


    insight_df = (
        build_insight_points(
            df
        )
    )


    # ========================================================
    # TREND POINTS
    # ========================================================

    print(
        "\nBuilding richer historical trend points..."
    )


    trend_df = (
        build_trend_points(
            full_df=df,
            insight_df=insight_df,
        )
    )


    # ========================================================
    # VALIDATE
    # ========================================================

    validate_outputs(
        insight_df,
        trend_df,
    )


    print_sample(
        insight_df
    )


    # ========================================================
    # SAVE
    # ========================================================

    INSIGHT_OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    insight_df.to_csv(
        INSIGHT_OUTPUT_FILE,
        index=False,
    )


    trend_df.to_csv(
        TREND_OUTPUT_FILE,
        index=False,
    )


    print(
        "\n"
        +
        "=" * 80
    )

    print(
        "OUTPUTS SAVED"
    )

    print(
        "=" * 80
    )


    print(
        INSIGHT_OUTPUT_FILE.resolve()
    )


    print(
        TREND_OUTPUT_FILE.resolve()
    )


    print(
        "\nITERATION-2 STRUCTURED INSIGHTS COMPLETED."
    )


if __name__ == "__main__":

    main()
