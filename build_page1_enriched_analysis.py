from pathlib import Path

import numpy as np
import pandas as pd

from google.cloud import bigquery

from config import (
    PROJECT_ID,
    DATASET_ID,
)


# ============================================================
# FILES
# ============================================================

MAPPING_FILE = Path(
    "output/page1_final_account_mapping.csv"
)

OUTPUT_FILE = Path(
    "output/page1_enriched_analysis.csv"
)


# ============================================================
# FISCAL MONTH ORDER
# ============================================================

MONTH_ORDER_SQL = """
CASE Month
    WHEN 'Apr' THEN 1
    WHEN 'May' THEN 2
    WHEN 'Jun' THEN 3
    WHEN 'Jul' THEN 4
    WHEN 'Aug' THEN 5
    WHEN 'Sep' THEN 6
    WHEN 'Oct' THEN 7
    WHEN 'Nov' THEN 8
    WHEN 'Dec' THEN 9
    WHEN 'Jan' THEN 10
    WHEN 'Feb' THEN 11
    WHEN 'Mar' THEN 12
    ELSE 99
END
"""


# ============================================================
# LATEST FISCAL YEAR
# ============================================================

def get_latest_year(client):

    query = f"""
    SELECT
        MAX(
            SAFE_CAST(
                Year AS INT64
            )
        ) AS latest_year

    FROM `{PROJECT_ID}.{DATASET_ID}.CF`
    """

    df = client.query(
        query
    ).to_dataframe()


    if df.empty:

        return None


    value = df.iloc[0][
        "latest_year"
    ]


    if pd.isna(value):

        return None


    return int(
        value
    )


# ============================================================
# LOAD PAGE-1 ACCOUNT MAPPING
# ============================================================

def load_mapping():

    if not MAPPING_FILE.exists():

        raise FileNotFoundError(
            f"Mapping file not found: "
            f"{MAPPING_FILE.resolve()}"
        )


    mapping_df = pd.read_csv(
        MAPPING_FILE,
        dtype={
            "Account_CD": str,
        },
    )


    mapping_df[
        "Account_CD"
    ] = (

        mapping_df[
            "Account_CD"
        ]

        .astype(str)

        .str.strip()
    )


    return mapping_df


# ============================================================
# EXTRACT PAGE-1 DATA
# ============================================================

def extract_page1_data(
    client,
    year,
    account_codes,
):

    query = f"""
    WITH loaded_months AS (

        SELECT
            Month

        FROM `{PROJECT_ID}.{DATASET_ID}.CF`

        WHERE SAFE_CAST(
            Year AS INT64
        ) = @year

          AND Scenario_CD =
              'ACTUAL_CONS'

        GROUP BY
            Month

        HAVING COUNTIF(
            Value != 0
        ) > 0
    ),

    base AS (

        SELECT

            SAFE_CAST(
                Year AS INT64
            ) AS Fiscal_Year,

            Month,

            {MONTH_ORDER_SQL}
                AS Month_Order,

            Entity_CD,

            Entity_Name,

            Account_CD,

            Account_Name,

            Scenario_CD,

            View_CD,

            Currency,

            Currency_CD,

            -- ==============================================
            -- CURRENCY KEY
            -- ==============================================
            --
            -- This gives us one stable grouping key.
            --
            -- Prefer Currency_CD.
            -- If Currency_CD is blank/null,
            -- fall back to Currency.
            -- ==============================================

            COALESCE(
                NULLIF(
                    TRIM(
                        CAST(
                            Currency_CD
                            AS STRING
                        )
                    ),
                    ''
                ),
                NULLIF(
                    TRIM(
                        CAST(
                            Currency
                            AS STRING
                        )
                    ),
                    ''
                ),
                'UNKNOWN'
            ) AS Currency_Key,

            Value

        FROM `{PROJECT_ID}.{DATASET_ID}.CF`

        WHERE SAFE_CAST(
            Year AS INT64
        ) = @year

          AND Month IN (

              SELECT
                  Month

              FROM loaded_months
          )

          AND Account_CD
              IN UNNEST(
                  @account_codes
              )

          AND Scenario_CD IN (

              'ACTUAL_CONS',

              'BUDGET_CONS',

              'ACTUAL_RESTATE_V_BUDGET_CONS',

              'YoYAbs_ACTUAL_CONS_V_ACTUAL_CONS'
          )

          AND View_CD IN (

              'Periodic',

              'Year-To-Date'
          )
    )


    SELECT

        Fiscal_Year,

        Month,

        Month_Order,

        Entity_CD,

        Entity_Name,

        Currency,

        Currency_CD,

        Currency_Key,

        Account_CD,

        Account_Name,


        -- ==============================================
        -- PERIODIC
        -- ==============================================

        MAX(

            CASE

                WHEN View_CD =
                     'Periodic'

                 AND Scenario_CD =
                     'ACTUAL_CONS'

                THEN Value

            END

        ) AS Periodic_Actual,


        MAX(

            CASE

                WHEN View_CD =
                     'Periodic'

                 AND Scenario_CD =
                     'BUDGET_CONS'

                THEN Value

            END

        ) AS Periodic_Budget,


        MAX(

            CASE

                WHEN View_CD =
                     'Periodic'

                 AND Scenario_CD =
                     'ACTUAL_RESTATE_V_BUDGET_CONS'

                THEN Value

            END

        ) AS Periodic_Vs_Budget,


        MAX(

            CASE

                WHEN View_CD =
                     'Periodic'

                 AND Scenario_CD =
                     'YoYAbs_ACTUAL_CONS_V_ACTUAL_CONS'

                THEN Value

            END

        ) AS Periodic_YoY_Abs,


        -- ==============================================
        -- YTD
        -- ==============================================

        MAX(

            CASE

                WHEN View_CD =
                     'Year-To-Date'

                 AND Scenario_CD =
                     'ACTUAL_CONS'

                THEN Value

            END

        ) AS YTD_Actual,


        MAX(

            CASE

                WHEN View_CD =
                     'Year-To-Date'

                 AND Scenario_CD =
                     'BUDGET_CONS'

                THEN Value

            END

        ) AS YTD_Budget,


        MAX(

            CASE

                WHEN View_CD =
                     'Year-To-Date'

                 AND Scenario_CD =
                     'ACTUAL_RESTATE_V_BUDGET_CONS'

                THEN Value

            END

        ) AS YTD_Vs_Budget,


        MAX(

            CASE

                WHEN View_CD =
                     'Year-To-Date'

                 AND Scenario_CD =
                     'YoYAbs_ACTUAL_CONS_V_ACTUAL_CONS'

                THEN Value

            END

        ) AS YTD_YoY_Abs


    FROM base


    GROUP BY

        Fiscal_Year,

        Month,

        Month_Order,

        Entity_CD,

        Entity_Name,

        Currency,

        Currency_CD,

        Currency_Key,

        Account_CD,

        Account_Name


    ORDER BY

        Entity_Name,

        Currency_Key,

        Account_Name,

        Month_Order
    """


    job_config = (
        bigquery.QueryJobConfig(

            query_parameters=[

                bigquery.ScalarQueryParameter(
                    "year",
                    "INT64",
                    year,
                ),

                bigquery.ArrayQueryParameter(
                    "account_codes",
                    "STRING",
                    account_codes,
                ),
            ]
        )
    )


    return client.query(

        query,

        job_config=job_config,

    ).to_dataframe()


# ============================================================
# PERCENTAGE CHANGE
# ============================================================

def calculate_pct_change(
    current,
    previous,
):

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

        / abs(
            previous
        )

    ) * 100


# ============================================================
# NORMALIZE STRING COLUMNS
# ============================================================

def normalize_strings(df):

    string_columns = [

        "Month",

        "Entity_CD",

        "Entity_Name",

        "Currency",

        "Currency_CD",

        "Currency_Key",

        "Account_CD",

        "Account_Name",
    ]


    for column in string_columns:

        if column in df.columns:

            df[
                column
            ] = (

                df[
                    column
                ]

                .fillna("")

                .astype(str)

                .str.strip()
            )


    return df


# ============================================================
# BUILD COMPARISONS
# ============================================================

def add_comparisons(df):

    df = normalize_strings(
        df
    )


    # ========================================================
    # VERY IMPORTANT
    #
    # Currency_Key is now inside sorting and grouping.
    #
    # Therefore:
    #
    # Germany + EUR + EBITDA
    #
    # is completely separate from:
    #
    # Germany + LC + EBITDA
    #
    # ========================================================

    df = (

        df

        .sort_values(

            [

                "Entity_CD",

                "Currency_Key",

                "Account_CD",

                "Fiscal_Year",

                "Month_Order",
            ]
        )

        .reset_index(
            drop=True
        )
    )


    groups = df.groupby(

        [

            "Entity_CD",

            "Currency_Key",

            "Account_CD",
        ],

        sort=False,

        dropna=False,
    )


    # ========================================================
    # PREVIOUS MONTH
    # ========================================================

    df[
        "Previous_Month"
    ] = groups[
        "Month"
    ].shift(1)


    df[
        "Previous_Periodic_Actual"
    ] = groups[
        "Periodic_Actual"
    ].shift(1)


    # ========================================================
    # MONTH-ON-MONTH CHANGE
    # ========================================================

    df[
        "MoM_Change"
    ] = (

        df[
            "Periodic_Actual"
        ]

        -

        df[
            "Previous_Periodic_Actual"
        ]
    )


    df[
        "MoM_Pct"
    ] = df.apply(

        lambda row:

        calculate_pct_change(

            row[
                "Periodic_Actual"
            ],

            row[
                "Previous_Periodic_Actual"
            ],
        ),

        axis=1,
    )


    # ========================================================
    # TWO MONTHS AGO
    # ========================================================

    df[
        "Two_Months_Ago_Actual"
    ] = groups[
        "Periodic_Actual"
    ].shift(2)


    df[
        "Two_Month_Change"
    ] = (

        df[
            "Periodic_Actual"
        ]

        -

        df[
            "Two_Months_Ago_Actual"
        ]
    )


    # ========================================================
    # THREE MONTHS AGO
    # ========================================================

    df[
        "Three_Months_Ago_Actual"
    ] = groups[
        "Periodic_Actual"
    ].shift(3)


    df[
        "Three_Month_Change"
    ] = (

        df[
            "Periodic_Actual"
        ]

        -

        df[
            "Three_Months_Ago_Actual"
        ]
    )


    # ========================================================
    # ACTUAL MINUS BUDGET
    # ========================================================

    df[
        "Actual_Minus_Budget"
    ] = (

        df[
            "Periodic_Actual"
        ]

        -

        df[
            "Periodic_Budget"
        ]
    )


    # ========================================================
    # YTD ACTUAL MINUS BUDGET
    # ========================================================

    df[
        "YTD_Actual_Minus_Budget"
    ] = (

        df[
            "YTD_Actual"
        ]

        -

        df[
            "YTD_Budget"
        ]
    )


    # ========================================================
    # DIRECTION
    # ========================================================

    df[
        "MoM_Direction"
    ] = np.where(

        df[
            "MoM_Change"
        ].isna(),

        "NO_PREVIOUS_MONTH",

        np.where(

            df[
                "MoM_Change"
            ] > 0,

            "INCREASE",

            np.where(

                df[
                    "MoM_Change"
                ] < 0,

                "DECREASE",

                "NO_CHANGE",
            )
        )
    )


    # ========================================================
    # MATERIALITY INPUTS
    # ========================================================

    df[
        "Abs_MoM_Change"
    ] = (

        df[
            "MoM_Change"
        ]

        .abs()
    )


    df[
        "Abs_Budget_Variance"
    ] = (

        df[
            "Periodic_Vs_Budget"
        ]

        .abs()
    )


    df[
        "Abs_YoY_Change"
    ] = (

        df[
            "Periodic_YoY_Abs"
        ]

        .abs()
    )


    df[
        "Abs_YTD_Budget_Variance"
    ] = (

        df[
            "YTD_Vs_Budget"
        ]

        .abs()
    )


    # ========================================================
    # MATERIALITY SCORE
    # ========================================================
    #
    # Still purely numerical.
    #
    # Gemini does NOT decide which movement is material.
    #
    # We use:
    #
    # MoM
    # +
    # Periodic Budget Variance
    # +
    # YoY
    #
    # YTD is kept separately because otherwise
    # cumulative values can dominate every ranking.
    # ========================================================

    df[
        "Materiality_Score"
    ] = (

        df[
            "Abs_MoM_Change"
        ]
        .fillna(0)

        +

        df[
            "Abs_Budget_Variance"
        ]
        .fillna(0)

        +

        df[
            "Abs_YoY_Change"
        ]
        .fillna(0)
    )


    return df


# ============================================================
# VALIDATE GRAIN
# ============================================================

def validate_grain(df):

    duplicate_count = (

        df

        .duplicated(

            subset=[

                "Fiscal_Year",

                "Month",

                "Entity_CD",

                "Currency_Key",

                "Account_CD",
            ]
        )

        .sum()
    )


    print(
        "\nGRAIN VALIDATION"
    )

    print(
        "=" * 80
    )


    print(
        "Expected grain:"
    )

    print(
        "Fiscal Year × Month × Entity × Currency × Account"
    )


    print(
        f"\nDuplicate rows at expected grain: "
        f"{duplicate_count}"
    )


    if duplicate_count > 0:

        print(
            "\nWARNING:"
        )

        print(
            "Duplicates exist at the new Currency-aware grain."
        )


# ============================================================
# VALIDATE CURRENCY
# ============================================================

def validate_currency(df):

    print(
        "\nCURRENCY VALIDATION"
    )

    print(
        "=" * 80
    )


    currency_summary = (

        df[
            [

                "Currency_Key",

                "Currency",

                "Currency_CD",
            ]
        ]

        .drop_duplicates()

        .sort_values(
            "Currency_Key"
        )
    )


    print(
        currency_summary.to_string(
            index=False
        )
    )


    print(
        f"\nDistinct Currency Keys: "
        f"{df['Currency_Key'].nunique()}"
    )


    print(
        "\nRows by Currency:"
    )


    print(

        df[
            "Currency_Key"
        ]

        .value_counts()

        .to_string()
    )


# ============================================================
# CURRENCY / ENTITY EXAMPLE
# ============================================================

def print_currency_example(df):

    counts = (

        df[
            [

                "Entity_CD",

                "Entity_Name",

                "Currency_Key",
            ]
        ]

        .drop_duplicates()

        .groupby(
            [
                "Entity_CD",
                "Entity_Name",
            ]
        )[
            "Currency_Key"
        ]

        .nunique()

        .reset_index(
            name="Currency_Count"
        )

        .sort_values(
            "Currency_Count",
            ascending=False,
        )
    )


    multiple_currency = counts[

        counts[
            "Currency_Count"
        ] > 1

    ]


    if multiple_currency.empty:

        print(
            "\nNo entity with multiple currencies "
            "was found in this dataset."
        )

        return


    entity_cd = (

        multiple_currency[
            "Entity_CD"
        ].iloc[0]
    )


    entity_name = (

        multiple_currency[
            "Entity_Name"
        ].iloc[0]
    )


    sample = (

        df[

            df[
                "Entity_CD"
            ]
            == entity_cd

        ]

        [
            [

                "Entity_Name",

                "Month",

                "Currency_Key",

                "Account_Name",

                "Periodic_Actual",

                "Previous_Periodic_Actual",

                "MoM_Change",
            ]
        ]

        .sort_values(

            [

                "Currency_Key",

                "Account_Name",

                "Month",
            ]
        )

        .head(30)
    )


    print(
        "\nMULTI-CURRENCY SAMPLE"
    )

    print(
        "=" * 120
    )


    print(
        f"Entity: "
        f"{entity_name}"
    )


    print(
        sample.to_string(
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
        "BUILDING CURRENCY-AWARE PAGE-1 ENRICHED ANALYSIS"
    )

    print(
        "=" * 80
    )


    # ========================================================
    # LOAD ACCOUNT MAPPING
    # ========================================================

    mapping_df = (
        load_mapping()
    )


    account_codes = (

        mapping_df[
            "Account_CD"
        ]

        .dropna()

        .astype(str)

        .str.strip()

        .tolist()
    )


    print(
        f"\nLocked Page-1 accounts: "
        f"{len(account_codes)}"
    )


    # ========================================================
    # BIGQUERY CLIENT
    # ========================================================

    client = bigquery.Client(
        project=PROJECT_ID
    )


    latest_year = (
        get_latest_year(
            client
        )
    )


    if latest_year is None:

        print(
            "No Fiscal Year found."
        )

        return


    print(
        f"Latest Fiscal Year: "
        f"{latest_year}"
    )


    # ========================================================
    # EXTRACT
    # ========================================================

    print(
        "\nExtracting Currency-aware Page-1 data..."
    )


    df = extract_page1_data(

        client=

            client,

        year=

            latest_year,

        account_codes=

            account_codes,
    )


    if df.empty:

        print(
            "No Page-1 data returned."
        )

        return


    # ========================================================
    # CALCULATIONS
    # ========================================================

    print(
        "Building Currency-aware comparisons..."
    )


    df = add_comparisons(
        df
    )


    # ========================================================
    # BASIC VALIDATION
    # ========================================================

    print(
        "\nDATASET STATUS"
    )

    print(
        "=" * 80
    )


    print(
        f"Rows: "
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
        f"Accounts: "
        f"{df['Account_CD'].nunique()}"
    )


    print(
        f"Currencies: "
        f"{df['Currency_Key'].nunique()}"
    )


    # ========================================================
    # LOADED MONTHS
    # ========================================================

    month_summary = (

        df[
            [

                "Month_Order",

                "Month",
            ]
        ]

        .drop_duplicates()

        .sort_values(
            "Month_Order"
        )
    )


    print(
        "\nLoaded Months:"
    )


    print(

        ", ".join(

            month_summary[
                "Month"
            ].tolist()
        )
    )


    # ========================================================
    # VALIDATE
    # ========================================================

    validate_grain(
        df
    )


    validate_currency(
        df
    )


    print_currency_example(
        df
    )


    # ========================================================
    # SAVE
    # ========================================================

    OUTPUT_FILE.parent.mkdir(

        parents=True,

        exist_ok=True,
    )


    df.to_csv(

        OUTPUT_FILE,

        index=False,
    )


    print(
        "\n"
        + "=" * 80
    )


    print(
        "OUTPUT SAVED"
    )

    print(
        "=" * 80
    )


    print(
        OUTPUT_FILE.resolve()
    )


    print(
        "\nITERATION-2 CURRENCY-AWARE "
        "ENRICHED ANALYSIS COMPLETED."
    )


if __name__ == "__main__":

    main()
