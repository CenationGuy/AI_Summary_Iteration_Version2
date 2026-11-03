from pathlib import Path

import numpy as np
import pandas as pd
from google.cloud import bigquery

from config import PROJECT_ID, DATASET_ID


# ============================================================
# FILES
# ============================================================

MAPPING_FILE = Path("output/page1_final_account_mapping.csv")

OUTPUT_FILE = Path("output/page1_enriched_analysis.csv")


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
# CURRENCY MAPPING
# ============================================================

CURRENCY_MAPPING_SQL = """
CASE

    WHEN UPPER(TRIM(Currency_CD)) = 'PARENTCURRENCY TOTAL'
        THEN 'EUR'

    WHEN UPPER(TRIM(Currency_CD)) = 'ENTITY CURRENCY TOTAL'
        THEN 'LC'

    ELSE NULL

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

    value = df.iloc[0]["latest_year"]

    if pd.isna(value):
        return None

    return int(value)


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

    mapping_df["Account_CD"] = (
        mapping_df["Account_CD"]
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

            -- ==============================================
            -- RAW CURRENCY FIELD FROM SOURCE
            -- ==============================================

            Currency_CD AS Currency_CD_Raw,

            -- ==============================================
            -- ITERATION-2 BUSINESS CURRENCY
            --
            -- ParentCurrency Total
            --      → EUR
            --
            -- Entity Currency Total
            --      → LC
            -- ==============================================

            {CURRENCY_MAPPING_SQL}
                AS Currency,

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

          AND Account_CD IN UNNEST(
              @account_codes
          )

          -- ==============================================
          -- ONLY THE TWO REQUIRED CURRENCY VIEWS
          -- ==============================================

          AND UPPER(
                TRIM(
                    Currency_CD
                )
              ) IN (

                'PARENTCURRENCY TOTAL',

                'ENTITY CURRENCY TOTAL'
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

        Currency_CD_Raw,

        Account_CD,

        Account_Name,


        -- ==============================================
        -- PERIODIC ACTUAL
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


        -- ==============================================
        -- PERIODIC BUDGET
        -- ==============================================

        MAX(

            CASE

                WHEN View_CD =
                     'Periodic'

                 AND Scenario_CD =
                     'BUDGET_CONS'

                THEN Value

            END

        ) AS Periodic_Budget,


        -- ==============================================
        -- PERIODIC VS BUDGET
        -- ==============================================

        MAX(

            CASE

                WHEN View_CD =
                     'Periodic'

                 AND Scenario_CD =
                     'ACTUAL_RESTATE_V_BUDGET_CONS'

                THEN Value

            END

        ) AS Periodic_Vs_Budget,


        -- ==============================================
        -- PERIODIC YOY
        -- ==============================================

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
        -- YTD ACTUAL
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


        -- ==============================================
        -- YTD BUDGET
        -- ==============================================

        MAX(

            CASE

                WHEN View_CD =
                     'Year-To-Date'

                 AND Scenario_CD =
                     'BUDGET_CONS'

                THEN Value

            END

        ) AS YTD_Budget,


        -- ==============================================
        -- YTD VS BUDGET
        -- ==============================================

        MAX(

            CASE

                WHEN View_CD =
                     'Year-To-Date'

                 AND Scenario_CD =
                     'ACTUAL_RESTATE_V_BUDGET_CONS'

                THEN Value

            END

        ) AS YTD_Vs_Budget,


        -- ==============================================
        -- YTD YOY
        -- ==============================================

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

    WHERE Currency IS NOT NULL


    GROUP BY

        Fiscal_Year,

        Month,

        Month_Order,

        Entity_CD,

        Entity_Name,

        Currency,

        Currency_CD_Raw,

        Account_CD,

        Account_Name


    ORDER BY

        Entity_Name,

        Currency,

        Account_Name,

        Month_Order
    """


    job_config = bigquery.QueryJobConfig(

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
        /
        abs(previous)
    ) * 100


# ============================================================
# NORMALIZE STRINGS
# ============================================================

def normalize_strings(df):

    string_columns = [

        "Month",

        "Entity_CD",

        "Entity_Name",

        "Currency",

        "Currency_CD_Raw",

        "Account_CD",

        "Account_Name",
    ]


    for column in string_columns:

        if column in df.columns:

            df[column] = (
                df[column]
                .fillna("")
                .astype(str)
                .str.strip()
            )

    return df


# ============================================================
# ADD COMPARISONS
# ============================================================

def add_comparisons(df):

    df = normalize_strings(
        df
    )


    # ========================================================
    # SORT AT CORRECT ITERATION-2 GRAIN
    #
    # Entity
    # × Currency
    # × Account
    # × Month
    # ========================================================

    df = (
        df
        .sort_values(
            [
                "Entity_CD",
                "Currency",
                "Account_CD",
                "Fiscal_Year",
                "Month_Order",
            ]
        )
        .reset_index(
            drop=True
        )
    )


    # ========================================================
    # CURRENCY IS CRITICAL HERE
    #
    # EUR will only compare with EUR.
    # LC will only compare with LC.
    # ========================================================

    groups = df.groupby(
        [
            "Entity_CD",
            "Currency",
            "Account_CD",
        ],
        sort=False,
        dropna=False,
    )


    # ========================================================
    # PREVIOUS MONTH
    # ========================================================

    df["Previous_Month"] = (
        groups["Month"]
        .shift(1)
    )


    df["Previous_Periodic_Actual"] = (
        groups["Periodic_Actual"]
        .shift(1)
    )


    # ========================================================
    # MONTH-ON-MONTH
    # ========================================================

    df["MoM_Change"] = (
        df["Periodic_Actual"]
        -
        df["Previous_Periodic_Actual"]
    )


    df["MoM_Pct"] = df.apply(
        lambda row:
        calculate_pct_change(
            row["Periodic_Actual"],
            row["Previous_Periodic_Actual"],
        ),
        axis=1,
    )


    # ========================================================
    # TWO MONTHS AGO
    # ========================================================

    df["Two_Months_Ago_Actual"] = (
        groups["Periodic_Actual"]
        .shift(2)
    )


    df["Two_Month_Change"] = (
        df["Periodic_Actual"]
        -
        df["Two_Months_Ago_Actual"]
    )


    # ========================================================
    # THREE MONTHS AGO
    # ========================================================

    df["Three_Months_Ago_Actual"] = (
        groups["Periodic_Actual"]
        .shift(3)
    )


    df["Three_Month_Change"] = (
        df["Periodic_Actual"]
        -
        df["Three_Months_Ago_Actual"]
    )


    # ========================================================
    # ACTUAL MINUS BUDGET
    # ========================================================

    df["Actual_Minus_Budget"] = (
        df["Periodic_Actual"]
        -
        df["Periodic_Budget"]
    )


    # ========================================================
    # YTD ACTUAL MINUS BUDGET
    # ========================================================

    df["YTD_Actual_Minus_Budget"] = (
        df["YTD_Actual"]
        -
        df["YTD_Budget"]
    )


    # ========================================================
    # MOM DIRECTION
    # ========================================================

    df["MoM_Direction"] = np.where(

        df["MoM_Change"].isna(),

        "NO_PREVIOUS_MONTH",

        np.where(

            df["MoM_Change"] > 0,

            "INCREASE",

            np.where(

                df["MoM_Change"] < 0,

                "DECREASE",

                "NO_CHANGE",
            )
        )
    )


    # ========================================================
    # ABSOLUTE MOVEMENTS
    # ========================================================

    df["Abs_MoM_Change"] = (
        df["MoM_Change"]
        .abs()
    )


    df["Abs_Budget_Variance"] = (
        df["Periodic_Vs_Budget"]
        .abs()
    )


    df["Abs_YoY_Change"] = (
        df["Periodic_YoY_Abs"]
        .abs()
    )


    df["Abs_YTD_Budget_Variance"] = (
        df["YTD_Vs_Budget"]
        .abs()
    )


    # ========================================================
    # MATERIALITY SCORE
    # ========================================================

    df["Materiality_Score"] = (

        df["Abs_MoM_Change"]
        .fillna(0)

        +

        df["Abs_Budget_Variance"]
        .fillna(0)

        +

        df["Abs_YoY_Change"]
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
                "Currency",
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


    if duplicate_count == 0:

        print(
            "PASS: No duplicates at the expected grain."
        )

    else:

        print(
            "WARNING: Duplicate rows exist."
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


    mapping = (
        df[
            [
                "Currency_CD_Raw",
                "Currency",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            "Currency"
        )
    )


    print(
        mapping.to_string(
            index=False
        )
    )


    print(
        f"\nDistinct mapped currencies: "
        f"{df['Currency'].nunique()}"
    )


    print(
        "\nRows by Currency:"
    )


    print(
        df["Currency"]
        .value_counts()
        .to_string()
    )


    currencies = set(
        df["Currency"]
        .dropna()
        .unique()
    )


    expected = {
        "EUR",
        "LC",
    }


    if currencies == expected:

        print(
            "\nPASS: Currency mapping contains EUR and LC."
        )

    else:

        print(
            "\nWARNING:"
        )

        print(
            f"Expected currencies: {expected}"
        )

        print(
            f"Found currencies: {currencies}"
        )


# ============================================================
# SHOW EUR / LC SAMPLE
# ============================================================

def print_currency_sample(df):

    entity_currency_counts = (
        df[
            [
                "Entity_CD",
                "Entity_Name",
                "Currency",
            ]
        ]
        .drop_duplicates()
        .groupby(
            [
                "Entity_CD",
                "Entity_Name",
            ]
        )["Currency"]
        .nunique()
        .reset_index(
            name="Currency_Count"
        )
        .sort_values(
            "Currency_Count",
            ascending=False,
        )
    )


    candidates = entity_currency_counts[
        entity_currency_counts[
            "Currency_Count"
        ] >= 2
    ]


    if candidates.empty:

        print(
            "\nNo entity with both EUR and LC "
            "was found."
        )

        return


    entity_cd = (
        candidates.iloc[0][
            "Entity_CD"
        ]
    )


    entity_name = (
        candidates.iloc[0][
            "Entity_Name"
        ]
    )


    sample = (
        df[
            df["Entity_CD"]
            ==
            entity_cd
        ]
        [
            [
                "Entity_Name",
                "Month",
                "Currency",
                "Currency_CD_Raw",
                "Account_Name",
                "Periodic_Actual",
                "Previous_Periodic_Actual",
                "MoM_Change",
                "MoM_Pct",
            ]
        ]
        .sort_values(
            [
                "Account_Name",
                "Currency",
                "Month_Order"
            ]
            if "Month_Order" in df.columns
            else
            [
                "Account_Name",
                "Currency",
                "Month",
            ]
        )
    )


    print(
        "\nEUR / LC VALIDATION SAMPLE"
    )

    print(
        "=" * 120
    )


    print(
        f"Entity: {entity_name}"
    )


    print(
        sample.head(40)
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
        "PAGE-1 CURRENCY-AWARE ENRICHED ANALYSIS"
    )

    print(
        "=" * 80
    )


    # ========================================================
    # ACCOUNT MAPPING
    # ========================================================

    mapping_df = load_mapping()


    account_codes = (
        mapping_df["Account_CD"]
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
    # BIGQUERY
    # ========================================================

    client = bigquery.Client(
        project=PROJECT_ID
    )


    latest_year = get_latest_year(
        client
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
        "\nExtracting Page-1 data..."
    )


    df = extract_page1_data(

        client=client,

        year=latest_year,

        account_codes=account_codes,
    )


    if df.empty:

        print(
            "No Page-1 data returned."
        )

        return


    # ========================================================
    # ADD COMPARISONS
    # ========================================================

    print(
        "Building EUR / LC-aware comparisons..."
    )


    df = add_comparisons(
        df
    )


    # ========================================================
    # STATUS
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
        f"{df['Currency'].nunique()}"
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
            month_summary["Month"]
            .tolist()
        )
    )


    # ========================================================
    # VALIDATIONS
    # ========================================================

    validate_grain(
        df
    )


    validate_currency(
        df
    )


    print_currency_sample(
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
        +
        "=" * 80
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
        "\nITERATION-2 ENRICHED ANALYSIS COMPLETED."
    )


if __name__ == "__main__":

    main()
