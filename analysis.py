"""
Project 2: Student Academic Performance Analytics
Tasks 19-33 | Python, NumPy and Pandas

Run from the project root:
    python src/analysis.py

Dataset:
    data/02_student_performance.csv

The supplied teaching dataset is synthetic. This program reports patterns,
not causal relationships. Cleaning assumptions are documented below.
"""

from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_FILE = PROJECT_ROOT / "data" / "02_student_performance.csv"
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

MARK_COLUMNS = [
    "Python_Marks",
    "Mathematics_Marks",
    "Communication_Marks",
]


def load_data(file_path):
    """Load the CSV and return a DataFrame."""
    return pd.read_csv(file_path)


def inspect_data(df):
    """Task 19: display the structure and basic descriptive information."""
    print("\n" + "=" * 65)
    print("TASK 19 - LOAD AND INSPECT DATA")
    print("=" * 65)
    print("\nFirst five rows:\n", df.head())
    print("\nLast five rows:\n", df.tail())
    print("\nDataset shape (rows, columns):", df.shape)
    print("\nColumn names:\n", df.columns.tolist())
    print("\nDataFrame information:")
    df.info()
    print("\nNumerical descriptive statistics:\n", df.describe())


def report_data_quality(df):
    """Task 20: report missing values and duplicate Student_ID records."""
    print("\n" + "=" * 65)
    print("TASK 20 - MISSING VALUES AND DUPLICATE STUDENT IDS")
    print("=" * 65)
    print("\nMissing values by column:\n", df.isnull().sum())

    duplicate_rows = df[df["Student_ID"].duplicated(keep=False)].sort_values("Student_ID")
    print("\nRows involved in duplicate Student_ID values:\n")
    if duplicate_rows.empty:
        print("No duplicate Student_ID values found.")
    else:
        print(duplicate_rows.to_string(index=False))

    return duplicate_rows


def clean_data(df):
    """
    Tasks 21-22: clean missing attendance and invalid marks.

    Assumptions:
    - Missing attendance is replaced with the median attendance.
    - Duplicate Student_ID records: keep the first occurrence for this
      teaching dataset. In real institutional data, records should be
      reconciled against the source before deleting either record.
    - Marks outside 0-100 are invalid and are converted to NaN, not clipped.
    - Invalid marks remain missing; averages use the available valid subjects.
    """
    print("\n" + "=" * 65)
    print("TASKS 21-22 - DATA CLEANING")
    print("=" * 65)

    cleaned = df.copy()

    attendance_missing = int(cleaned["Attendance_Percentage"].isna().sum())
    attendance_median = cleaned["Attendance_Percentage"].median()
    cleaned["Attendance_Percentage"] = cleaned["Attendance_Percentage"].fillna(
        attendance_median
    )
    print(
        f"\nTask 21: filled {attendance_missing} missing attendance value(s) "
        f"using median = {attendance_median:.2f}%."
    )

    duplicate_count = int(cleaned["Student_ID"].duplicated(keep="first").sum())
    cleaned = cleaned.drop_duplicates(subset="Student_ID", keep="first").copy()
    print(
        f"Removed {duplicate_count} repeated Student_ID row(s), keeping the "
        "first occurrence as a documented teaching-data assumption."
    )

    invalid_summary = {}
    for column in MARK_COLUMNS:
        invalid_mask = (cleaned[column] < 0) | (cleaned[column] > 100)
        invalid_summary[column] = int(invalid_mask.sum())
        print(f"\nTask 22: invalid values in {column}:")
        if invalid_mask.any():
            print(cleaned.loc[invalid_mask, ["Student_ID", column]].to_string(index=False))
            cleaned.loc[invalid_mask, column] = np.nan
        else:
            print("None")

    print("\nInvalid score counts:", invalid_summary)
    return cleaned


def calculate_performance(df):
    """Tasks 23-25: calculate totals, averages, grades and department averages."""
    print("\n" + "=" * 65)
    print("TASKS 23-25 - PERFORMANCE CALCULATIONS")
    print("=" * 65)

    result = df.copy()
    # min_count=1 prevents a student with no valid marks from receiving total 0.
    result["Total_Marks"] = result[MARK_COLUMNS].sum(axis=1, min_count=1)
    # Pandas mean skips NaN, so the average uses only valid subject marks.
    result["Average_Marks"] = result[MARK_COLUMNS].mean(axis=1)

    # Grade bands are project-defined because the assignment does not prescribe them.
    conditions = [
        result["Average_Marks"] >= 90,
        result["Average_Marks"] >= 80,
        result["Average_Marks"] >= 70,
        result["Average_Marks"] >= 60,
        result["Average_Marks"] >= 40,
        result["Average_Marks"] < 40,
    ]
    labels = ["A+", "A", "B", "C", "D", "F"]
    result["Grade"] = np.select(
        conditions, labels, default="Incomplete"
    )

    print("\nTask 23-24: student totals, averages and grades:\n")
    print(
        result[
            ["Student_ID", "Total_Marks", "Average_Marks", "Grade"]
        ].round(2).to_string(index=False)
    )
    print("\nGrade distribution:\n", result["Grade"].value_counts())

    department_summary = (
        result.groupby("Department")["Average_Marks"]
        .mean()
        .sort_values(ascending=False)
        .rename("Department_Average_Marks")
        .reset_index()
    )
    print("\nTask 25: department-wise average marks:\n")
    print(department_summary.round(2).to_string(index=False))
