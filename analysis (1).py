"""Student Academic Performance Analytics
Tasks 19-33 | Python, Object-Oriented Programming, NumPy and Pandas

Run from the project root:
    python src/analysis.py

The script also works if analysis.py is copied to the project root, provided
that the data folder remains beside it.

The supplied CSV is kept unchanged. Cleaning is performed on an in-memory copy
and exported separately to output/cleaned_student_data.csv.
"""

from pathlib import Path
import numpy as np
import pandas as pd


class StudentPerformanceAnalyzer:
    """Load, clean, analyse and export student performance data."""

    MARK_COLUMNS = [
        "Python_Marks",
        "Mathematics_Marks",
        "Communication_Marks",
    ]
    REQUIRED_COLUMNS = [
        "Student_ID",
        "Department",
        "Attendance_Percentage",
        "Study_Hours_Per_Day",
        "Internet_Access",
        *MARK_COLUMNS,
    ]

    def __init__(self, dataset_path=None, output_dir=None):
        script_dir = Path(__file__).resolve().parent
        project_root = script_dir.parent if script_dir.name.lower() == "src" else script_dir

        if dataset_path is not None:
            self.dataset_path = Path(dataset_path).expanduser().resolve()
            project_root = self.dataset_path.parent.parent if self.dataset_path.parent.name.lower() == "data" else self.dataset_path.parent
        else:
            possible_paths = [
                project_root / "data" / "02_student_performance.csv",
                script_dir / "data" / "02_student_performance.csv",
                script_dir / "02_student_performance.csv",
            ]
            self.dataset_path = next((p for p in possible_paths if p.exists()), possible_paths[0])

        self.output_dir = Path(output_dir).resolve() if output_dir else project_root / "output"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.raw_df = None
        self.df = None
        self.duplicate_rows = None
        self.invalid_score_rows = None
        self.department_summary = None
        self.top_ten = None
        self.at_risk = None
        self.attendance_summary = None
        self.study_summary = None
        self.internet_summary = None
        self.outliers = None
        self.recommendations = []

    @staticmethod
    def heading(title):
        print("\n" + "=" * 76)
        print(title)
        print("=" * 76)

    def load_data(self):
        """Task 19: load the CSV and validate its required columns."""
        if not self.dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {self.dataset_path}\n"
                "Place 02_student_performance.csv inside the project's data folder."
            )
        self.raw_df = pd.read_csv(self.dataset_path)
        missing_columns = sorted(set(self.REQUIRED_COLUMNS) - set(self.raw_df.columns))
        if missing_columns:
            raise ValueError(f"CSV is missing required columns: {missing_columns}")
        self.heading("TASK 19 - LOAD AND INSPECT DATA")
        print("Dataset:", self.dataset_path)
        print("\nFirst five records:\n", self.raw_df.head().to_string(index=False))
        print("\nLast five records:\n", self.raw_df.tail().to_string(index=False))
        print("\nDataset dimensions:", self.raw_df.shape)
        print("\nColumn names:", self.raw_df.columns.tolist())
        print("\nData types:\n", self.raw_df.dtypes)
        print("\nDescriptive statistics:\n", self.raw_df.describe(include="all").to_string())
        return self.raw_df

    def inspect_data_quality(self):
        """Task 20: report missing values, duplicate IDs and invalid marks."""
        self.heading("TASK 20 - DATA QUALITY REPORT")
        print("Missing values by column:\n", self.raw_df.isna().sum().to_string())

        self.duplicate_rows = self.raw_df[
            self.raw_df["Student_ID"].duplicated(keep=False)
        ].sort_values("Student_ID")
        print("\nRows containing repeated Student_ID values:")
        if self.duplicate_rows.empty:
            print("No duplicate Student_ID values found.")
        else:
            print(self.duplicate_rows.to_string(index=False))

        invalid_parts = []
        for column in self.MARK_COLUMNS:
            invalid = self.raw_df[column].notna() & ~self.raw_df[column].between(0, 100)
            if invalid.any():
                part = self.raw_df.loc[invalid, ["Student_ID", column]].copy()
                part["Score_Column"] = column
                part = part.rename(columns={column: "Invalid_Score"})
                invalid_parts.append(part)
        self.invalid_score_rows = (
            pd.concat(invalid_parts, ignore_index=True)
            if invalid_parts else pd.DataFrame(columns=["Student_ID", "Invalid_Score", "Score_Column"])
        )
        print("\nMarks outside the valid 0-100 range:")
        print(self.invalid_score_rows.to_string(index=False) if not self.invalid_score_rows.empty else "None found.")

    def clean_data(self):
        """Tasks 21-22: impute attendance and invalidate out-of-range marks."""
        self.heading("TASKS 21-22 - DATA CLEANING")
        self.df = self.raw_df.copy(deep=True)

        missing_attendance = int(self.df["Attendance_Percentage"].isna().sum())
        attendance_median = self.df["Attendance_Percentage"].median()
        self.df["Attendance_Percentage"] = self.df["Attendance_Percentage"].fillna(attendance_median)
        print(f"Missing attendance values filled: {missing_attendance}; median used: {attendance_median:.2f}%")

        duplicate_count = int(self.df["Student_ID"].duplicated(keep="first").sum())
        self.df = self.df.drop_duplicates(subset="Student_ID", keep="first").copy()
        print(f"Repeated Student_ID rows removed: {duplicate_count} (first occurrence retained for this exercise).")

        invalid_count = 0
        for column in self.MARK_COLUMNS:
            invalid_mask = self.df[column].notna() & ~self.df[column].between(0, 100)
            invalid_count += int(invalid_mask.sum())
            self.df.loc[invalid_mask, column] = np.nan
        print(f"Out-of-range subject marks converted to missing values: {invalid_count}.")
        print("The source CSV has not been changed; all cleaning is applied to a copy.")
        return self.df

    def calculate_student_performance(self):
        """Tasks 23-24: calculate totals, averages and grades."""
        self.heading("TASKS 23-24 - TOTALS, AVERAGES AND GRADES")
        valid_marks = self.df[self.MARK_COLUMNS]
        self.df["Total_Marks"] = valid_marks.sum(axis=1, min_count=1)
        self.df["Average_Marks"] = valid_marks.mean(axis=1)

        # Lambda function: assign a grade using the project's documented cut-offs.
        grade = lambda score: (  # noqa: E731 - intentionally demonstrates lambda
            "Incomplete" if pd.isna(score) else
            "A+" if score >= 90 else
            "A" if score >= 80 else
            "B" if score >= 70 else
            "C" if score >= 60 else
            "D" if score >= 40 else "F"
        )
        self.df["Grade"] = self.df["Average_Marks"].apply(grade)
        print("Grade thresholds: A+ >=90, A >=80, B >=70, C >=60, D >=40, F <40.")
        print("\nStudent totals, averages and grades (sample):\n",
              self.df[["Student_ID", "Total_Marks", "Average_Marks", "Grade"]].head(10).to_string(index=False))
        print("\nGrade distribution:\n", self.df["Grade"].value_counts(dropna=False).sort_index().to_string())

    def analyse_groups(self):
        """Tasks 25-30: department, ranking, risk and group comparisons."""
        self.heading("TASKS 25-30 - GROUP AND PERFORMANCE ANALYSIS")
        self.department_summary = self.df.groupby("Department", dropna=False).agg(
            Student_Count=("Student_ID", "count"),
            Average_Marks=("Average_Marks", "mean"),
            Average_Attendance=("Attendance_Percentage", "mean"),
        ).reset_index().sort_values("Average_Marks", ascending=False)
        print("\nTask 25 - Department-wise averages:\n", self.department_summary.round(2).to_string(index=False))

        self.top_ten = self.df.nlargest(10, "Average_Marks")[
            ["Student_ID", "Department", "Average_Marks", "Grade"]
        ].copy()
        print("\nTask 26 - Top 10 students:\n", self.top_ten.round(2).to_string(index=False))

        self.at_risk = self.df[self.df["Average_Marks"] < 40][
            ["Student_ID", "Department", "Average_Marks", "Grade"]
        ].sort_values("Average_Marks")
        print("\nTask 27 - Students below 40% average:\n",
              self.at_risk.round(2).to_string(index=False) if not self.at_risk.empty else "None found.")

        attendance_bins = [-np.inf, 60, 75, 90, np.inf]
        attendance_labels = ["Below 60%", "60-74.9%", "75-89.9%", "90% and above"]
        self.df["Attendance_Band"] = pd.cut(
            self.df["Attendance_Percentage"], bins=attendance_bins,
            labels=attendance_labels, right=False
        )
        self.attendance_summary = self.df.groupby("Attendance_Band", observed=False).agg(
            Student_Count=("Student_ID", "count"), Average_Marks=("Average_Marks", "mean")
        ).reset_index()
        print("\nTask 28 - Average marks by attendance band:\n", self.attendance_summary.round(2).to_string(index=False))

        study_bins = [-np.inf, 2, 4, 6, np.inf]
        study_labels = ["Below 2 hours", "2-3.9 hours", "4-5.9 hours", "6 hours and above"]
        self.df["Study_Hours_Band"] = pd.cut(
            self.df["Study_Hours_Per_Day"], bins=study_bins,
            labels=study_labels, right=False
        )
        self.study_summary = self.df.groupby("Study_Hours_Band", observed=False).agg(
            Student_Count=("Student_ID", "count"), Average_Marks=("Average_Marks", "mean")
        ).reset_index()
        print("\nTask 29 - Average marks by study-hour band:\n", self.study_summary.round(2).to_string(index=False))
        correlation = self.df["Study_Hours_Per_Day"].corr(self.df["Average_Marks"])
        print(f"Study-hours / average-marks Pearson correlation: {correlation:.3f}")

        self.internet_summary = self.df.groupby("Internet_Access", dropna=False).agg(
            Student_Count=("Student_ID", "count"), Average_Marks=("Average_Marks", "mean")
        ).reset_index()
        print("\nTask 30 - Internet access comparison:\n", self.internet_summary.round(2).to_string(index=False))
        return correlation

    def calculate_numpy_statistics(self):
        """Task 31: calculate median, population SD and quartiles with NumPy."""
        self.heading("TASK 31 - NUMPY STATISTICS FOR PYTHON MARKS")
        values = self.df["Python_Marks"].dropna().to_numpy(dtype=float)
        if values.size == 0:
            print("No valid Python_Marks values available.")
            return {}
        q1, median, q3 = np.percentile(values, [25, 50, 75])
        stats = {
            "count": int(values.size),
            "median": float(np.median(values)),
            "population_std": float(np.std(values, ddof=0)),
            "q1": float(q1),
            "q2_median": float(median),
            "q3": float(q3),
        }
        for key, value in stats.items():
            print(f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}")
        return stats

    def detect_iqr_outliers(self):
        """Task 32: detect potential Python score outliers using the IQR rule."""
        self.heading("TASK 32 - IQR OUTLIER DETECTION")
        scores = self.df["Python_Marks"].dropna()
        q1 = scores.quantile(0.25)
        q3 = scores.quantile(0.75)
        iqr = q3 - q1
        lower_fence = q1 - 1.5 * iqr
        upper_fence = q3 + 1.5 * iqr
        self.outliers = self.df[
            self.df["Python_Marks"].notna() &
            ((self.df["Python_Marks"] < lower_fence) | (self.df["Python_Marks"] > upper_fence))
        ][["Student_ID", "Python_Marks", "Average_Marks"]].copy()
        print(f"Q1={q1:.2f}, Q3={q3:.2f}, IQR={iqr:.2f}")
        print(f"Lower fence={lower_fence:.2f}; upper fence={upper_fence:.2f}")
        print("Potential outliers:\n", self.outliers.to_string(index=False) if not self.outliers.empty else "None found.")
        print("An IQR flag is a prompt for review, not proof of an incorrect score.")
        return self.outliers

    def create_recommendations(self, study_correlation):
        """Task 33: create five recommendations, mindful of descriptive limits."""
        self.heading("TASK 33 - FIVE ACADEMIC SUPPORT RECOMMENDATIONS")
        recommendations = [
            f"Provide targeted tutoring and regular progress checks for the {len(self.at_risk)} student(s) below the 40% average threshold.",
            "Monitor attendance and assessment performance together; contact students in lower-attendance bands early without assuming attendance alone causes results.",
            (f"Encourage structured study plans and review their effectiveness; the observed study-hours correlation is {study_correlation:.2f}, which is descriptive rather than causal."),
            "Check whether students without internet access face barriers to digital resources; offer campus access or downloadable learning material where needed.",
            "Review subject-level marks and IQR-flagged scores individually before intervention; distinguish genuine unusual performance from data-entry issues.",
        ]
        self.recommendations = recommendations
        for number, item in enumerate(recommendations, start=1):
            print(f"{number}. {item}")
        return recommendations

    def save_outputs(self, numpy_stats, study_correlation):
        """Export the cleaned data and task results without altering the raw CSV."""
        self.heading("SAVING PROJECT OUTPUTS")
        outputs = {
            "cleaned_student_data.csv": self.df,
            "department_summary.csv": self.department_summary,
            "top_10_students.csv": self.top_ten,
            "at_risk_students_below_40.csv": self.at_risk,
            "attendance_band_analysis.csv": self.attendance_summary,
            "study_hours_analysis.csv": self.study_summary,
            "internet_access_comparison.csv": self.internet_summary,
            "python_marks_iqr_outliers.csv": self.outliers,
            "duplicate_student_id_records.csv": self.duplicate_rows,
            "invalid_score_records.csv": self.invalid_score_rows,
        }
        for filename, table in outputs.items():
            table.to_csv(self.output_dir / filename, index=False)

        with (self.output_dir / "analysis_summary.txt").open("w", encoding="utf-8") as report:
            report.write("STUDENT ACADEMIC PERFORMANCE ANALYTICS\n")
            report.write("Descriptive analysis of a synthetic teaching dataset\n\n")
            report.write(f"Source dataset: {self.dataset_path.name}\n")
            report.write(f"Original rows: {len(self.raw_df)}\n")
            report.write(f"Rows after duplicate-ID handling: {len(self.df)}\n")
            report.write(f"Study-hours Pearson correlation: {study_correlation:.3f}\n")
            report.write("\nNumPy statistics (Python marks):\n")
            for key, value in numpy_stats.items():
                report.write(f"{key}: {value}\n")
            report.write("\nRecommendations:\n")
            for i, recommendation in enumerate(self.recommendations, 1):
                report.write(f"{i}. {recommendation}\n")
            report.write("\nInterpretation note: associations are descriptive and do not establish causation.\n")
        print(f"Saved {len(outputs) + 1} output files to: {self.output_dir}")

    def run(self):
        """Execute the complete assignment workflow in order."""
        self.load_data()
        self.inspect_data_quality()
        self.clean_data()
        self.calculate_student_performance()
        study_correlation = self.analyse_groups()
        numpy_stats = self.calculate_numpy_statistics()
        self.detect_iqr_outliers()
        self.create_recommendations(study_correlation)
        self.save_outputs(numpy_stats, study_correlation)
        self.heading("PROJECT COMPLETED")
        print("All analysis tasks (19-33) have been executed successfully.")
        print("Original CSV retained unchanged.")


def main():
    analyzer = StudentPerformanceAnalyzer()
    analyzer.run()


if __name__ == "__main__":
    main()
