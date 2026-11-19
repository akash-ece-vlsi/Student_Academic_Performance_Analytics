# Student Academic Performance Analytics

## Project overview
This individual Python project analyses a synthetic student dataset to explore
academic performance in relation to attendance, study hours and internet access.
It identifies students below a defined performance threshold and presents
descriptive findings that can inform academic-support decisions.

## Technology
- Python 3.9+
- Pandas
- NumPy

## Dataset
`data/02_student_performance.csv`

The dataset contains 100 synthetic student records and these fields:
`Student_ID`, `Department`, `Attendance_Percentage`, `Study_Hours_Per_Day`,
`Internet_Access`, `Python_Marks`, `Mathematics_Marks`, and
`Communication_Marks`.

This is a synthetic teaching dataset, not real student records.

## Setup and execution
1. Install Python 3.9 or later.
2. Open a terminal in the project root.
3. Install dependencies:

   ```bash
   python -m pip install pandas numpy
   ```

4. Run:

   ```bash
   python src/analysis.py
   ```

The program prints the requested analysis and writes CSV result tables to
`output/`.

## Tasks covered
Tasks 19-33: dataset loading and inspection; missing/duplicate checks; attendance
imputation; invalid-score checks; totals, averages and grades; department
comparison; top 10 and below-40% students; attendance, study-hours and internet
comparisons; NumPy statistics; IQR outlier detection; and five recommendations.

## Data-cleaning and analysis assumptions
- Missing attendance values are filled using the median attendance.
- Duplicate `Student_ID` rows are reported. For this teaching dataset, the first
  occurrence is retained. In a real system, duplicate records must be reconciled
  against the source before one is discarded.
- Scores below 0 or above 100 are invalid and converted to missing values; they
  are not clipped to the nearest valid score.
- Student averages use the available valid subject scores. A student with no
  valid subject scores receives an `Incomplete` grade.
- Grade bands are defined in the project because the assignment does not specify
  exact grade cut-offs: A+ (90-100), A (80-<90), B (70-<80), C (60-<70),
  D (40-<60), F (<40).
- Attendance and study-hour bands are explicitly defined in the source code.
- NumPy standard deviation uses the population definition (`ddof=0`).
- IQR outliers are potential statistical outliers, not automatically data errors.
- Comparisons and correlation are descriptive; they do not establish causation.

## Repository structure
```text
Student_Academic_Performance_Analytics/
├── SRS/
│   ├── Project_SRS.docx
│   └── Project_SRS.pdf
├── src/
│   └── analysis.py
├── data/
│   └── 02_student_performance.csv
├── output/
│   └── (generated CSV analysis tables)
└── README.md
```

## Reproducibility
Keep the dataset in the `data` folder and run the script from any working
directory; paths are resolved relative to the script location.
