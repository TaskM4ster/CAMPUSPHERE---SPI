# Datasets

## student-mat.csv (raw)

- **Source:** UCI Machine Learning Repository, Student Performance dataset
  (https://archive.ics.uci.edu/dataset/320/student+performance)
- **File used:** student-mat.csv (Mathematics course grades)
- **License / usage terms:** 10.24432/C5TG7T
- **Citation:** Cortez, P. (2008). Student Performance [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5TG7T.
- **Records:** 395 students
- **Features:** 30 input variables covering demographics, family background,
  study habits, school support, and behavior
  (see the full list below)
- **Target variable:** G3 (final grade, 0-20), converted to a risk label
  (at_risk: 0-9, needs_review: 10-12, on_track: 13-20)
- **Excluded from features:** G3 (it defines the label and would leak the answer)

## Limitations

- Data comes from two Portuguese secondary schools in one subject (Mathematics),
  so results may not transfer to other subjects, schools, or countries.
- Some students have G3 = 0. These are often students who dropped out, not
  only low performers, so the at_risk label mixes two different situations.
- G1 and G2 are strongly correlated with G3. Models that use them may mostly
  learn that low earlier grades predict low final grades.
- The data is a single snapshot, so it cannot show how a student's risk changes over time.
- Attendance is recorded as total absences, not per-subject or per-week attendance.
