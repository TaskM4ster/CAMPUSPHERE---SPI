# Datasets

## student-mat.csv (raw)

- **Source:** UCI Machine Learning Repository, Student Performance dataset
  (https://archive.ics.uci.edu/dataset/320/student+performance)
- **DOI:** 10.24432/C5TG7T
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Citation:** Cortez, P. (2008). Student Performance [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5TG7T
- **File used:** student-mat.csv (Mathematics course grades)
- **Delimiter:** semicolon (`;`). Load with `pd.read_csv(path, sep=";")`.
- **Records:** 395 students, 33 columns
- **Features:** 30 input variables covering demographics, family background,
  study habits, school support, and behavior (see the full list in the UCI documentation)
- **Target variable:** G3 (final grade, 0-20), converted to a risk label
  (at_risk: 0-9, needs_review: 10-12, on_track: 13-20)
- **Excluded from features:** G3 (it defines the label and would leak the answer)

## student-por.csv (raw)

- **Source:** same UCI Student Performance dataset, Portuguese course file
- **File used:** student-por.csv (Portuguese language course grades), used only for external validation
- **Records:** 649 students, 33 columns, same column names as the Mathematics file
- **Overlap with Mathematics file:** students present in both files are identified with the
  13-attribute rule in UCI's merge script (`student-merge.R`): school, sex, age, address,
  famsize, Pstatus, Medu, Fedu, Mjob, Fjob, reason, nursery, internet. The UCI script reports
  382 joined rows. Because the join can match one student more than once, the number of unique
  shared students is reported separately in the Results section.

## Data integrity

Both files were downloaded from the UCI Student Performance page on October 23, 2026. SHA-256 checksums
identify the exact file versions used in this project.

| File              | SHA-256                                                            |
| ----------------- | ------------------------------------------------------------------ |
| `student-mat.csv` | `E47F9EE225E1EE6E69B7564E6DAC7123E80B8486677FE111F351964CEF5DEC80` |
| `student-por.csv` | `A7594A11D7771C0EFE1A740824E0E833DA9C4CAD07C39A9766A874575563FB3F` |

To verify a copy, run this in PowerShell from the project folder:

```powershell
Get-FileHash datasets\raw\student-mat.csv -Algorithm SHA256
Get-FileHash datasets\raw\student-por.csv -Algorithm SHA256
```

**Replacement note:** An earlier version of `student-mat.csv` was a synthetic file that imitated
the UCI data. It was found during verification and replaced with the authentic UCI file. All
results generated from the synthetic file have been discarded. The synthetic file's SHA-256 is
`3AEA440DF3C2C3F8B8AE6C4CDA92504E36479F85A7D1DE2B58C98A6063163D79`.

## Limitations

- Data comes from two Portuguese secondary schools in one subject (Mathematics), so results may
  not transfer to other subjects, schools, or countries.
- Some students have G3 = 0. These are often students who dropped out, not only low performers,
  so the at_risk label mixes two different situations.
- G1 and G2 are strongly correlated with G3. Models that use them may mostly learn that low
  earlier grades predict low final grades.
- The data is a single snapshot, so it cannot show how a student's risk changes over time.
- Attendance is recorded as total absences, not per-subject or per-week attendance.
