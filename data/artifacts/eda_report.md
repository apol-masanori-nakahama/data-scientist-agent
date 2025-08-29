
# EDA Report

**Input:** data/upload.csv

## Schema & Missingness
- Dtypes: see `cell_dtypes.csv`
- Missingness: see `cell_missing.csv`

## Raw Describe
     Unnamed: 0  count  unique               top   freq         mean         std     min       25%        50%      75%        max
0   PassengerId  418.0     NaN               NaN    NaN  1100.500000  120.810458  892.00  996.2500  1100.5000  1204.75  1309.0000
1        Pclass  418.0     NaN               NaN    NaN     2.265550    0.841838    1.00    1.0000     3.0000     3.00     3.0000
2          Name  418.0   418.0  Kelly, Mr. James    1.0          NaN         NaN     NaN       NaN        NaN      NaN        NaN
3           Sex  418.0     2.0              male  266.0          NaN         NaN     NaN       NaN        NaN      NaN        NaN
4           Age  332.0     NaN               NaN    NaN    30.272590   14.181209    0.17   21.0000    27.0000    39.00    76.0000
5         SibSp  418.0     NaN               NaN    NaN     0.447368    0.896760    0.00    0.0000     0.0000     1.00     8.0000
6         Parch  418.0     NaN               NaN    NaN     0.392344    0.981429    0.00    0.0000     0.0000     0.00     9.0000
7        Ticket  418.0   363.0          PC 17608    5.0          NaN         NaN     NaN       NaN        NaN      NaN        NaN
8          Fare  417.0     NaN               NaN    NaN    35.627188   55.907576    0.00    7.8958    14.4542    31.50   512.3292
9         Cabin   91.0    76.0   B57 B59 B63 B66    3.0          NaN         NaN     NaN       NaN        NaN      NaN        NaN
10     Embarked  418.0     3.0                 S  270.0          NaN         NaN     NaN       NaN        NaN      NaN        NaN