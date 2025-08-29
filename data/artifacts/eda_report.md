
# EDA Report

**Input:** data/upload.csv

## Schema & Missingness
- Dtypes: see `cell_dtypes.csv`
- Missingness: see `cell_missing.csv`

## Raw Describe
     Unnamed: 0  count  unique                      top   freq        mean         std   min       25%       50%    75%       max
0   PassengerId  891.0     NaN                      NaN    NaN  446.000000  257.353842  1.00  223.5000  446.0000  668.5  891.0000
1      Survived  891.0     NaN                      NaN    NaN    0.383838    0.486592  0.00    0.0000    0.0000    1.0    1.0000
2        Pclass  891.0     NaN                      NaN    NaN    2.308642    0.836071  1.00    2.0000    3.0000    3.0    3.0000
3          Name  891.0   891.0  Braund, Mr. Owen Harris    1.0         NaN         NaN   NaN       NaN       NaN    NaN       NaN
4           Sex  891.0     2.0                     male  577.0         NaN         NaN   NaN       NaN       NaN    NaN       NaN
5           Age  714.0     NaN                      NaN    NaN   29.699118   14.526497  0.42   20.1250   28.0000   38.0   80.0000
6         SibSp  891.0     NaN                      NaN    NaN    0.523008    1.102743  0.00    0.0000    0.0000    1.0    8.0000
7         Parch  891.0     NaN                      NaN    NaN    0.381594    0.806057  0.00    0.0000    0.0000    0.0    6.0000
8        Ticket  891.0   681.0                   347082    7.0         NaN         NaN   NaN       NaN       NaN    NaN       NaN
9          Fare  891.0     NaN                      NaN    NaN   32.204208   49.693429  0.00    7.9104   14.4542   31.0  512.3292
10        Cabin  204.0   147.0                  B96 B98    4.0         NaN         NaN   NaN       NaN       NaN    NaN       NaN
11     Embarked  889.0     3.0                        S  644.0         NaN         NaN   NaN       NaN       NaN    NaN       NaN