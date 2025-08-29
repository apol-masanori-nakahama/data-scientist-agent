import os
os.environ['MPLBACKEND']='Agg'
try:
    import resource
    resource.setrlimit(resource.RLIMIT_CPU, (30, 30))
    resource.setrlimit(resource.RLIMIT_AS, (1600000000, 1600000000))
except Exception:
    pass
INPUT_CSV=r'''data/sample.csv'''

import pandas as pd
df = pd.read_csv(INPUT_CSV)
miss = df.isna().sum().reset_index()
miss.columns = ["column","na_count"]
miss.to_csv("data/artifacts/cell_missing.csv", index=False)
