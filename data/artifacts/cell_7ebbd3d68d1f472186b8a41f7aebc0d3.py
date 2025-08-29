import os
os.environ['MPLBACKEND']='Agg'
try:
    import resource
    resource.setrlimit(resource.RLIMIT_CPU, (30, 30))
    resource.setrlimit(resource.RLIMIT_AS, (1600000000, 1600000000))
except Exception:
    pass
INPUT_CSV=r'''data/upload.csv'''

import pandas as pd
df = pd.read_csv(INPUT_CSV)
desc = df.describe(include='all').transpose()
desc.to_csv("data/artifacts/cell_desc.csv")
