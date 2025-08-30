import os
os.environ['MPLBACKEND']='Agg'
try:
    import resource
    resource.setrlimit(resource.RLIMIT_CPU, (30, 30))
    resource.setrlimit(resource.RLIMIT_AS, (1600000000, 1600000000))
except Exception:
    pass
INPUT_CSV=r'''/private/var/folders/vp/95tr04gs5k310fwyfrtxxqkm0000gn/T/pytest-of-nhama/pytest-28/test_cli_smoke0/sample.csv'''

import pandas as pd
df = pd.read_csv(INPUT_CSV)
print("shape:", df.shape)
print(df.head(5).to_string())
dtypes = df.dtypes.astype(str).reset_index()
dtypes.columns = ["column","dtype"]
dtypes.to_csv("data/artifacts/cell_dtypes.csv", index=False)
