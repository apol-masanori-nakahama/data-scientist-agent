import os
os.environ['MPLBACKEND']='Agg'
INPUT_CSV=r'''data/upload.csv'''

import pandas as pd
df = pd.read_csv(INPUT_CSV)
print("shape:", df.shape)
print(df.head(5).to_string())
dtypes = df.dtypes.astype(str).reset_index()
dtypes.columns = ["column","dtype"]
dtypes.to_csv("data/artifacts/cell_dtypes.csv", index=False)
