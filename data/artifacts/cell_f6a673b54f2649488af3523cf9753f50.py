import os
os.environ['MPLBACKEND']='Agg'
try:
    import resource
    resource.setrlimit(resource.RLIMIT_CPU, (30, 30))
    resource.setrlimit(resource.RLIMIT_AS, (1600000000, 1600000000))
except Exception:
    pass
INPUT_CSV=r'''data/upload.csv'''

import pandas as pd, matplotlib.pyplot as plt
df = pd.read_csv(INPUT_CSV)
num = df.select_dtypes(include="number").columns[:5]
for c in num:
    ax = df[c].dropna().hist(bins=30)
    ax.figure.suptitle(f"hist_{c}")
    ax.figure.savefig(f"data/artifacts/cell_hist_{c}.png")
    plt.close(ax.figure)
