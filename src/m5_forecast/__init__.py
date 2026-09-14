import pandas as pd

# pandas 3.0 defaults to an Arrow-backed string dtype for all string columns.
# That dtype has already been the source of two confirmed bugs in this project
# (a pandera dtype-check mismatch, and a pd.util.hash_pandas_object crash on
# categorical columns) and is suspected in an intermittent SIGSEGV during CSV
# loading/merging. This project doesn't rely on any Arrow-string feature, so
# reverting to the legacy object dtype sidesteps that whole surface globally.
pd.set_option("future.infer_string", False)
