import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# All atributes exept Type (nomial) are continuos ratio.
attribute_names = ["Id", "RI", "Na", "Mg", "Al", "Si", "K ", "Ca", "Ba", "Fe", "Type"]
filename = "./data/glass.data"

glass_type = {
    1: "building_windows_float_processed",
    2: "building_windows_non_float_processed",
    3: "vehicle_windows_float_processed",
    4: "vehicle_windows_non_float_processed (none in this database)",
    5: "containers",
    6: "tableware",
    7: "headlamps",
}

### Read data ###
df = pd.read_csv(filename, names=attribute_names)
df_RI = df["RI"]
df_type = df["Type"]
# Storing type in different DF

attribute_names = ["Na", "Mg", "Al", "Si", "K ", "Ca", "Ba", "Fe"]

# Revoming Type and Id
df.drop(["Id", "Type"], axis=1, inplace=True)

if __name__ == "__main__":
    print(df["RI"].describe())
