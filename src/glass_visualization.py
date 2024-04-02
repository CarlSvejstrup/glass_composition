import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from math import ceil, floor, sqrt
import matplotlib.pyplot as plt
import seaborn as sns
from visualization_plots import *
from data_loader import *



#### Binarize RI for classification####
# # Calculate median of RI
# RI_median = np.median(df["RI"], axis=0)
# # Create mask for RI to binarize the data for classification
# mask = df["RI"] > RI_median

# df["RI"] = np.where(mask, 1, 0)


# Check for dubplicates
# print(df.duplicated().sum())
# Remove duplicates
# df.drop_duplicates(inplace=True)

# Check for Nan and null values
# print(df.isna().sum(), df.isnull().sum())

### Summary statistics ###
summary_statistics = df.describe()
# print(summary_statistics)
# print(summary_statistics.to_latex(float_format="%.2f", bold_rows = True, caption="Summary statistics for the glass dataset", label="tab:summary_statistics"))


### Standadizing ###
standardize = StandardScaler()
df_standard = standardize.fit_transform(df)
df_standard = pd.DataFrame(df_standard, columns=df.columns)

#### PCA ####
pca = PCA(n_components=2)
df_pca = pca.fit_transform(df_standard)
df_pca = pd.DataFrame(
    df_pca,
    columns=["PCA1", "PCA2"],
)
df_pca_full = pd.concat([df_pca, df_RI], axis=1)

pca_all = PCA()
pca_all.fit(df_standard)
explained_variance_ratio_all = pca_all.explained_variance_ratio_

print(explained_variance_ratio_all)

####  Plots form 'visualization_plots.py' #####
### Boxplot ###
# boxplot_func(df)

### Histogram ###
# histogram_func(df)

### Correlation heatmap ###
# correlation_heatmap(df)

### Loadings plot ###
# loadings_plot(pca, df_pca, attribute_names[1:-1], df_type)

# pca_scatter_2d(df_pca_full)
# scatter(df, len(df.columns)
# plot_pca_exp_var(explained_variance_ratio, threshold=0.9):
# plot_pca_3d(X_pca)
# boxplot_func(df)
# histogram_func(df)
# correlation_heatmap(df)
# loadings_plot(pca, df_pca, attribute_names[1:-1], df_type)

# plot_pca_exp_var(explained_variance_ratio_all, threshold=0.9)

# # Number of principal components
# pcs = [0, 1]
# legendStrs = ["PC" + str(e + 1) for e in pcs]
# # colors = ["r", "g", "b"]
# bw = 0.5
# # Attributes excluding 'Id' and 'Type'
# attribute_names_corrected = attribute_names[1:-1]
# r = np.arange(len(attribute_names_corrected))

# # Obtain the PCA component coefficients/loadings
# component_coefficients = pca.components_

# plt.figure(figsize=(12, 7), dpi=300)

# for i, pc in enumerate(pcs):
#     plt.bar(
#         r + (i + 0.5) * bw,
#         component_coefficients[pc, :],
#         width=bw,
#         label=legendStrs[i],
#         # color=colors[i],
#     )

# plt.xticks(r + bw, attribute_names_corrected, rotation=45)
# plt.xlabel("Attributes", fontsize=15)
# plt.ylabel("PCA Component Coefficients", fontsize=15)
# plt.legend()
# plt.grid(True)
# plt.title("PCA Component Coefficients", fontsize=20, fontweight="bold")
# plt.tight_layout()
# plt.savefig("PCA_component_coefficients.png")
# plt.show()
boxplot_function(df_standard)
