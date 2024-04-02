import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from math import ceil, floor, sqrt

# Plotting colors
colors = [
    "coral",
    "darkcyan",
    "firebrick",
    "goldenrod",
    "darkolivegreen",
    "saddlebrown",
    "plum",
    "khaki",
    "purple",
]

glass_type = {
    1: "building_windows_float_processed",
    2: "building_windows_non_float_processed",
    3: "vehicle_windows_float_processed",
    4: "vehicle_windows_non_float_processed (none in this database)",
    5: "containers",
    6: "tableware",
    7: "headlamps",
}


def plot_pca_3d(pca_trans):
    """
    Visualizes the 3D projection of a dataset using PCA components.

    Parameters:
    - pca_trans: numpy.ndarray
        The transformed dataset after applying PCA.

    The function creates and displays a 3D scatter plot of the first three principal components.
    """
    fig = plt.figure()
    ax_3d = fig.add_subplot(projection="3d")
    ax_3d.scatter(
        pca_trans[:, 0],
        pca_trans[:, 1],
        pca_trans[:, 2],
        marker=".",
        s=2,  # Size of datapoints
    )

    plt.show()


def scatter(data, plot_n, marker_size=2):
    """
    Visualizes the scatter plot of the first n features in the dataset.

    Parameters:
    - data: pandas DataFrame
        The dataset to be visualized.

    - plot_n: int
        The number of features to be visualized in the scatter plot.

    - marker_size: int
        The size of the markers in the scatter plot.

    The function creates and displays a scatter plot for each of the first n features in the dataset.
    """

    n = len(data.columns[0:plot_n])
    ncols = 4
    nrows = 4
    nplots_per_fig = ncols * nrows
    nfigs = ceil(n / nplots_per_fig)  # Total number of figures needed

    for i, feature in enumerate(data.columns[0:plot_n]):
        fig_num = i // nplots_per_fig  # Determine the current figure number
        plot_num = (
            i % nplots_per_fig
        )  # Determine the subplot index within the current figure

        if plot_num == 0:  # Create a new figure if starting a new set of subplots
            fig, axs = plt.subplots(nrows, ncols, figsize=(20, 8))
            axs = axs.flatten()

        data_arr = np.asarray(data[feature])
        axs[plot_num].plot(
            data_arr, "o", markersize=marker_size, color=colors[i % len(colors)]
        )
        axs[plot_num].set_title(feature)
        axs[plot_num].set_xlabel("Feature value")

        plt.tight_layout()

        # Hide any unused subplots in the last figure
        if i == n - 1:  # Check if this is the last plot
            for j in range(plot_num + 1, nplots_per_fig):
                axs[j].set_visible(False)

    plt.show()  # Display all figures at once after creating all of them


def plot_pca_exp_var(pca_var, threshold=0.9):
    """
    Visualizes the variance explained by the principal components.

    Parameters:
    - pca_var: numpy.ndarray
        The variance explained by each principal component.

    - threshold: float

    The function creates and displays a line plot of the variance explained by each principal component,
    as well as the cumulative variance explained by all principal components.

    The threshold parameter is used to display a horizontal line indicating the threshold for cumulative variance.
    """
    fig = plt.figure(figsize=(10, 8), dpi=300)
    ax_var = fig.add_subplot()
    ax_var.plot(range(1, len(pca_var) + 1), pca_var, "o--")
    ax_var.plot(range(1, len(pca_var) + 1), np.cumsum(pca_var), "x--")
    ax_var.axhline(threshold, color="black", linestyle="--")

    # Add the threshold to the y-axis ticks
    new_ticks = np.append(ax_var.get_yticks(), threshold)
    ax_var.set_yticks(new_ticks)
    ax_var.set_ylim(-0.05, 1.05)

    # Set the x-axis tick labels to 'PCA1', 'PCA2', etc.
    pca_labels = ["PC{}".format(i) for i in range(1, len(pca_var) + 1)]
    ax_var.set_xticks(range(1, len(pca_var) + 1))
    # Ensure there's a tick for each label
    ax_var.set_xticklabels(pca_labels)

    plt.title("Variance explained by Principal Components", fontsize=20, weight="bold")
    plt.xlabel("Principal components", fontsize=15)
    plt.ylabel("Fraction of variance explained", fontsize=15)
    plt.legend(["Individual", "Cumulative", "Threshold"])

    plt.grid()
    plt.savefig("pca_exp_var.png")
    plt.show()


def plot_pca_3d(pca_trans):
    """
    Visualizes the 3D projection of a dataset using PCA components.

    Parameters:
    - pca_trans: numpy.ndarray
        The transformed dataset after applying PCA.

    The function creates and displays a 3D scatter plot of the first three principal components.
    """
    fig = plt.figure()
    ax_3d = fig.add_subplot(projection="3d")
    ax_3d.scatter(
        pca_trans[:, 0],
        pca_trans[:, 1],
        pca_trans[:, 2],
        marker=".",
        s=2,  # Size of datapoints
    )

    plt.show()


def histogram_func(df):
    """
    Visualizes the distribution of each attribute in the dataset using histograms.

    Parameters:
    - df: pandas DataFrame
        The dataset to be visualized.

    The function creates and displays a histogram for each attribute in the dataset.
    """
    # Set the aesthetics for the plots
    sns.set_theme(style="whitegrid")

    # Set the color palette for the plots
    pallete = sns.color_palette(n_colors=len(df.columns))

    # Plotting distributions for each attribute to check for outliers and distribution shape
    fig, axs = plt.subplots(3, 3, figsize=(15, 9), dpi=300)

    for i, col in enumerate(df.columns):
        # Calculate row and column index for each subplot
        row = i // 3
        col_idx = i % 3

        # Histogram plotting
        sns.histplot(df[col], kde=True, ax=axs[row, col_idx], color=pallete[i])
        axs[row, col_idx].set_title(f"Distribution of {col}")

    plt.suptitle("Distribution of Glass Attributes", fontsize=20, weight="bold")

    plt.tight_layout()
    plt.savefig("histogram.png")
    plt.show()


def boxplot_function(X):
    """
    Visualizes the distribution of each attribute in the dataset using boxplots. TEST

    Parameters:
    - X: pandas DataFrame
        The dataset to be visualized.

    The function creates and displays a boxplot for each attribute in the dataset.
    """
    plt.figure(figsize=(20, 10))
    sns.set_theme(style="whitegrid", rc={"figure.dpi": 300})

    g = sns.boxplot(X, log_scale=False, orient="h", notch=True)
    g.set_title(
        "Boxplot of Standardized Glass Attributes", fontsize=20, weight="bold", pad=20
    )
    g.set_xlabel("Attributes", fontsize=15)
    g.set_ylabel("Standardized Values", fontsize=15)

    plt.savefig("standardized_boxplot.png")
    plt.show()


def correlation_heatmap(df):
    """
    Visualizes the correlation matrix of the dataset.

    Parameters:
    - df: pandas DataFrame
        The dataset to be visualized.

    The function creates and displays a heatmap of the correlation matrix for the dataset.
    """
    corr = df.corr()

    fig = plt.figure(figsize=(10, 8), dpi=300)
    sns.heatmap(
        corr, annot=True, cmap="coolwarm", cbar_kws={"label": "Correlation Coefficient"}
    )
    plt.title("Correlation matrix", fontsize=20, weight="bold", pad=20)
    plt.xlabel("Attributes")
    plt.ylabel("Attributes")

    plt.savefig("correlation_heatmap.png")
    plt.show()


def loadings_plot(pca, pca_full, features):
    """
    Visualizes the PCA loading plot.
    parameters:
    - pca: PCA object
        The fitted PCA object.
    - pca_full: pandas DataFrame
        The dataset after applying PCA including original RI (for colormap).

    - features: list
        The list of original feature names.

    The function creates and displays a scatter plot of the first two principal components,
    color-coded by the Type column to indicate different categories within the data.

    """
    # Scatter plot of the first two principal components

    # Assuming `pca` is your fitted PCA object and `features` is the list of original feature names
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

    # Normalize the 'RI' values for color mapping
    norm = plt.Normalize(pca_full["RI"].min(), pca_full["RI"].max())
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])

    fig = plt.figure(figsize=(8, 8), dpi=300)

    ### Scatterplot ###
    plt.scatter(
        pca_full["PCA1"],
        pca_full["PCA2"],
        c=pca_full["RI"],
        cmap="viridis",
        norm=norm,
        s=10,
    )

    # Add a color bar to the right of the plots
    cbar = fig.colorbar(
        sm,
        orientation="vertical",
        label="Refractive Index value",
    )

    ### Loadings plot ###
    for i in range(loadings.shape[0]):  # Iterate over the number of features
        # Draw the loading vectors
        plt.quiver(
            0,
            0,
            loadings[i, 0],
            loadings[i, 1],
            angles="xy",
            scale_units="xy",
            scale=0.25,
            # color=colors[i % len(colors)],
            color="darkorange",
            width=0.004,
        )

        # Manual adjustments for the overlapping labels of features at indices 4 and 5
        if i == 4:  # Adjusting the first overlapping feature
            plt.text(
                loadings[i, 0] * 5,
                loadings[i, 1] * 4.5,
                features[i],  # + f" ({colors[i]})",
                ha="right",
                va="center",
                color="red",
            )
        elif i == 5:  # Adjusting the second overlapping feature
            plt.text(
                loadings[i, 0] * 4.1,
                loadings[i, 1] * 7,
                features[i],  # + f" ({colors[i]})",
                ha="center",
                va="bottom",
                color="red",
            )
        else:
            plt.text(
                loadings[i, 0] * 4.5,
                loadings[i, 1] * 4.5,
                features[i],
                ha="center",
                va="center",
                color="red",
            )
    cbar.set_label("Refractive Index Value", size=15)
    plt.title(
        "Biplot of Principal Component 1 and 2", fontsize=20, weight="bold", pad=20
    )
    plt.xlabel("Principal Component 1", fontsize=15)
    plt.ylabel("Principal Component 2", fontsize=15)
    plt.axis("equal")
    plt.grid()

    # Adjust limits if necessary
    plt.xlim(-5, 5)
    plt.ylim(-2, 4)
    plt.savefig("Biplot.png")
    plt.show()


def pairplot(df):
    # Setting up the figure
    fig, axes = plt.subplots(
        df.shape[1] - 1, df.shape[1] - 1, figsize=(15, 20), dpi=300
    )
    # Adjust as needed
    #  plt.subplots_adjust(top=3, right=0.8)  # Adjust space on the right for the colorbar

    # Normalize the 'RI' values for color mapping
    norm = plt.Normalize(df["RI"].min(), df["RI"].max())
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])

    # Iterate over each subplot position
    for i, feature_i in enumerate(
        df.columns[:-1]
    ):  # Assuming last column is 'RI' or 'RI_quartiles'
        for j, feature_j in enumerate(df.columns[:-1]):
            ax = axes[i, j]
            if i == j:
                ax.hist(
                    df[feature_i],
                    bins="auto",
                    color="black",
                    alpha=0.5,
                    edgecolor="black",
                )
            else:
                # Adjust the 's' parameter to make the data point size smaller
                scatter = ax.scatter(
                    df[feature_j],
                    df[feature_i],
                    c=df["RI"],
                    cmap="viridis",
                    norm=norm,
                    s=10,
                )
            ax.set_xlabel(feature_j, size=15)
            ax.set_ylabel(feature_i, size=15)
            ax.label_outer()  # Hide x-ticks and y-ticks for inner plots

            plt.tight_layout()

    # Add a color bar to the right of the plots
    cbar = fig.colorbar(
        sm,
        ax=axes.ravel().tolist(),
        orientation="vertical",
        label="Refractive Index value",
    )
    cbar.set_label("Refractive Index value", size=15)

    # Apply tight layout with padding, then adjust the top margin
    # plt.tight_layout(pad=5.0)
    plt.subplots_adjust(
        top=0.95, right=0.8
    )  # Adjust this value as needed to fit the title

    # Setting a title for the figure
    fig.suptitle(
        "Pairwise Relationships between Glass Attributes", size=20, weight="bold"
    )

    # Setting a title for the figure
    fig.suptitle(
        "Pairwise Relationships between Glass Attributes", size=20, weight="bold"
    )

    plt.savefig("pairplot_grid.png")
    plt.show()


def pca_scatter_2d(X):
    """
    Visualizes the 2D projection of a dataset using PCA components.

    Parameters:
    - X: pandas DataFrame containing PCA1, PCA2, and Type columns.

    The function creates and displays a scatter plot of the first two principal components,
    color-coded by the Type column to indicate different categories within the data.
    """

    fig = plt.figure(figsize=(10, 8), dpi=300)

    norm = plt.Normalize(X["RI"].min(), X["RI"].max())
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])

    plt.scatter(X["PCA1"], X["PCA2"], c=X["RI"], cmap="viridis", norm=norm, s=10)

    # Add a color bar to the right of the plots
    cbar = fig.colorbar(
        sm,
        orientation="vertical",
        label="Refractive Index value",
    )
    cbar.set_label("Refractive Index value", size=15)
    plt.title("PCA of Glass Dataset", fontsize=20, weight="bold", pad=20)
    plt.xlabel("Principal Component 1", fontsize=15)
    plt.ylabel("Principal Component 2", fontsize=15)
    # plt.legend(title="s")
    plt.savefig("pca_scatter_2d.png")
    plt.show()


if __name__ == "__main__":
    print(
        "This is a module with visualization functions. Import it to use the functions."
    )
