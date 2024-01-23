import matplotlib.pyplot as plt

import mpl_toolkits.mplot3d
from matplotlib import ticker

from sklearn import datasets, manifold


# Simple test dataset
n_samples = 1500
S_points, S_color = datasets.make_s_curve(n_samples, random_state=0)


# Plotting functions, taken from somewhere?
def plot_3d(points, points_color, title):
    x, y, z = points.T

    fig, ax = plt.subplots(
        figsize=(6, 6),
        facecolor="white",
        tight_layout=True,
        subplot_kw={"projection": "3d"},
    )
    fig.suptitle(title, size=16)
    col = ax.scatter(x, y, z, c=points_color, s=50, alpha=0.8)
    ax.view_init(azim=-60, elev=9)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.zaxis.set_major_locator(ticker.MultipleLocator(1))

    fig.colorbar(col, ax=ax, orientation="horizontal", shrink=0.6, aspect=60, pad=0.01)
    plt.show()


def plot_2d(points, points_color, title):
    fig, ax = plt.subplots(figsize=(3, 3), facecolor="white", constrained_layout=True)
    fig.suptitle(title, size=16)
    add_2d_scatter(ax, points, points_color)
    plt.show()


def add_2d_scatter(ax, points, points_color, title=None):
    x, y = points.T
    ax.scatter(x, y, c=points_color, s=50, alpha=0.8)
    ax.set_title(title)
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.yaxis.set_major_formatter(ticker.NullFormatter())


print("here")
plot_3d(S_points, S_color, "Original S-curve samples")

#################################################################
#      A bunch of standard LLE Approaches and some notes        #
###############################################################

n_neighbors = 12  # neighbourhood to connect each point to
n_components = 2  # manifold dimensionality

params = {
    "n_neighbors": n_neighbors,
    "n_components": n_components,
    "eigen_solver": "auto",
    "random_state": 0,
}

# Simple Local Linear Embedding, see here:
# An Introduction to Locally Linear Embedding Saul, L. and Roweis, S.
#
# Notes: Took about ~5min for 2*10^6 points, 10 dimensions
lle_standard = manifold.LocallyLinearEmbedding(method="standard", **params)
S_standard = lle_standard.fit_transform(S_points)

# Hesseian Local Linear Embedding, see:
# “Hessian Eigenmaps: Locally linear embedding techniques for high-dimensional data” Donoho, D. & Grimes, C. 
# 
# Revolves around a hessian-based quadratic form at each neighborhood which is used to recover the locally linear structure
# Has nice convergence properties.
# Notes: Very slow for 10^6 points beyond 4-5 dimensions
lle_hessian = manifold.LocallyLinearEmbedding(method="hessian", **params)
S_hessian = lle_hessian.fit_transform(S_points)


# Characterizes the local geometry at each neighborhood via its tangent space, 
# and performs a global optimization to align these local tangent spaces to learn 
# the embedding. 
# 
# Notes: Ran in very similar time to LLE
lle_ltsa = manifold.LocallyLinearEmbedding(method="ltsa", **params)
S_ltsa = lle_ltsa.fit_transform(S_points)


# Modified LLE. Seems to be useful when the number of neighbours is greater than the number 
# of input dimensions (since the matrix defining each local neighborhood is rank-deficient).
# Don't understand it really, runs with about the same efficiency as LLE

#  “MLLE: Modified Locally Linear Embedding Using Multiple Weights” Zhang, Z. & Wang, J.
lle_mod = manifold.LocallyLinearEmbedding(method="modified", **params)
S_mod = lle_mod.fit_transform(S_points)


# Plots the LLE stuff
fig, axs = plt.subplots(
    nrows=2, ncols=2, figsize=(7, 7), facecolor="white", constrained_layout=True
)
fig.suptitle("Locally Linear Embeddings", size=16)

lle_methods = [
    ("Standard locally linear embedding", S_standard),
    ("Local tangent space alignment", S_ltsa),
    ("Hessian eigenmap", S_hessian),
    ("Modified locally linear embedding", S_mod),
]
for ax, method in zip(axs.flat, lle_methods):
    name, points = method
    add_2d_scatter(ax, points, S_color, name)

plt.show()

# Basic ISOMAP implementation. Fastest approach?
isomap = manifold.Isomap(n_neighbors=n_neighbors, n_components=n_components, p=1)
S_isomap = isomap.fit_transform(S_points)

plot_2d(S_isomap, S_color, "Isomap Embedding")


# Basic spectral implementation. Faster than everything that isn't a Hessian
spectral = manifold.SpectralEmbedding(
    n_components=n_components, n_neighbors=n_neighbors, random_state=42
)
S_spectral = spectral.fit_transform(S_points)

plot_2d(S_spectral, S_color, "Spectral Embedding")