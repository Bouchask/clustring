# Clustering Analysis Project: Mathematical Framework for Optimal K Selection

**Author:** Yahya Bouchak  
**Level:** Master – SIIA  
**Field:** Unsupervised Machine Learning  
**Date:** January 2026

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Problem Definition](#2-problem-definition)
3. [Preprocessing Pipeline](#3-preprocessing-pipeline)
4. [Clustering Algorithms](#4-clustering-algorithms)
5. [Mathematical Metrics for Optimal K Selection](#5-mathematical-metrics-for-optimal-k-selection)
6. [Mathematical Relationships Between Metrics](#6-mathematical-relationships-between-metrics)
7. [Optimal K Selection Methodology](#7-optimal-k-selection-methodology)
8. [Decision Algorithm and Consensus Strategy](#8-decision-algorithm-and-consensus-strategy)
9. [Implementation Results](#9-implementation-results)
10. [Conclusion](#10-conclusion)

---

## 1. Introduction

This project addresses an **unsupervised clustering problem** with the primary objective of determining the **optimal number of clusters (K)** using **rigorous mathematical, statistical, and probabilistic criteria**.

### Key Principles

The methodology adopted in this work is based on:

1. **Objectivity**: Mathematical decision rules replace subjective visual inspection
2. **Multi-perspective validation**: Multiple metrics from different mathematical frameworks
3. **Consensus-based selection**: Democratic voting mechanism across metrics
4. **Academic rigor**: Compliance with Master-level statistical and mathematical standards

### Why Multiple Metrics?

The optimal number of clusters is not a universally defined concept. Different metrics capture different aspects of cluster quality:

- **Geometric perspective**: Cluster compactness and separation in Euclidean space
- **Statistical perspective**: Variance decomposition and ratio analysis
- **Probabilistic perspective**: Likelihood-based model selection with complexity penalties

By combining these perspectives, we achieve a **robust and defensible** selection of K.

---

## 2. Problem Definition

### 2.1 Mathematical Formulation

Given a dataset:

$$X = \{x_1, x_2, \dots, x_n\}, \quad x_i \in \mathbb{R}^d$$

where:
- $n$ is the number of observations
- $d$ is the dimensionality of the feature space

The goal of clustering is to find a partition:

$$\mathcal{C} = \{C_1, C_2, \dots, C_K\}$$

Such that:

$$\bigcup_{k=1}^{K} C_k = X \quad \text{and} \quad C_i \cap C_j = \emptyset \; \forall i \neq j$$

### 2.2 Optimization Objectives

The partition should satisfy two competing objectives:

**1. Intra-cluster cohesion** (minimization):

$$\min \sum_{k=1}^{K} \sum_{x_i \in C_k} \lVert x_i - \mu_k \rVert^2$$

**Interpretation**: Minimize the total squared distance of all points from their respective cluster centroids. This ensures that points within the same cluster are **as close together as possible** (compact clusters).

**2. Inter-cluster separation** (maximization):

$$\max \sum_{k=1}^{K} |C_k| \lVert \mu_k - \mu \rVert^2$$

**Interpretation**: Maximize the weighted squared distance of cluster centroids from the global centroid. This ensures that different clusters are **as far apart as possible** (well-separated clusters).

where:
- $\mu_k = \frac{1}{|C_k|} \sum_{x_i \in C_k} x_i$ is the centroid of cluster $k$
- $\mu = \frac{1}{n} \sum_{i=1}^{n} x_i$ is the global centroid (mean of all data points)
- $|C_k|$ is the number of points in cluster $k$ (weighting factor)

**Trade-off**: These two objectives are inherently competing:
- Creating more clusters (larger $K$) improves cohesion but may reduce meaningful separation
- Creating fewer clusters (smaller $K$) may improve separation but reduces cohesion
- The optimal $K$ balances both objectives

### 2.3 The K Selection Problem

The fundamental challenge is that:

- **As K increases**: Intra-cluster cohesion improves (smaller, tighter clusters)
- **As K increases**: Model complexity increases (risk of overfitting)
- **Extreme case K = n**: Each point is its own cluster (WCSS = 0, but meaningless)

Therefore, we need **principled stopping criteria** to identify the optimal K that balances:
- Cluster quality
- Model parsimony
- Statistical validity

---

## 3. Preprocessing Pipeline

### 3.1 Data Cleaning

The preprocessing stage ensures data quality through:

1. **Missing value removal**: Eliminates incomplete observations
   ```python
   df = df.dropna()
   ```

2. **Duplicate elimination**: Ensures unique observations
   ```python
   df = df.drop_duplicates()
   ```

### 3.2 Feature Engineering

Temporal features are transformed into quantitative representations:

```python
df["Duration_Sec"] = (df["Charging End Time"] - df["Charging Start Time"]).dt.total_seconds()
```

This captures the duration information as a continuous variable suitable for distance-based clustering.

### 3.3 Categorical Encoding

Two encoding strategies are applied based on cardinality:

**1. Binary variables** (cardinality ≤ 2): **Label Encoding**

$$x_{\text{encoded}} \in \{0, 1\}$$

**Example**: Gender → {Male: 0, Female: 1}

**Rationale**: For binary variables, simple numerical encoding preserves the information without introducing unnecessary dimensions.

**2. Multi-class variables** (cardinality > 2): **One-Hot Encoding**

$$x \in \{c_1, \dots, c_m\} \rightarrow \mathbf{x} \in \{0,1\}^m$$

**Example**: Vehicle Type → {Sedan, SUV, Truck} becomes:
- Sedan: [1, 0, 0]
- SUV: [0, 1, 0]
- Truck: [0, 0, 1]

**Rationale**: One-hot encoding prevents the algorithm from assuming ordinal relationships between categories (e.g., Sedan < SUV < Truck), which would be inappropriate for nominal variables.

### 3.4 Feature Scaling

Standardization ensures all features contribute equally to distance calculations:

$$z_j = \frac{x_j - \mu_j}{\sigma_j}$$

where:
- $\mu_j$ is the mean of feature $j$
- $\sigma_j$ is the standard deviation of feature $j$

**Why standardization is critical**: Distance-based algorithms (K-Means, GMM with full covariance) are sensitive to feature scales. Without standardization, features with larger ranges dominate the distance metric.

### 3.5 Dimensionality Reduction (Visualization Only)

Principal Component Analysis (PCA) is applied to reduce dimensionality to 2D:

$$X_{\text{PCA}} = X \cdot W$$

where $W \in \mathbb{R}^{d \times 2}$ contains the top 2 principal components.

**CRITICAL NOTE**: PCA is used **exclusively for visualization**. All clustering algorithms and metric computations operate on the **full-dimensional standardized data** $X_{\text{scaled}}$, not on $X_{\text{PCA}}$.

**Justification**: 
- PCA may discard variance that is relevant for clustering
- Metrics computed on reduced data may not reflect true cluster quality
- Visualization in 2D helps interpret results but should not drive the selection process

---

## 4. Clustering Algorithms

### 4.1 K-Means Clustering

K-Means is a **centroid-based partitioning algorithm** that assigns each point to the nearest cluster center.

#### 4.1.1 Mathematical Objective

K-Means minimizes the **Within-Cluster Sum of Squares (WCSS)**:

$$\text{WCSS}(K) = \sum_{k=1}^{K} \sum_{x_i \in C_k} \lVert x_i - \mu_k \rVert^2$$

where:
- $K$ is the number of clusters
- $C_k$ is the set of points in cluster $k$
- $\mu_k = \frac{1}{|C_k|} \sum_{x_i \in C_k} x_i$ is the centroid of cluster $k$
- $\lVert \cdot \rVert^2$ denotes squared Euclidean distance

#### 4.1.2 Algorithm (Lloyd's Algorithm)

The K-Means algorithm iteratively refines cluster assignments using the following steps:

**Step 1: Initialize**

Randomly select $K$ initial centroids:

$$\mu_1, \mu_2, \dots, \mu_K \in \mathbb{R}^d$$

**Common initialization methods**:
- Random selection from data points
- K-Means++ (smart initialization for faster convergence)
- Multiple random initializations (select best result)

**Step 2: Assignment Step**

Assign each point $x_i$ to the nearest centroid based on Euclidean distance:

$$C_k = \{x_i : \lVert x_i - \mu_k \rVert \leq \lVert x_i - \mu_j \rVert, \; \forall j \in \{1, \dots, K\}\}$$

**Interpretation**: Each cluster $C_k$ contains all points for which centroid $\mu_k$ is the closest among all $K$ centroids. This creates a **Voronoi partition** of the feature space.

**Step 3: Update Step**

Recalculate each centroid as the mean of all points assigned to it:

$$\mu_k = \frac{1}{|C_k|} \sum_{x_i \in C_k} x_i$$

**Interpretation**: The new centroid is positioned at the center of mass of all points in cluster $C_k$. This minimizes the within-cluster sum of squared distances.

**Step 4: Convergence**

Repeat Steps 2 and 3 until one of the following conditions is met:
- Centroids no longer change: $\mu_k^{(t+1)} = \mu_k^{(t)}$ for all $k$
- Assignments no longer change: $C_k^{(t+1)} = C_k^{(t)}$ for all $k$
- Maximum number of iterations reached
- Change in WCSS falls below a threshold: $|\text{WCSS}^{(t+1)} - \text{WCSS}^{(t)}| < \epsilon$

**Convergence guarantee**: Lloyd's algorithm is guaranteed to converge to a local minimum of the WCSS objective function, though not necessarily the global minimum.

#### 4.1.3 Properties

- **Complexity**: $O(nKdi)$ where $i$ is the number of iterations
- **Assumptions**: Spherical clusters with similar sizes and densities
- **Limitations**: Sensitive to initialization, assumes convex clusters

### 4.2 Gaussian Mixture Models (GMM)

GMM provides a **probabilistic soft-clustering** approach by modeling the data as a mixture of Gaussian distributions.

#### 4.2.1 Probabilistic Model

The likelihood of observing data point $x$ is:

$$p(x \mid \Theta) = \sum_{k=1}^{K} \pi_k \mathcal{N}(x \mid \mu_k, \Sigma_k)$$

where:
- $\pi_k$ is the mixing coefficient (prior probability of cluster $k$), with $\sum_{k=1}^{K} \pi_k = 1$
- $\mathcal{N}(x \mid \mu_k, \Sigma_k)$ is a multivariate Gaussian distribution
- $\Theta = \{\pi_1, \dots, \pi_K, \mu_1, \dots, \mu_K, \Sigma_1, \dots, \Sigma_K\}$ are the model parameters

The Gaussian density is:

$$\mathcal{N}(x \mid \mu, \Sigma) = \frac{1}{(2\pi)^{d/2} |\Sigma|^{1/2}} \exp\left(-\frac{1}{2}(x - \mu)^T \Sigma^{-1} (x - \mu)\right)$$

#### 4.2.2 Parameter Estimation (EM Algorithm)

The **Expectation-Maximization (EM)** algorithm estimates $\Theta$ by maximizing the log-likelihood:

$$\mathcal{L}(\Theta) = \sum_{i=1}^{n} \log p(x_i \mid \Theta) = \sum_{i=1}^{n} \log \sum_{k=1}^{K} \pi_k \mathcal{N}(x_i \mid \mu_k, \Sigma_k)$$

**E-step**: Compute posterior probabilities (responsibilities):
$$\gamma_{ik} = \frac{\pi_k \mathcal{N}(x_i \mid \mu_k, \Sigma_k)}{\sum_{j=1}^{K} \pi_j \mathcal{N}(x_i \mid \mu_j, \Sigma_j)}$$

**M-step**: Update parameters:
$$N_k = \sum_{i=1}^{n} \gamma_{ik}$$
$$\mu_k = \frac{1}{N_k} \sum_{i=1}^{n} \gamma_{ik} x_i$$
$$\Sigma_k = \frac{1}{N_k} \sum_{i=1}^{n} \gamma_{ik} (x_i - \mu_k)(x_i - \mu_k)^T$$
$$\pi_k = \frac{N_k}{n}$$

#### 4.2.3 Advantages Over K-Means

1. **Soft assignments**: Points can belong to multiple clusters with probabilities
2. **Flexible cluster shapes**: Covariance matrices $\Sigma_k$ allow elliptical clusters
3. **Principled model selection**: BIC provides a likelihood-based criterion for K

---

## 5. Mathematical Metrics for Optimal K Selection

This section provides the **complete mathematical foundation** for each metric, including derivations, interpretations, and optimization criteria.

### 5.1 Silhouette Score

The Silhouette Score measures how well each point fits within its assigned cluster relative to other clusters.

#### 5.1.1 Mathematical Definition

For each data point $x_i$ assigned to cluster $C_k$, the Silhouette coefficient is computed using the following components:

**1. Intra-cluster distance (cohesion)**:

$$a(i) = \frac{1}{|C_k| - 1} \sum_{x_j \in C_k, j \neq i} d(x_i, x_j)$$

**Interpretation**: $a(i)$ measures the average distance from point $x_i$ to all other points in the same cluster $C_k$. 
- **Lower values** indicate that $x_i$ is well-clustered (close to its neighbors)
- **Higher values** suggest $x_i$ is far from other points in its cluster
- The denominator is $|C_k| - 1$ because we exclude the point itself

**2. Nearest-cluster distance (separation)**:

$$b(i) = \min_{C_l \neq C_k} \frac{1}{|C_l|} \sum_{x_j \in C_l} d(x_i, x_j)$$

**Interpretation**: $b(i)$ measures the minimum average distance from point $x_i$ to points in other clusters.
- Computed by averaging distances to all points in each neighboring cluster
- Taking the minimum identifies the "nearest" neighboring cluster
- **Higher values** indicate that $x_i$ is well-separated from other clusters
- **Lower values** suggest $x_i$ is close to a neighboring cluster (possibly misclassified)

**3. Silhouette coefficient for point $i$**:

$$s(i) = \frac{b(i) - a(i)}{\max\{a(i), b(i)\}}$$

**Interpretation**: $s(i)$ combines cohesion and separation into a single metric:
- **Numerator** $b(i) - a(i)$: Difference between separation and cohesion
  - Positive when $b(i) > a(i)$ (good: point is closer to its own cluster)
  - Negative when $a(i) > b(i)$ (bad: point is closer to another cluster)
- **Denominator** $\max\{a(i), b(i)\}$: Normalization factor
  - Ensures $s(i) \in [-1, 1]$

**Value interpretation**:
- $s(i) \approx 1$: Point is very well-matched to its cluster (ideal)
- $s(i) \approx 0$: Point is on the boundary between clusters (ambiguous)
- $s(i) \approx -1$: Point is likely misclassified (should be in another cluster)

**4. Average Silhouette Score for the entire dataset**:

$$S(K) = \frac{1}{n} \sum_{i=1}^{n} s(i)$$

**Interpretation**: $S(K)$ is the mean silhouette coefficient across all $n$ data points.
- Aggregates individual point quality into a global cluster quality measure
- **Higher values** indicate better overall clustering quality
- Provides a single number to compare different values of $K$

#### 5.1.2 Interpretation

- $s(i) \approx 1$: Point is well-matched to its cluster and far from others
- $s(i) \approx 0$: Point is on the boundary between clusters
- $s(i) \approx -1$: Point is likely misclassified

#### 5.1.3 Optimization Criterion

$$K_{\text{Silhouette}}^* = \arg\max_{K \in \{2, \dots, K_{\max}\}} S(K)$$

**Decision rule**: Select the K that maximizes the average silhouette score.

#### 5.1.4 Advantages and Limitations

**Advantages**:
- Intuitive geometric interpretation
- No assumptions about cluster shape
- Considers both cohesion and separation

**Limitations**:
- Computationally expensive for large datasets: $O(n^2)$
- Sensitive to outliers
- May favor compact, spherical clusters

### 5.2 Calinski-Harabasz Index (Variance Ratio Criterion)

The CH index measures the ratio of **between-cluster variance** to **within-cluster variance**.

#### 5.2.1 Mathematical Formulation

The total sum of squares (TSS) can be decomposed into between-cluster and within-cluster components:

$$\text{TSS} = \text{BSS} + \text{WSS}$$

**where the following components are defined:**

**1. Total Sum of Squares (TSS)**:

$$\text{TSS} = \sum_{i=1}^{n} \lVert x_i - \mu \rVert^2$$

**Interpretation**: Total variance in the dataset, measuring the squared distance of all points from the global centroid $\mu = \frac{1}{n}\sum_{i=1}^{n} x_i$.

**2. Between-cluster Sum of Squares (BSS)**:

$$\text{BSS} = \sum_{k=1}^{K} |C_k| \lVert \mu_k - \mu \rVert^2$$

**Interpretation**: Variance between cluster centroids, weighted by cluster sizes. This measures how spread out the clusters are from the global centroid. Higher values indicate better-separated clusters.

where:
- $|C_k|$ is the number of points in cluster $k$
- $\mu_k$ is the centroid of cluster $k$
- $\mu$ is the global centroid of all data

**3. Within-cluster Sum of Squares (WSS)**:

$$\text{WSS} = \sum_{k=1}^{K} \sum_{x_i \in C_k} \lVert x_i - \mu_k \rVert^2$$

**Interpretation**: Total variance within all clusters, measuring how tightly points are grouped around their respective centroids. Lower values indicate more compact clusters. This is identical to the K-Means objective function WCSS.

**Variance Decomposition Property**:

The fundamental relationship $\text{TSS} = \text{BSS} + \text{WSS}$ shows that:
- Total variance is constant (independent of clustering)
- Increasing BSS (better separation) automatically decreases WSS (better compactness)
- The goal is to maximize the proportion of variance explained by cluster structure

**4. The Calinski-Harabasz Index**:

$$\text{CH}(K) = \frac{\text{BSS}/(K-1)}{\text{WSS}/(n-K)} = \frac{n-K}{K-1} \cdot \frac{\text{BSS}}{\text{WSS}}$$

**Components explained**:
- **Numerator** $\frac{\text{BSS}}{K-1}$: Average between-cluster variance (degrees of freedom = $K-1$)
- **Denominator** $\frac{\text{WSS}}{n-K}$: Average within-cluster variance (degrees of freedom = $n-K$)
- **Ratio**: Higher values indicate better-defined clusters (high separation, low dispersion)

#### 5.2.2 Statistical Interpretation

The CH index is analogous to the **F-statistic** in ANOVA:

- Numerator: Average between-cluster variance (signal)
- Denominator: Average within-cluster variance (noise)
- Higher values indicate better-defined clusters

#### 5.2.3 Optimization Criterion

$$K_{\text{CH}}^* = \arg\max_{K \in \{2, \dots, K_{\max}\}} \text{CH}(K)$$

**Decision rule**: Select the K that maximizes the variance ratio.

#### 5.2.4 Behavior Analysis

As $K$ increases:
- **WSS decreases** monotonically (smaller clusters are tighter)
- **BSS increases** monotonically (more centroids spread across space)
- **CH(K)** typically has a peak at intermediate K values

The peak represents the optimal trade-off between:
- Too few clusters: Large WSS, small BSS
- Too many clusters: Diminishing returns in WSS reduction

#### 5.2.5 Advantages and Limitations

**Advantages**:
- Fast computation: $O(nKd)$
- Based on well-established statistical principles
- Works well for convex, well-separated clusters

**Limitations**:
- Assumes clusters have similar sizes
- Biased toward spherical clusters
- May not detect optimal K for complex cluster shapes

### 5.3 Davies-Bouldin Index

The DB index measures the average **similarity** between each cluster and its most similar cluster.

#### 5.3.1 Mathematical Definition

For each cluster $C_i$, define the following components:

**1. Intra-cluster scatter** (average distance to centroid):

$$S_i = \frac{1}{|C_i|} \sum_{x \in C_i} \lVert x - \mu_i \rVert$$

where $\mu_i$ is the centroid of cluster $C_i$ and $\lVert \cdot \rVert$ denotes the Euclidean norm.

**Interpretation**: $S_i$ measures the average dispersion of points within cluster $i$. Smaller values indicate tighter, more compact clusters.

**2. Inter-cluster distance** (distance between centroids):

$$M_{ij} = \lVert \mu_i - \mu_j \rVert$$

where $\mu_i$ and $\mu_j$ are the centroids of clusters $i$ and $j$ respectively.

**Interpretation**: $M_{ij}$ measures the separation between cluster centroids. Larger values indicate better-separated clusters.

**3. Similarity measure** between clusters $i$ and $j$:

$$R_{ij} = \frac{S_i + S_j}{M_{ij}}$$

**Interpretation**: This ratio combines intra-cluster compactness and inter-cluster separation:
- **Numerator** $(S_i + S_j)$: Sum of internal dispersions (lower is better)
- **Denominator** $M_{ij}$: Centroid separation (higher is better)
- **High** $R_{ij}$ values occur when:
  - Clusters are close together (small $M_{ij}$), OR
  - Clusters are internally dispersed (large $S_i$ or $S_j$)
- **Low** $R_{ij}$ values indicate well-separated, compact cluster pairs (desirable)

**4. Davies-Bouldin Index**:

$$\text{DB}(K) = \frac{1}{K} \sum_{i=1}^{K} \max_{j \neq i} R_{ij}$$

**Step-by-step computation**:
1. For each cluster $i$, compute $R_{ij}$ for all other clusters $j \neq i$
2. Select the maximum: $\max_{j \neq i} R_{ij}$ (the "worst case" similarity)
3. Average these maximum similarities across all $K$ clusters

**Interpretation**: 
- DB measures the average "worst-case" similarity across all clusters
- Each cluster is compared to its most similar neighbor
- Lower DB values indicate better clustering:
  - Small intra-cluster scatter ($S_i$ small)
  - Large inter-cluster separation ($M_{ij}$ large)
  - Low similarity ratios ($R_{ij}$ small)

#### 5.3.2 Interpretation

- For each cluster $i$, find the most similar cluster $j^* = \arg\max_{j \neq i} R_{ij}$
- Average these maximum similarities across all clusters
- Lower DB values indicate better clustering:
  - Small intra-cluster scatter ($S_i$ small)
  - Large inter-cluster separation ($M_{ij}$ large)

#### 5.3.3 Optimization Criterion

$$K_{\text{DB}}^* = \arg\min_{K \in \{2, \dots, K_{\max}\}} \text{DB}(K)$$

**Decision rule**: Select the K that minimizes the Davies-Bouldin index.

#### 5.3.4 Advantages and Limitations

**Advantages**:
- Intuitive interpretation: measures cluster separation and compactness
- No assumptions about cluster distribution
- Fast computation: $O(K^2 d)$

**Limitations**:
- Sensitive to centroid-based cluster representations
- May not work well for non-convex clusters
- Can be influenced by outliers

### 5.4 Bayesian Information Criterion (BIC)

BIC is a **model selection criterion** based on likelihood theory with a penalty for model complexity.

#### 5.4.1 Mathematical Formulation

For a probabilistic model with parameters $\Theta$:

$$\text{BIC}(K) = -2 \log \mathcal{L}(\Theta) + p \log(n)$$

where:
- $\mathcal{L}(\Theta) = \prod_{i=1}^{n} p(x_i \mid \Theta)$ is the likelihood of the data
- $p$ is the number of free parameters in the model
- $n$ is the number of observations

For Gaussian Mixture Models:

$$\text{BIC}(K) = -2 \sum_{i=1}^{n} \log p(x_i \mid \Theta) + p(K) \log(n)$$

where the number of parameters is:

$$p(K) = K \cdot d + K \cdot \frac{d(d+1)}{2} + (K-1)$$

This includes:
- $K \cdot d$ mean parameters
- $K \cdot \frac{d(d+1)}{2}$ covariance parameters (for full covariance matrices)
- $K - 1$ mixing coefficients (one is redundant due to $\sum \pi_k = 1$)

#### 5.4.2 Interpretation of Terms

1. **Likelihood term** $-2 \log \mathcal{L}(\Theta)$:
   - Measures how well the model fits the data
   - Lower values = better fit
   - Decreases monotonically with K

2. **Penalty term** $p \log(n)$:
   - Penalizes model complexity
   - Prevents overfitting
   - Increases with K

#### 5.4.3 Optimization Criterion

$$K_{\text{BIC}}^* = \arg\min_{K \in \{2, \dots, K_{\max}\}} \text{BIC}(K)$$

**Decision rule**: Select the K that minimizes BIC, representing the optimal balance between fit and complexity.

#### 5.4.4 Bayesian Foundation

BIC approximates the **log Bayes factor** for model comparison:

$$\text{BIC}(K) \approx -2 \log p(X \mid M_K)$$

where $p(X \mid M_K)$ is the marginal likelihood (evidence) for model $M_K$.

Under certain regularity conditions (large $n$), BIC provides an **asymptotically consistent** estimator of the true model.

#### 5.4.5 Advantages and Limitations

**Advantages**:
- Theoretically grounded in Bayesian model selection
- Penalizes overfitting
- Works well for GMM-based clustering

**Limitations**:
- Assumes the model family (GMM) is appropriate
- Penalty may be too strong for small datasets
- Requires likelihood computation (specific to probabilistic models)

---

## 6. Mathematical Relationships Between Metrics

Understanding how these metrics relate to each other is crucial for interpreting their collective behavior.

### 6.1 Metric Taxonomy

| Metric | Mathematical Framework | Optimization | Complexity Penalty | Cluster Assumptions |
|--------|------------------------|--------------|-------------------|---------------------|
| **Silhouette** | Geometric (distance-based) | Maximize | Implicit (through max) | None (distance-only) |
| **Calinski-Harabasz** | Statistical (variance decomposition) | Maximize | Implicit (degrees of freedom) | Spherical, similar sizes |
| **Davies-Bouldin** | Geometric (centroid similarity) | Minimize | None | Centroid-based |
| **BIC** | Probabilistic (likelihood-based) | Minimize | Explicit ($p \log n$) | Gaussian distributions |

### 6.2 Shared Principles

Despite different mathematical frameworks, all metrics share common principles:

1. **Compactness preference**: All favor tight clusters
   - Silhouette: Minimize $a(i)$
   - CH: Minimize WSS
   - DB: Minimize $S_i$
   - BIC: Maximize likelihood (tighter clusters = higher probability)

2. **Separation preference**: All favor well-separated clusters
   - Silhouette: Maximize $b(i)$
   - CH: Maximize BSS
   - DB: Maximize $M_{ij}$
   - BIC: Distinct Gaussians have lower overlap

3. **Complexity aversion**: All resist overfitting, but differ in mechanism
   - Silhouette: Boundary points penalize fine-grained splits
   - CH: Degrees of freedom adjustment ($K-1$, $n-K$)
   - DB: No explicit penalty (can favor higher K)
   - BIC: Explicit logarithmic penalty

### 6.3 Why Metrics May Disagree

Different metrics may suggest different optimal K values due to:

1. **Different sensitivity to cluster properties**:
   - **CH** is more sensitive to cluster size imbalance
   - **Silhouette** is more sensitive to overlapping clusters
   - **DB** is more sensitive to centroid positions
   - **BIC** is more sensitive to Gaussian assumptions

2. **Different penalty mechanisms**:
   - **BIC** has the strongest explicit penalty for complexity
   - **CH** has moderate implicit penalty through degrees of freedom
   - **Silhouette** has weak implicit penalty (favors interpretable K)
   - **DB** has minimal penalty (may favor higher K)

3. **Different mathematical scales**:
   - Metrics operate on different scales and cannot be directly compared
   - Peak detection methods differ (global maximum vs. elbow detection)

### 6.4 Complementary Nature

The metrics are **complementary**, not redundant:

- **Geometric + Statistical + Probabilistic** = Robust validation
- Agreement across metrics → High confidence in K selection
- Disagreement → Need for domain knowledge or additional investigation

### 6.5 Expected Behavior Patterns

#### 6.5.1 As K Increases from 2 to K_max

| Metric | Typical Behavior |
|--------|------------------|
| **WCSS** | Monotonic decrease (always) |
| **Silhouette** | Inverted U-shape (peak at optimal K) |
| **CH** | Inverted U-shape (peak at optimal K) |
| **DB** | U-shape (minimum at optimal K) |
| **BIC** | U-shape (minimum at optimal K) |

#### 6.5.2 Why WCSS Alone Is Insufficient

The **Elbow Method** (plotting WCSS vs. K) is popular but problematic:

$$\text{WCSS}(K) = \sum_{k=1}^{K} \sum_{x_i \in C_k} \lVert x_i - \mu_k \rVert^2$$

**Issues**:
1. **Subjective**: No mathematical definition of "elbow"
2. **Ambiguous**: Many datasets have no clear elbow
3. **Always decreasing**: No stopping criterion
4. **No statistical validity**: No hypothesis testing framework

**Solution**: Use metrics with **explicit optimization criteria** (max or min).

---

## 7. Optimal K Selection Methodology

This section describes the **step-by-step process** for determining the optimal number of clusters.

### 7.1 Search Space Definition

Define the range of K values to evaluate:

$$K \in \{K_{\min}, K_{\min}+1, \dots, K_{\max}\}$$

**Practical considerations**:
- **K_min = 2**: Minimum meaningful clustering (K=1 is trivial)
- **K_max**: Typically chosen based on:
  - **Rule of thumb**: $K_{\max} \approx \sqrt{n/2}$
  - **Domain knowledge**: Expected number of natural groups
  - **Computational budget**: Each K requires model fitting

For this project: $K \in \{2, 3, 4, 5, 6, 7, 8, 9, 10\}$

### 7.2 Metric Computation Algorithm

For each candidate value $K$:

```python
for K in range(K_min, K_max + 1):
    # Step 1: Fit K-Means
    kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    
    # Step 2: Compute geometric/statistical metrics
    silhouette = silhouette_score(X, labels)
    ch_score = calinski_harabasz_score(X, labels)
    db_score = davies_bouldin_score(X, labels)
    
    # Step 3: Fit GMM for probabilistic metric
    gmm = GaussianMixture(n_components=K, random_state=42)
    gmm.fit(X)
    bic = gmm.bic(X)
    
    # Step 4: Store results
    metrics_table[K] = (silhouette, ch_score, db_score, bic)
```

### 7.3 Individual Metric Optimization

For each metric, identify its optimal K:

1. **Silhouette Score**:
   $$K_{\text{Sil}}^* = \arg\max_{K} S(K)$$

2. **Calinski-Harabasz Index**:
   $$K_{\text{CH}}^* = \arg\max_{K} \text{CH}(K)$$

3. **Davies-Bouldin Index**:
   $$K_{\text{DB}}^* = \arg\min_{K} \text{DB}(K)$$

4. **Bayesian Information Criterion**:
   $$K_{\text{BIC}}^* = \arg\min_{K} \text{BIC}(K)$$

### 7.4 Example Metrics Table

| K | Silhouette ↑ | CH Index ↑ | DB Index ↓ | BIC ↓ |
|---|-------------|-----------|-----------|-------|
| 2 | 0.5234 | 1245.67 | 0.8923 | 15234.5 |
| 3 | 0.6123 | 1567.89 | 0.7234 | 14567.2 |
| 4 | **0.6456** | **1789.34** | **0.6789** | **14123.8** |
| 5 | 0.6234 | 1656.23 | 0.7012 | 14234.6 |
| 6 | 0.6012 | 1534.56 | 0.7456 | 14456.9 |
| 7 | 0.5789 | 1423.45 | 0.7789 | 14678.3 |
| 8 | 0.5567 | 1334.12 | 0.8123 | 14890.7 |
| 9 | 0.5345 | 1256.78 | 0.8456 | 15102.4 |
| 10 | 0.5123 | 1189.45 | 0.8789 | 15314.8 |

In this example:
- $K_{\text{Sil}}^* = 4$ (maximum Silhouette)
- $K_{\text{CH}}^* = 4$ (maximum CH)
- $K_{\text{DB}}^* = 4$ (minimum DB)
- $K_{\text{BIC}}^* = 4$ (minimum BIC)

All metrics agree: **K* = 4**

---

## 8. Decision Algorithm and Consensus Strategy

When metrics disagree, we need a **robust decision mechanism**.

### 8.1 Majority Voting Algorithm

The consensus-based selection strategy uses democratic voting:

```python
def select_best_k(metrics_table):
    """
    Select optimal K using majority voting across metrics.
    
    Returns:
        best_k: The K value supported by the majority of metrics
    """
    # Step 1: Identify optimal K for each metric
    k_silhouette = max(metrics_table, key=lambda k: metrics_table[k]['silhouette'])
    k_ch = max(metrics_table, key=lambda k: metrics_table[k]['ch'])
    k_db = min(metrics_table, key=lambda k: metrics_table[k]['db'])
    k_bic = min(metrics_table, key=lambda k: metrics_table[k]['bic'])
    
    # Step 2: Collect votes
    votes = [k_silhouette, k_ch, k_db, k_bic]
    
    # Step 3: Count occurrences
    from collections import Counter
    vote_counts = Counter(votes)
    
    # Step 4: Select K with most votes
    best_k = vote_counts.most_common(1)[0][0]
    
    return best_k, vote_counts
```

### 8.2 Decision Cases

#### Case 1: Full Consensus (4 votes)
```
Votes: [4, 4, 4, 4]
Result: K* = 4 (100% confidence)
```
**Interpretation**: Strong evidence for K=4 across all mathematical frameworks.

#### Case 2: Strong Majority (3 votes)
```
Votes: [4, 4, 4, 5]
Result: K* = 4 (75% confidence)
```
**Interpretation**: Three metrics agree on K=4; one outlier suggests K=5. Select K=4 but investigate K=5.

#### Case 3: Weak Majority (2 votes, tie-breaking)
```
Votes: [3, 3, 4, 5]
Vote counts: {3: 2, 4: 1, 5: 1}
Result: K* = 3 (50% confidence)
```
**Interpretation**: No strong consensus. Additional analysis recommended:
- Examine domain constraints
- Visualize both K=3 and K=4 solutions
- Consider external validation

#### Case 4: Complete Disagreement (no majority)
```
Votes: [2, 3, 4, 5]
Vote counts: {2: 1, 3: 1, 4: 1, 5: 1}
Result: Requires manual investigation
```
**Interpretation**: Metrics fundamentally disagree. Possible causes:
- No clear cluster structure in data
- Different metrics capture different aspects
- Need for domain expert input

### 8.3 Confidence Scoring

Define a confidence score for the selected K:

$$\text{Confidence}(K^*) = \frac{\text{Number of votes for } K^*}{\text{Total number of metrics}} \times 100\%$$

| Confidence | Interpretation | Action |
|------------|---------------|--------|
| 100% (4/4) | Very High | Accept K* with confidence |
| 75% (3/4) | High | Accept K*, note minority vote |
| 50% (2/4) | Moderate | Accept K* with caution, investigate alternatives |
| 25% (1/4) | Low | Reject automatic selection, manual analysis required |

### 8.4 Tiebreaker Rules

When multiple K values have equal votes, apply tiebreakers in order:

1. **Parsimony principle**: Choose smaller K (Occam's razor)
2. **Silhouette priority**: Choose K preferred by Silhouette (most interpretable)
3. **BIC priority**: Choose K preferred by BIC (theoretically grounded)

### 8.5 Output Report Format

The algorithm generates a structured report:

```
=== CLUSTERING METRICS REPORT ===

K=2 | Silhouette=0.5234 | CH=1245.67 | DB=0.8923 | BIC=15234.5
K=3 | Silhouette=0.6123 | CH=1567.89 | DB=0.7234 | BIC=14567.2
K=4 | Silhouette=0.6456 | CH=1789.34 | DB=0.6789 | BIC=14123.8 ← OPTIMAL
K=5 | Silhouette=0.6234 | CH=1656.23 | DB=0.7012 | BIC=14234.6
...

METRIC VOTES:
  - Silhouette suggests: K=4
  - Calinski-Harabasz suggests: K=4
  - Davies-Bouldin suggests: K=4
  - BIC suggests: K=4

CONSENSUS DECISION: K* = 4
CONFIDENCE: 100% (4/4 metrics agree)
```

---

## 9. Implementation Results

### 9.1 Project Structure

```
clustering_project/
│
├── data/
│   └── raw/
│       └── ev_charging_patterns.csv
│
├── results/
│   ├── metrics_report.txt          # Numerical results for all K
│   ├── final_clusters.csv          # Data with cluster assignments
│   └── plots/
│       ├── clusters_2d_K_4.png     # 2D PCA visualization
│       └── clusters_3d_K_4.png     # 3D visual representation
│
├── preprocissing.py                # Data cleaning and scaling
├── clustering.py                   # Metric computation and visualization
├── main.py                         # Pipeline orchestration
└── README.md                       # This document
```

### 9.2 Computational Workflow

1. **Preprocessing** (`preprocissing.py`):
   - Load raw data
   - Clean (remove NaN, duplicates)
   - Engineer features (duration calculation)
   - Encode categorical variables
   - Standardize features
   - Generate 2D PCA for visualization only

2. **Metric Computation** (`clustering.py`):
   - For K ∈ {2, ..., 10}:
     - Fit K-Means on full-dimensional data
     - Compute Silhouette, CH, DB on full-dimensional data
     - Fit GMM and compute BIC on full-dimensional data
   - Save metrics to `results/metrics_report.txt`

3. **Decision** (`main.py`):
   - Apply majority voting algorithm
   - Select optimal K*
   - Print consensus result

4. **Final Clustering**:
   - Fit K-Means with K* on full-dimensional data
   - Assign cluster labels
   - Save to `results/final_clusters.csv`
   - Generate visualizations on 2D PCA (interpretation only)

### 9.3 Visualization Strategy

**CRITICAL**: Visualizations are created on 2D PCA projections **only for interpretation**, not for decision-making.

#### 9.3.1 2D Scatter Plot
- **Purpose**: Illustrate cluster separation in reduced space
- **Data**: PCA-reduced (2 components)
- **Limitation**: May not reflect true high-dimensional structure

#### 9.3.2 3D Visual Trick
- **Purpose**: Add visual depth to 2D projection
- **Method**: Use cluster ID as z-coordinate
- **Limitation**: Purely visual; not mathematically meaningful

### 9.4 Reproducibility

All computations use fixed random seeds:
```python
random_state=42  # For K-Means and GMM
```

This ensures:
- Identical centroid initialization across runs
- Reproducible cluster assignments
- Consistent metric values

---

## 10. Conclusion

### 10.1 Summary of Methodology

This project demonstrates a **rigorous, multi-metric approach** to unsupervised clustering:

1. **Objective criteria**: Mathematical optimization rules replace subjective judgment
2. **Multi-perspective validation**: Four complementary metrics from different frameworks
3. **Consensus-based decision**: Democratic voting aggregates evidence
4. **Full-dimensional analysis**: Metrics computed on complete feature space
5. **Visualization for interpretation**: PCA used only for human understanding

### 10.2 Key Contributions

1. **Mathematical rigor**: Complete derivations and interpretations of all metrics
2. **Explicit decision algorithm**: Reproducible, transparent K-selection process
3. **Consensus strategy**: Principled approach to handling metric disagreement
4. **Academic standards**: Compliant with Master-level research methodology

### 10.3 Theoretical Foundations

The methodology integrates three mathematical perspectives:

| Perspective | Metric | Foundation |
|-------------|--------|-----------|
| **Geometric** | Silhouette, DB | Euclidean distance theory |
| **Statistical** | Calinski-Harabasz | ANOVA, variance decomposition |
| **Probabilistic** | BIC | Bayesian model selection, likelihood theory |

### 10.4 Practical Implications

This approach is superior to ad-hoc methods because:

1. **Objectivity**: Decisions are based on mathematical criteria, not visual inspection
2. **Robustness**: Multiple metrics reduce sensitivity to individual metric biases
3. **Transparency**: Every step is explicit and reproducible
4. **Scalability**: Works for high-dimensional data where visualization fails

### 10.5 Limitations and Future Work

**Current limitations**:
- Metrics assume distance-based or density-based cluster definitions
- May not detect highly non-convex or overlapping clusters
- No explicit handling of noise/outliers

**Potential extensions**:
1. **Additional metrics**: Include Dunn Index, Gap Statistic, or X-Means
2. **Ensemble clustering**: Bootstrap aggregation of cluster assignments
3. **Stability analysis**: Measure cluster consistency across subsamples
4. **Hierarchical validation**: Combine hierarchical and partitional approaches
5. **Domain-specific constraints**: Incorporate business rules or expert knowledge

### 10.6 Final Remarks

The optimal number of clusters is not a purely mathematical question—it depends on:
- **Mathematical evidence**: Provided by our metrics
- **Domain knowledge**: What makes sense in the application context
- **Practical constraints**: Interpretability, actionability, computational cost

This project provides the **mathematical foundation** for informed decision-making, while acknowledging that the final choice may also incorporate domain expertise.

---

## References

1. **Rousseeuw, P. J. (1987)**. "Silhouettes: A graphical aid to the interpretation and validation of cluster analysis." *Journal of Computational and Applied Mathematics*, 20, 53-65.

2. **Caliński, T., & Harabasz, J. (1974)**. "A dendrite method for cluster analysis." *Communications in Statistics - Theory and Methods*, 3(1), 1-27.

3. **Davies, D. L., & Bouldin, D. W. (1979)**. "A cluster separation measure." *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 1(2), 224-227.

4. **Schwarz, G. (1978)**. "Estimating the dimension of a model." *The Annals of Statistics*, 6(2), 461-464.

5. **MacQueen, J. (1967)**. "Some methods for classification and analysis of multivariate observations." *Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics and Probability*, 1, 281-297.

6. **Dempster, A. P., Laird, N. M., & Rubin, D. B. (1977)**. "Maximum likelihood from incomplete data via the EM algorithm." *Journal of the Royal Statistical Society: Series B*, 39(1), 1-22.

---

**Document prepared by:**  
**Yahya Bouchak**  
Master Student – SIIA  
January 2026

---

**Version:** 2.0  
**Last Updated:** January 31, 2026  
**License:** Academic Use Only