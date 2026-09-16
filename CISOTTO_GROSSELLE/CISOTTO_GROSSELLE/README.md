# Multiclass Logistic Regression

A Python implementation of first-order optimization algorithms from scratch using **NumPy** to solve multiclass logistic regression problems by minimizing negative log-likelihood on high-dimensional data.

---

## Algorithms

* **Gradient Descent (GD):** Full-vector first-order optimization.
* **Block Coordinate Gradient Descent (BCGD):** Implemented with the **Gauss-Southwell rule** for greedy coordinate selection.
* **Hyperparameter Tuning:** Grid Search on Lipschitz constant factors ($q$) to analyze trade-offs between per-iteration progress and overall CPU execution time.
* **Benchmarking:** Comparative evaluation on both synthetic datasets and real-world high-dimensional data.

---

## Repository Structure

* `Multiclass_logistic_optimization.py`  
  Contains the core Python implementation of GD and BCGD algorithms from scratch, along with synthetic data generation, hyperparameter tuning routines, and performance metrics evaluation.

* `isolet5.data`  
  The real-world ISOLET5 speech recognition dataset used for training, testing, and performance benchmarking.

* `Report.pdf`  
  Detailed project report containing mathematical derivations, experimental setups, performance plots, and analytical discussions.

---

## Dataset: ISOLET5

The repository includes `isolet5.data`, a benchmark speech recognition dataset consisting of feature vectors generated from spoken letter pronunciation. It serves as a high-dimensional testbed for comparing the convergence rates and CPU execution times of GD vs. BCGD (Gauss-Southwell).

---

## Requirements & Tech Stack

* **Language:** Python 3.x
* **Primary Libraries:** NumPy, Matplotlib, SciPy / scikit-learn (for evaluation/utilities)

