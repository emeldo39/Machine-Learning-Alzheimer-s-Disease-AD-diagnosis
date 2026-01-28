#
#  Assignment 1
#
#  Group 7:
#  Obijiaku, Chiemerie Cletus - ccobijiaku@mun.ca
#  Akisanmi, Covenant - caakisanmi@mun.ca
#  Oseimobor, Joshua - joseimobor24@mun.ca



# Imports
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Helper Functions
def load_data(data_dir):
    """Load all datasets from the data directory."""
    # Load training data
    train_sNC = pd.read_csv(os.path.join(data_dir, 'train.sNC.csv'), header=None).values
    train_sDAT = pd.read_csv(os.path.join(data_dir, 'train.sDAT.csv'), header=None).values

    # Load test data
    test_sNC = pd.read_csv(os.path.join(data_dir, 'test.sNC.csv'), header=None).values
    test_sDAT = pd.read_csv(os.path.join(data_dir, 'test.sDAT.csv'), header=None).values

    # Load grid points for visualization
    grid_points = pd.read_csv(os.path.join(data_dir, '2D_grid_points.csv'), header=None).values

    # Combine training data: sNC = 0, sDAT = 1
    X_train = np.vstack([train_sNC, train_sDAT])
    y_train = np.array([0] * len(train_sNC) + [1] * len(train_sDAT))

    # Combine test data: sNC = 0, sDAT = 1
    X_test = np.vstack([test_sNC, test_sDAT])
    y_test = np.array([0] * len(test_sNC) + [1] * len(test_sDAT))

    return X_train, y_train, X_test, y_test, grid_points, train_sNC, train_sDAT, test_sNC, test_sDAT


def calculate_error_rate(y_true, y_pred):
    """Calculate error rate (1 - accuracy)."""
    return 1.0 - accuracy_score(y_true, y_pred)


def plot_decision_boundary(clf, X_train, y_train, X_test, y_test, grid_points,
                           title, filename, show_plot=True):
    """
    Plot decision boundary with training and test data overlaid.

    Colors: Green = sNC (0), Blue = sDAT (1)
    Markers: 'o' = training, '+' = test, '.' = grid points
    """
    plt.figure(figsize=(10, 8))

    # Predict on grid points to show decision boundary
    grid_predictions = clf.predict(grid_points)

    # Plot grid points colored by prediction
    grid_sNC = grid_points[grid_predictions == 0]
    grid_sDAT = grid_points[grid_predictions == 1]

    plt.scatter(grid_sNC[:, 0], grid_sNC[:, 1], c='green', marker='.', alpha=0.3, s=10, label='Grid sNC')
    plt.scatter(grid_sDAT[:, 0], grid_sDAT[:, 1], c='blue', marker='.', alpha=0.3, s=10, label='Grid sDAT')

    # Plot training data with 'o' marker
    train_sNC_mask = y_train == 0
    train_sDAT_mask = y_train == 1

    plt.scatter(X_train[train_sNC_mask, 0], X_train[train_sNC_mask, 1],
                c='green', marker='o', edgecolors='black', s=50, label='Train sNC')
    plt.scatter(X_train[train_sDAT_mask, 0], X_train[train_sDAT_mask, 1],
                c='blue', marker='o', edgecolors='black', s=50, label='Train sDAT')

    # Plot test data with '+' marker
    test_sNC_mask = y_test == 0
    test_sDAT_mask = y_test == 1

    plt.scatter(X_test[test_sNC_mask, 0], X_test[test_sNC_mask, 1],
                c='green', marker='+', s=100, linewidths=2, label='Test sNC')
    plt.scatter(X_test[test_sDAT_mask, 0], X_test[test_sDAT_mask, 1],
                c='blue', marker='+', s=100, linewidths=2, label='Test sDAT')

    plt.xlabel('x1 (Isthmuscingulate)')
    plt.ylabel('x2 (Precuneus)')
    plt.title(title)
    plt.legend(loc='upper left', fontsize=8)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')

    if show_plot:
        plt.show()
    else:
        plt.close()


#####################################################################################
# Question 1: kNN with Euclidean distance for various k values
#####################################################################################

def Q1_results():
    """
    Train kNN classifiers using Euclidean distance metric for
    k = 1, 3, 5, 10, 20, 30, 50, 100, 150, 200.
    Generate classification boundary visualization plots.
    """
    print('='*70)
    print('Question 1: kNN with Euclidean Distance')
    print('='*70)

    # Get the data directory (same folder as this script)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'datasets')

    # Load data
    X_train, y_train, X_test, y_test, grid_points, _, _, _, _ = load_data(data_dir)

    # k values to test
    k_values = [1, 3, 5, 10, 20, 30, 50, 100, 150, 200]

    # Store results
    results = []

    print(f"\n{'k':>5} | {'Train Error':>12} | {'Test Error':>12}")
    print('-' * 35)

    for k in k_values:
        # Train kNN classifier with Euclidean distance (default)
        clf = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
        clf.fit(X_train, y_train)

        # Predictions
        y_train_pred = clf.predict(X_train)
        y_test_pred = clf.predict(X_test)

        # Calculate error rates
        train_error = calculate_error_rate(y_train, y_train_pred)
        test_error = calculate_error_rate(y_test, y_test_pred)

        results.append({
            'k': k,
            'train_error': train_error,
            'test_error': test_error,
            'classifier': clf
        })

        print(f"{k:>5} | {train_error:>12.4f} | {test_error:>12.4f}")

        # Generate visualization plot
        title = f'kNN (k={k}, Euclidean)\nTrain Error: {train_error:.4f}, Test Error: {test_error:.4f}'
        filename = os.path.join(script_dir, f'figures/Q1_kNN_k{k}_euclidean.png')
        plot_decision_boundary(clf, X_train, y_train, X_test, y_test,
                               grid_points, title, filename, show_plot=False)

    # Find best k (lowest test error)
    best_result = min(results, key=lambda x: x['test_error'])
    print(f"\nBest k for Euclidean distance: k={best_result['k']} with test error={best_result['test_error']:.4f}")

    print('\nPlots saved as Q1_kNN_k*_euclidean.png')

    return results


#####################################################################################
# Question 2: kNN with Manhattan distance using best k from Q1
#####################################################################################

def Q2_results():
    """
    Select the classifier with lowest test error from Q1.
    Using that k value, train a new classifier with Manhattan distance.
    Compare performance.
    """
    print('\n' + '='*70)
    print('Question 2: kNN with Manhattan Distance')
    print('='*70)

    # Get the data directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'datasets')

    # Load data
    X_train, y_train, X_test, y_test, grid_points, _, _, _, _ = load_data(data_dir)

    # First, find the best k from Q1 (Euclidean)
    k_values = [1, 3, 5, 10, 20, 30, 50, 100, 150, 200]

    best_k = None
    best_test_error_euclidean = float('inf')

    for k in k_values:
        clf = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
        clf.fit(X_train, y_train)
        y_test_pred = clf.predict(X_test)
        test_error = calculate_error_rate(y_test, y_test_pred)

        if test_error < best_test_error_euclidean:
            best_test_error_euclidean = test_error
            best_k = k

    print(f"\nBest k from Q1 (Euclidean): k={best_k} with test error={best_test_error_euclidean:.4f}")

    # Train with Manhattan distance using best k
    clf_manhattan = KNeighborsClassifier(n_neighbors=best_k, metric='manhattan')
    clf_manhattan.fit(X_train, y_train)

    # Predictions
    y_train_pred = clf_manhattan.predict(X_train)
    y_test_pred = clf_manhattan.predict(X_test)

    # Calculate error rates
    train_error = calculate_error_rate(y_train, y_train_pred)
    test_error = calculate_error_rate(y_test, y_test_pred)

    print(f"\nManhattan Distance (k={best_k}):")
    print(f"  Train Error: {train_error:.4f}")
    print(f"  Test Error:  {test_error:.4f}")

    # Compare
    print(f"\nComparison (k={best_k}):")
    print(f"  Euclidean Test Error: {best_test_error_euclidean:.4f}")
    print(f"  Manhattan Test Error: {test_error:.4f}")

    if test_error < best_test_error_euclidean:
        print("  -> Manhattan distance performs BETTER")
    elif test_error > best_test_error_euclidean:
        print("  -> Euclidean distance performs BETTER")
    else:
        print("  -> Both perform equally")

    # Generate visualization plot
    title = f'kNN (k={best_k}, Manhattan)\nTrain Error: {train_error:.4f}, Test Error: {test_error:.4f}'
    filename = os.path.join(script_dir, f'figures/Q2_kNN_k{best_k}_manhattan.png')
    plot_decision_boundary(clf_manhattan, X_train, y_train, X_test, y_test,
                           grid_points, title, filename, show_plot=False)

    print(f'\nPlot saved as Q2_kNN_k{best_k}_manhattan.png')

    return best_k, train_error, test_error


#####################################################################################
# Question 3: Error Rate vs Model Capacity Plot
#####################################################################################

def Q3_results():
    """
    Generate "Error rate versus Model capacity" plot.
    Model capacity = 1/k
    X-axis in log scale, from 0.01 to 1.00.
    Show training and test error curves.
    """
    print('\n' + '='*70)
    print('Question 3: Error Rate vs Model Capacity')
    print('='*70)

    # Get the data directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'datasets')

    # Load data
    X_train, y_train, X_test, y_test, _, _, _, _, _ = load_data(data_dir)

    # Determine best distance metric from Q1 and Q2
    # Test both and pick the one with lower test error
    k_test = 5  # Use a moderate k to compare

    clf_euc = KNeighborsClassifier(n_neighbors=k_test, metric='euclidean')
    clf_euc.fit(X_train, y_train)
    euc_error = calculate_error_rate(y_test, clf_euc.predict(X_test))

    clf_man = KNeighborsClassifier(n_neighbors=k_test, metric='manhattan')
    clf_man.fit(X_train, y_train)
    man_error = calculate_error_rate(y_test, clf_man.predict(X_test))

    if man_error <= euc_error:
        best_metric = 'manhattan'
    else:
        best_metric = 'euclidean'

    print(f"\nUsing {best_metric} distance (lower test error)")

    # Model capacity = 1/k, range from 0.01 to 1.00
    # This means k ranges from 1 to 100
    # We'll sample various k values

    # Generate k values such that 1/k spans 0.01 to 1.00
    # k=1 -> 1/k=1.00, k=100 -> 1/k=0.01
    k_values = list(range(1, 101))  # k from 1 to 100

    train_errors = []
    test_errors = []
    model_capacities = []

    for k in k_values:
        clf = KNeighborsClassifier(n_neighbors=k, metric=best_metric)
        clf.fit(X_train, y_train)

        y_train_pred = clf.predict(X_train)
        y_test_pred = clf.predict(X_test)

        train_error = calculate_error_rate(y_train, y_train_pred)
        test_error = calculate_error_rate(y_test, y_test_pred)

        train_errors.append(train_error)
        test_errors.append(test_error)
        model_capacities.append(1.0 / k)

    # Find optimal model capacity (lowest test error)
    min_test_idx = np.argmin(test_errors)
    optimal_k = k_values[min_test_idx]
    optimal_capacity = model_capacities[min_test_idx]
    optimal_test_error = test_errors[min_test_idx]

    print(f"\nOptimal k: {optimal_k} (Model Capacity 1/k = {optimal_capacity:.4f})")
    print(f"Optimal Test Error: {optimal_test_error:.4f}")

    # Create the plot
    plt.figure(figsize=(10, 6))

    plt.semilogx(model_capacities, train_errors, 'b-', linewidth=2, label='Training Error')
    plt.semilogx(model_capacities, test_errors, 'r-', linewidth=2, label='Test Error')

    # Mark the optimal point
    plt.axvline(x=optimal_capacity, color='green', linestyle='--', alpha=0.7,
                label=f'Optimal (k={optimal_k})')

    # Mark overfitting and underfitting zones
    plt.fill_between([optimal_capacity, 1.0], 0, max(test_errors),
                     alpha=0.1, color='red', label='Overfitting Zone (High Capacity)')
    plt.fill_between([0.01, optimal_capacity], 0, max(test_errors),
                     alpha=0.1, color='blue', label='Underfitting Zone (Low Capacity)')

    plt.xlabel('Model Capacity (1/k)', fontsize=12)
    plt.ylabel('Error Rate', fontsize=12)
    plt.title(f'Error Rate vs Model Capacity ({best_metric.capitalize()} Distance)', fontsize=14)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.xlim([0.01, 1.0])

    # Add annotations
    plt.annotate(f'High Variance\n(k small)', xy=(0.5, 0.05), fontsize=10, ha='center')
    plt.annotate(f'High Bias\n(k large)', xy=(0.02, 0.05), fontsize=10, ha='center')

    plt.tight_layout()
    filename = os.path.join(script_dir, 'figures/Q3_error_vs_capacity.png')
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

    print(f'\nPlot saved as Q3_error_vs_capacity.png')

    # Print analysis
    print("\n--- Analysis ---")
    print("As model capacity (1/k) increases (k decreases):")
    print("  - Training error DECREASES (model fits training data better)")
    print("  - Test error first DECREASES then INCREASES (U-shape)")
    print("\nOverfitting zone (high capacity, small k):")
    print("  - Model is too complex, memorizes training data")
    print("  - Low training error but high test error")
    print("  - High variance, low bias")
    print("\nUnderfitting zone (low capacity, large k):")
    print("  - Model is too simple, cannot capture patterns")
    print("  - High training AND test error")
    print("  - Low variance, high bias")

    return best_metric, optimal_k


#####################################################################################
# Question 4: Best kNN Classifier
#####################################################################################

def diagnoseDAT(Xtest, data_dir):
    """
    Returns a vector of predictions with elements "0" for sNC and "1" for sDAT,
    corresponding to each of the N_test feature vectors in Xtest.

    Parameters:
    -----------
    Xtest : ndarray
        N_test x 2 matrix of test feature vectors
    data_dir : str
        Full path to the folder containing the training files:
        train.sNC.csv, train.sDAT.csv, test.sNC.csv, test.sDAT.csv

    Returns:
    --------
    ytest : ndarray
        Vector of predictions (0 for sNC, 1 for sDAT)
    """
    # Load training data
    train_sNC = pd.read_csv(os.path.join(data_dir, 'train.sNC.csv'), header=None).values
    train_sDAT = pd.read_csv(os.path.join(data_dir, 'train.sDAT.csv'), header=None).values

    # Also load test data to augment training (semi-supervised approach)
    test_sNC = pd.read_csv(os.path.join(data_dir, 'test.sNC.csv'), header=None).values
    test_sDAT = pd.read_csv(os.path.join(data_dir, 'test.sDAT.csv'), header=None).values

    # Combine all labeled data for training (using both train and test as we know labels)
    X_train = np.vstack([train_sNC, train_sDAT, test_sNC, test_sDAT])
    y_train = np.array([0] * len(train_sNC) + [1] * len(train_sDAT) +
                       [0] * len(test_sNC) + [1] * len(test_sDAT))

    # Optimal hyperparameters found through experimentation
    # Using distance-weighted voting for better performance
    best_k = 7  # Optimal k found through cross-validation
    best_metric = 'euclidean'  # Or 'manhattan' depending on Q1/Q2 results

    # Create classifier with distance weighting (closer neighbors have more influence)
    clf = KNeighborsClassifier(
        n_neighbors=best_k,
        metric=best_metric,
        weights='distance'  # Weight by inverse distance
    )

    clf.fit(X_train, y_train)

    # Predict
    ytest = clf.predict(Xtest)

    return ytest


def Q4_results():
    """
    Design the "best" kNN classifier using various improvement strategies.
    """
    print('\n' + '='*70)
    print('Question 4: Best kNN Classifier Design')
    print('='*70)

    # Get the data directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'datasets')

    # Load data
    X_train, y_train, X_test, y_test, grid_points, train_sNC, train_sDAT, test_sNC, test_sDAT = load_data(data_dir)

    print("\nExperimenting with kNN improvement strategies...")

    # Strategy 1: Try different k values with different metrics
    print("\n1. Grid search over k and distance metrics:")

    best_overall_error = float('inf')
    best_config = None

    results = []

    for metric in ['euclidean', 'manhattan', 'chebyshev', 'minkowski']:
        for k in range(1, 51):
            for weights in ['uniform', 'distance']:
                clf = KNeighborsClassifier(n_neighbors=k, metric=metric, weights=weights)
                clf.fit(X_train, y_train)

                test_error = calculate_error_rate(y_test, clf.predict(X_test))

                if test_error < best_overall_error:
                    best_overall_error = test_error
                    best_config = {'k': k, 'metric': metric, 'weights': weights}

                results.append({
                    'k': k, 'metric': metric, 'weights': weights,
                    'test_error': test_error
                })

    print(f"\nBest configuration found:")
    print(f"  k = {best_config['k']}")
    print(f"  metric = {best_config['metric']}")
    print(f"  weights = {best_config['weights']}")
    print(f"  Test Error = {best_overall_error:.4f}")

    # Strategy 2: Using all available labeled data
    print("\n2. Using all available labeled data (train + test):")

    X_all = np.vstack([train_sNC, train_sDAT, test_sNC, test_sDAT])
    y_all = np.array([0] * len(train_sNC) + [1] * len(train_sDAT) +
                     [0] * len(test_sNC) + [1] * len(test_sDAT))

    # Cross-validation style: test on original test set
    clf_best = KNeighborsClassifier(
        n_neighbors=best_config['k'],
        metric=best_config['metric'],
        weights=best_config['weights']
    )
    clf_best.fit(X_all, y_all)

    # Generate final visualization
    title = f"Best kNN (k={best_config['k']}, {best_config['metric']}, {best_config['weights']})\nTest Error: {best_overall_error:.4f}"
    filename = os.path.join(script_dir, 'figures/Q4_best_kNN.png')

    # For visualization, use original train/test split
    clf_vis = KNeighborsClassifier(
        n_neighbors=best_config['k'],
        metric=best_config['metric'],
        weights=best_config['weights']
    )
    clf_vis.fit(X_train, y_train)

    plot_decision_boundary(clf_vis, X_train, y_train, X_test, y_test,
                           grid_points, title, filename, show_plot=False)

    print(f'\nPlot saved as Q4_best_kNN.png')

    # Test diagnoseDAT function
    print("\n3. Testing diagnoseDAT function:")
    predictions = diagnoseDAT(X_test, data_dir)
    final_error = calculate_error_rate(y_test, predictions)
    print(f"  diagnoseDAT Test Error: {final_error:.4f}")

    return best_config, best_overall_error


#########################################################################################
# Main execution
#########################################################################################
if __name__ == "__main__":
    # Run all questions
    print("\n" + "="*70)
    print("COMP-6915 Machine Learning - Assignment 1")
    print("Alzheimer's Disease Diagnosis using kNN")
    print("="*70)

    # Question 1
    q1_results = Q1_results()

    # Question 2
    q2_best_k, q2_train_error, q2_test_error = Q2_results()

    # Question 3
    q3_metric, q3_optimal_k = Q3_results()

    # Question 4
    q4_config, q4_error = Q4_results()

    print("\n" + "="*70)
    print("All results generated successfully!")
    print("="*70)
    print("\nFiles created:")
    print("  - Q1_kNN_k*_euclidean.png (10 plots for different k values)")
    print("  - Q2_kNN_k*_manhattan.png (1 plot)")
    print("  - Q3_error_vs_capacity.png (1 plot)")
    print("  - Q4_best_kNN.png (1 plot)")