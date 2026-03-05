import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from warnings import filterwarnings
import os

# MLP doesn't always converge in 2000 iterations — suppress those
# warnings so the 60-line output stays readable
filterwarnings('ignore')


# LOADING DATA

# no header row; 62 columns:
#   cols 0-59 : 60 sonar frequency-band energy readings (features)
#   col 60    : integer label, 1=Rock 2=Mine  <-- use this as y
#   col 61    : string label 'R'/'M'          <-- skip entirely
# col 61 is just col 60 re-encoded as a string, so using it would
# be leaking the label directly into the feature set

# build path from __file__ it avoids hardcoding a path
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, 'sonar_all_data_2.csv')
df = pd.read_csv(data_path, header=None)

X = df.iloc[:, 0:60].values   # (208, 60)
y = df.iloc[:, 60].values     # (208,) — 1=Rock, 2=Mine

print(f"Feature matrix shape : {X.shape}   (208 observations x 60 sonar bands)")
print(f"Target vector shape  : {y.shape}   (class 1=Rock, class 2=Mine)")
print(f"Class counts         : Rocks={np.sum(y==1)}, Mines={np.sum(y==2)}\n")


# TRAIN / TEST SPLIT  (70% train, 30% test)

# fixing random_state means the same rows go to train/test every run,
# so accuracy differences across n_components come from PCA, not the split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42
)

# STANDARDIZE FEATURES

# scale features so PCA doesn't get thrown off by features with larger
# magnitudes dominating the variance — all 60 bands need equal footing
# only fit on training data — fitting on test data would be leakage
# (test set has to stay completely unseen until evaluation)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)   # apply train stats, don't refit


# PCA + MLP LOOP  (n_components = 1 through 60)

# sweep all 60 possible component counts to find the sweet spot:
# too few = we lose too much signal, classifier can't tell mines from rocks
# too many = we start folding in noise and test accuracy drops

print("=" * 55)
print("PCA + MLP Accuracy by Number of Principal Components")
print("=" * 55)

accuracies = []   # one entry per n_components (index 0 = 1 component)

for n in range(1, 61):

    # PCA rotates the 60-D feature space so the first n axes capture the
    # most variance — keeps the useful signal, drops the low-variance noise
    pca = PCA(n_components=n)

    # fit only on training data, then apply the same rotation to test
    # (same leakage rule as the scaler)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca  = pca.transform(X_test_scaled)

    # parameter choices:
    # hidden_layer_sizes=(100,) — one hidden layer, 100 neurons, plenty for ~145 training samples
    # activation='logistic'    — sigmoid, works fine for binary tasks
    # solver='adam'            — adaptive learning rate, solid default for small datasets
    # max_iter=2000            — enough iterations to converge across all 60 dimensionalities
    # alpha=0.00001            — almost no regularization; too much would underfit on 208 samples
    # tol=0.0001               — stop when loss stops improving by this much
    # random_state=42          — fix weight init so accuracy differences reflect PCA, not luck
    model = MLPClassifier(
        hidden_layer_sizes=(100,),
        activation='logistic',
        max_iter=2000,
        alpha=0.00001,
        solver='adam',
        tol=0.0001,
        random_state=42
    )
    model.fit(X_train_pca, y_train)

    y_pred = model.predict(X_test_pca)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)

    print(f"Components: {n:2d},  Test Accuracy: {acc:.4f}")


# PRINT BEST RESULTS

best_acc = max(accuracies)
best_n   = accuracies.index(best_acc) + 1   # list is 0-indexed, components start at 1

print("\n" + "=" * 55)
print(f"Maximum Accuracy: {best_acc:.4f} achieved with {best_n} components")
print("=" * 55 + "\n")


# CONFUSION MATRIX for Best Configuration

# re run the best n to get the full error breakdown
#
# FN = mine classified as rock — the scary error, this is the one that sinks the sub
# FP = rock classified as mine — annoying but safe, just an unnecessary detour

pca_best     = PCA(n_components=best_n)
X_train_best = pca_best.fit_transform(X_train_scaled)
X_test_best  = pca_best.transform(X_test_scaled)

model_best = MLPClassifier(
    hidden_layer_sizes=(100,),
    activation='logistic',
    max_iter=2000,
    alpha=0.00001,
    solver='adam',
    tol=0.0001,
    random_state=42
)
model_best.fit(X_train_best, y_train)
y_pred_best = model_best.predict(X_test_best)

cmat = confusion_matrix(y_test, y_pred_best)

print(f"Confusion Matrix (best config: {best_n} components)")
print("Rows = Actual class, Columns = Predicted class")
print("Classes: 1=Rock (row/col 0), 2=Mine (row/col 1)\n")
print(cmat)
print()
print("Interpretation:")
print(f"  Rocks correctly identified  (TN): {cmat[0,0]}")
print(f"  Rocks misclassified as Mine (FP): {cmat[0,1]}")
print(f"  Mines misclassified as Rock (FN): {cmat[1,0]}  <- DANGER: missed mines")
print(f"  Mines correctly identified  (TP): {cmat[1,1]}")

# Uncomment to verify mine/rock counts in test set:
# rocks = 0
# mines = 0
# for obj in y_test:
#     if obj == 2:
#         mines += 1
#     else:
#         rocks += 1
# print("rocks", rocks, "   mines", mines)


# PLOT: Accuracy vs Number of PCA Components

components = list(range(1, 61))

plt.figure(figsize=(10, 6))
plt.plot(components, accuracies, 'b-o', markersize=4, linewidth=1.5,
         label='Test Accuracy')

# red star makes the optimal trade-off point easy to see on the plot
plt.plot(best_n, best_acc, 'r*', markersize=15,
         label=f'Best: {best_acc:.4f} at {best_n} components')

plt.annotate(
    f'Max: {best_acc:.4f}\n({best_n} components)',
    xy=(best_n, best_acc),
    xytext=(best_n + 3, best_acc - 0.04),
    arrowprops=dict(arrowstyle='->', color='red'),
    fontsize=10, color='red'
)

plt.title('PCA Components vs Test Accuracy (MLPClassifier)', fontsize=14)
plt.xlabel('Number of Principal Components', fontsize=12)
plt.ylabel('Test Accuracy', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=11)
plt.tight_layout()

plot_path = os.path.join(script_dir, 'accuracy_vs_components.png')
plt.savefig(plot_path, dpi=150)
plt.show()
print(f"\nPlot saved to '{plot_path}'")
