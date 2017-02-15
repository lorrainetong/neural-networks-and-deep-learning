"""
mnist_svm
~~~~~~~~~

A classifier program for recognizing handwritten digits from the MNIST
data set, using an SVM classifier."""

#### Libraries
# Libraries
import os
# Third-party libraries
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

# Change working directory
os.chdir('/Users/tongqiao/neural-networks-and-deep-learning/src')
import mnist_loader
os.chdir('/Users/tongqiao/neural-networks-and-deep-learning')

def svm_baseline():
    training_data, validation_data, test_data = mnist_loader.load_data()

    # 5-fold cross-validation on 1000 samples, uncomment this section to
    # get best parameters
    # ------------------------------------------------------------------
    # C_range = np.logspace(-5, 5, 11, base = 2)
    # gamma_range = np.logspace(-5, 5, 11, base = 2)
    # param_grid = dict(gamma = gamma_range, C = C_range)
    # cv = KFold(n_splits = 5)
    # grid = GridSearchCV(SVC(), param_grid = param_grid, cv = cv)
    # grid.fit(training_data[0][0:1000], training_data[1][0:1000])
    # print("The best parameters are %s with a score of %0.2f"
    #   % (grid.best_params_, grid.best_score_))
    # ------------------------------------------------------------------

    # train
    clf = SVC(C = 2, gamma = 0.03125)
    clf.fit(training_data[0], training_data[1])
    # test
    predictions = [int(a) for a in clf.predict(test_data[0])]
    num_correct = sum(int(a == y) for a, y in zip(predictions, test_data[1]))
    print("Baseline classifier using an SVM.")
    print("%s of %s values correct." % (num_correct, len(test_data[1])))

if __name__ == "__main__":
    svm_baseline()
    
