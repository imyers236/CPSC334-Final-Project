# some useful mysklearn package import statements and reloads
import importlib
import os
import mysklearn.myutils
import mysklearn.myutils as myutils
import mysklearn.mypytable
from mysklearn.mypytable import MyPyTable 
import mysklearn.myclassifiers
from mysklearn.myclassifiers import MyKNeighborsClassifier, MyDummyClassifier, MyNaiveBayesClassifier, MyDecisionTreeClassifier
import mysklearn.myevaluation
import mysklearn.myevaluation as myevaluation

# Load file
madness_data = MyPyTable()
filename = os.path.join("input_data", "tournament_games2016-2021.csv")
madness_data.load_from_file(filename)
rows, cols = madness_data.get_shape()

# Assign X and y train and test for tournament seed only
X_seed = [[val] for val in madness_data.get_column("TournamentSeed")]
X_subset = [val for val in madness_data.get_columns(["LastOrdinalRank", "RegularSeasonFGPercentMean", "RegularSeasonAwayPercentageWon"])]
y = [val for val in madness_data.get_column("Winner")]


# Set up naive bayes, knn and dummy, and decision tree
naive_clf = MyNaiveBayesClassifier()
knn_clf = MyKNeighborsClassifier(n_neighbors=5, categorical=True)
dum_clf = MyDummyClassifier()
tree_clf = MyDecisionTreeClassifier()

# Cross val predict
tree_acc, tree_error, tree_precision, tree_recall, tree_f1 = myevaluation.cross_val_predict(tree_clf, X_seed, y, 10)
naive_acc, naive_error, naive_precision, naive_recall, naive_f1 = myevaluation.cross_val_predict(naive_clf, X_seed, y, 10)
knn_acc, knn_error, knn_precision, knn_recall, knn_f1 = myevaluation.cross_val_predict(knn_clf, X_seed, y, 10)
dum_acc, dum_error, dum_precision, dum_recall, dum_f1 = myevaluation.cross_val_predict(dum_clf, X_seed, y, 10)

# Print Results
print("===========================================")
print("STEP 1: Accuracy and error rate")
print("===========================================")
print("Decision Tree Classifier: accuracy =", round(tree_acc,2), " error rate =", round(tree_error,2))
print("Naive Bayes Classifier: accuracy =", round(naive_acc,2), " error rate =", round(naive_error,2))
print("k Nearest Neighbors Classifier: accuracy =", round(knn_acc,2), " error rate =", round(knn_error,2))
print("Dummy Classifier: accuracy =", round(dum_acc,2), " error rate =", round(dum_error,2))
print()
print("===========================================")
print("STEP 2: Precision, recall, and F1 measure")
print("===========================================")
print("Decision Tree Classifier: precision =", round(tree_precision,2), " recall =", round(tree_recall,2), " F1 =", round(tree_f1,2))
print("Naive Bayes Classifier: precision =", round(naive_precision,2), " recall =", round(naive_recall,2), " F1 =", round(naive_f1,2))
print("k Nearest Neighbors Classifier: precision =", round(knn_precision,2), " recall =", round(knn_recall,2), " F1 =", round(knn_f1,2))
print("Dummy Classifier: precision =", round(dum_precision,2), " recall =", round(dum_recall,2), " F1 =", round(dum_f1,2))
print()
print("===========================================")
print("STEP 3: Confusion Matrix")
print("===========================================")
labels = ["H", "A"]
print("Decision Tree Classifier:")
myevaluation.pretty_print_confusion_matrix(tree_clf, X_seed, y, labels, "Winner", k_sub_samples=10)
print("Naive Bayes Classifier:")
myevaluation.pretty_print_confusion_matrix(naive_clf, X_seed, y, labels, "Winner", k_sub_samples=10)
print("k Nearest Neighbors Classifier:")
myevaluation.pretty_print_confusion_matrix(knn_clf, X_seed, y, labels, "Winner", k_sub_samples=10)
print("Dummy Classifier:")
myevaluation.pretty_print_confusion_matrix(dum_clf, X_seed, y, labels, "Winner", k_sub_samples=10)
