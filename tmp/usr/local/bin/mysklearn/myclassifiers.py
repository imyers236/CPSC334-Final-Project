"""
Programmer: Ian Myers
Class: CPSC 322-02, Fall 2024
Programming Assignment #7
11/11/24
Description: This program acts as a wrapper for
  classifiers
"""

import math
from itertools import product
from mysklearn import myutils
from mysklearn.mysimplelinearregressor import MySimpleLinearRegressor


class MySimpleLinearRegressionClassifier:
    """Represents a simple linear regression classifier that discretizes
        predictions from a simple linear regressor (see MySimpleLinearRegressor).

    Attributes:
        discretizer(function): a function that discretizes a numeric value into
            a string label. The function's signature is func(obj) -> obj
        regressor(MySimpleLinearRegressor): the underlying regression model that
            fits a line to x and y data

    Notes:
        Terminology: instance = sample = row and attribute = feature = column
    """

    def __init__(self, discretizer, regressor=None):
        """Initializer for MySimpleLinearClassifier.

        Args:
            discretizer(function): a function that discretizes a numeric value into
                a string label. The function's signature is func(obj) -> obj
            regressor(MySimpleLinearRegressor): the underlying regression model that
                fits a line to x and y data (None if to be created in fit())
        """
        self.discretizer = discretizer
        if regressor is None:
            self.regressor = MySimpleLinearRegressor()
        else:
            self.regressor = regressor

    def fit(self, X_train, y_train):
        """Fits a simple linear regression line to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        """
        self.regressor.fit(X_train, y_train)

    def predict(self, X_test):
        """Makes predictions for test samples in X_test by applying discretizer
            to the numeric predictions from regressor.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        # predicts then sends through discretizer
        y_pred = self.regressor.predict(X_test)
        y_predicted = []
        for x in y_pred:
            y_predicted.append(self.discretizer(x))
        return y_predicted

class MyKNeighborsClassifier:
    """Represents a simple k nearest neighbors classifier.

    Attributes:
        n_neighbors(int): number of k neighbors
        X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples

    Notes:
        Loosely based on sklearn's KNeighborsClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
        Assumes data has been properly normalized before use.
    """
    def __init__(self, n_neighbors=3, categorical = False):
        """Initializer for MyKNeighborsClassifier.

        Args:
            n_neighbors(int): number of k neighbors
            categorical: if true X_test is categorical
        """
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None
        self.categorical = categorical

    def fit(self, X_train, y_train):
        """Fits a kNN classifier to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since kNN is a lazy learning algorithm, this method just stores X_train and y_train
        """
        self.X_train = X_train
        self.y_train = y_train

    def kneighbors(self, X_test):
        """Determines the k closest neighbors of each test instance.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            distances(list of list of float): 2D list of k nearest neighbor distances
                for each instance in X_test
            neighbor_indices(list of list of int): 2D list of k nearest neighbor
                indices in X_train (parallel to distances)
        """
        distances = []
        neighbor_indices = []
        for x in X_test:
            if self.categorical:
                dist_indices = myutils.get_distances_categorical(self.X_train, x)
            else:
                # returns distances and indices in sorted distance order
                dist_indices = myutils.get_distances(self.X_train, x)
            # gets the top k tuples
            top_k = dist_indices[:self.n_neighbors]
            k_distances = []
            k_indices = []
            # parses values into two different arrays
            for k in top_k:
                k_distances.append(k[1])
                k_indices.append(k[0])
            neighbor_indices.append(k_indices)
            distances.append(k_distances)
        return distances, neighbor_indices

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_pred = []
        distances, neighbor_indices = self.kneighbors(X_test)
        # Puts top k y values in a dictionary to find the majority
        for arr in neighbor_indices:
            classifier = {}
            for i in arr:
                if self.y_train[i] not in classifier:
                    classifier[self.y_train[i]] = 1
                else:
                    classifier[self.y_train[i]] = classifier[self.y_train[i]] + 1
            max_key = max(classifier, key=classifier.get)
            y_pred.append(max_key)
        return y_pred

class MyDummyClassifier:
    """Represents a "dummy" classifier using the "most_frequent" strategy.
        The most_frequent strategy is a Zero-R classifier, meaning it ignores
        X_train and produces zero "rules" from it. Instead, it only uses
        y_train to see what the most frequent class label is. That is
        always the dummy classifier's prediction, regardless of X_test.

    Attributes:
        most_common_label(obj): whatever the most frequent class label in the
            y_train passed into fit()

    Notes:
        Loosely based on sklearn's DummyClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html
    """
    def __init__(self):
        """Initializer for DummyClassifier.

        """
        self.most_common_label = None

    def fit(self, X_train, y_train):
        """Fits a dummy classifier to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since Zero-R only predicts the most frequent class label, this method
                only saves the most frequent class label.
        """
        # Finds the majority values in the y_train
        classifier = {}
        for i in y_train:
            if i not in classifier:
                classifier[i] = 1
            else:
                classifier[i] = classifier[i] + 1
        max_key = max(classifier, key=classifier.get)
        self.most_common_label = max_key

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_pred = []
        for i in range(len(X_test)):
            y_pred.append(self.most_common_label)
        return y_pred



class MyNaiveBayesClassifier:
    """Represents a Naive Bayes classifier.

    Attributes:
        priors(YOU CHOOSE THE MOST APPROPRIATE TYPE): The prior probabilities computed for each
            label in the training set.
        posteriors(YOU CHOOSE THE MOST APPROPRIATE TYPE): The posterior probabilities computed for each
            attribute value/label pair in the training set.

    Notes:
        Loosely based on sklearn's Naive Bayes classifiers: https://scikit-learn.org/stable/modules/naive_bayes.html
        You may add additional instance attributes if you would like, just be sure to update this docstring
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyNaiveBayesClassifier.
        """
        self.priors = None
        self.posteriors = None

    def fit(self, X_train, y_train):
        """Fits a Naive Bayes classifier to X_train and y_train.

        Args:
            X_train(list of list of obj): The list of training instances (samples)
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since Naive Bayes is an eager learning algorithm, this method computes the prior probabilities
                and the posterior probabilities for the training data.
            You are free to choose the most appropriate data structures for storing the priors
                and posteriors.
        """
        self.priors = {}
        self.posteriors = {}
        # Finds all the unique values in y_train
        classifier = {}
        for i in y_train:
            if i not in classifier:
                classifier[i] = 1
            else:
                classifier[i] = classifier[i] + 1
        # calculates the priors then puts them in the variable
        for key, value in classifier.items():
            self.priors[key] = value / len(y_train)

        # Add y_train to X_train
        group = [X_train[i] + [y_train[i]] for i in range(len(y_train))]
        for i in range(len(X_train)):
            if y_train[i] not in self.posteriors:
                self.posteriors[y_train[i]] = {}
            for j,val in enumerate(X_train[i]):
                if (str(j) + str(val)) not in self.posteriors[y_train[i]]:
                    self.posteriors[y_train[i]][(str(j) + str(val))] = 1
                else:
                    self.posteriors[y_train[i]][(str(j) + str(val))] += 1
        # calculates the priors then puts them in the variable
        for key, value in classifier.items():
            self.priors[key] = value / len(y_train)

        # Takes counts of every attribute and divides by priors
        for section, att_dict in self.posteriors.items():
            for key, count in att_dict.items():
                self.posteriors[section][key] = count / classifier[section]

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        # initialize dict that holds probs for every class
        probabilities = {}
        y_pred = []
        # runs for every list in X_test
        for x in X_test:
            for section, att_dict in self.posteriors.items():
                prob = 1
                # find attribute in every class and multiple posterior
                for i in range(len(x)):
                    key = str(i) + str(x[i])
                    if key in att_dict:
                        prob = prob * att_dict[key]
                    else:
                        prob = prob * 0
                # multiply prior
                prob = prob * self.priors[section]
                # put in dictionary
                probabilities[section] = prob
            # get highest probility's class label
            highest_prob = max(probabilities, key=probabilities.get)
            y_pred.append(highest_prob)
        return y_pred


class MyDecisionTreeClassifier:
    """Represents a decision tree classifier.

    Attributes:
        X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples
        header(list of strings): holds the amount of attributes
        attribute_domains(dict of lists): for TDIDT for the attribute selection
        tree(nested list): The extracted tree model.

    Notes:
        Loosely based on sklearn's DecisionTreeClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyDecisionTreeClassifier.
        """
        self.X_train = None
        self.y_train = None
        self.header = None
        self.attribute_domains = None
        self.tree = None

    def build_header(self, X_train):
        """
            Builds up the header to be used for TDIDT
        """
        header = []
        att = "att"
        for index in range(len(X_train[0])):
            header.append(att + str(index))
        return header

    def build_domain(self, X_train, header):
        """
            Builds up the attribute domain for TDIDT
        """
        attribute_domains = {}
        # initialize dict
        for att in header:
            attribute_domains[att] = []
        # fill dict
        for instance in X_train:
            for index, att in enumerate(instance):
                if att not in attribute_domains["att" + str(index)]:
                    attribute_domains["att" + str(index)].append(att)

        return attribute_domains

    def select_attribute(self, instances, attributes):
        """ 
            Calculates entropy of each attributes values 
                then takes the weighted average of all attributes 
                and returns the smallest averages attribute to be selected
        """
        weighted_average = {}
        # Goes through each attribute type
        for att in attributes:
            # finds the indexs in instances
            att_index = self.header.index(att)
            att_domain = self.attribute_domains[att]
            weighted_average_val = 0
            # goes through every type of value for the attribute
            for att_value in att_domain: # EX: "Junior" -> "Mid" -> "Senior"
                class_portions = {}
                # finds the amount of instances with each value of the domain
                for instance in instances:
                    if instance[att_index] == att_value:
                        if instance[-1] in class_portions:
                            class_portions[instance[-1]] += 1
                        else:
                            class_portions[instance[-1]] = 1
                # calculate entropy
                entropy_val = 0
                # get sum of class portions values
                entropy_denom = sum(class_portions.values())
                # find the entropy for each value
                for values in class_portions.values():
                    # divide amount of instances of value by sum
                    frac = values / entropy_denom
                    # compute entropy
                    entropy_val += -1*frac * math.log2(frac)
                # add entropy value to weighted_average_val
                weighted_average_val += entropy_val*(entropy_denom / len(instances))
            # put weighted_average_val in the dict under the attribute
            weighted_average[att] = weighted_average_val
        # return the smallest weighted average attribute
        min_key = min(weighted_average, key=weighted_average.get)
        return min_key

    def partition_instances(self, instances, attribute):
        """
            This is group by attribute domain (not values of attribute in instances)
        """
        att_index = self.header.index(attribute)
        att_domain = self.attribute_domains[attribute]
        partitions = {}
        for att_value in att_domain: # "Junior" -> "Mid" -> "Senior"
            partitions[att_value] = []
            for instance in instances:
                if instance[att_index] == att_value:
                    partitions[att_value].append(instance)

        return partitions

    def get_value_list_lengths(self, partitions):
        """
            Returns the total amount of instances within an attribute
        """
        total_att_instances_length = 0
        for value_list in sorted(partitions.values()):
            total_att_instances_length += len(value_list)
        return total_att_instances_length

    def all_same_class(self, instances):
        """
            Checks if all instances have th same class
        """
        first_class = instances[0][-1]
        for instance in instances:
            if instance[-1] != first_class:
                return False
        # get here, then all same class labels
        return True

    def majority_vote_lists(self, instances):
        """
            Checks for the most common class and returns that class
        """
        count_dict = {}
        for classes in instances:
            curr_class = classes[-1]
            if curr_class in count_dict:
                count_dict[curr_class] += 1
            else:
                count_dict[curr_class] = 1
        max_key = max(count_dict, key=count_dict.get)
        return max_key, count_dict[max_key]


    def tdidt(self, current_instances, available_attributes):
        """
            Creates decision tree based on the tdidt system 
        """
        # basic approach (uses recursion!!):
        # select an attribute to split on
        split_attribute = self.select_attribute(current_instances,available_attributes)
        available_attributes.remove(split_attribute) # can't split on this attribute again
        # in this subtree
        tree = ["Attribute", split_attribute]
        # group data by attribute domains (creates pairwise disjoint partitions)
        partitions = self.partition_instances(current_instances, split_attribute)
        # for each partition, repeat unless one of the following occurs (base case)
        for att_value in sorted(partitions.keys()): # process in alphabetical order
            att_partition = partitions[att_value] # gives us the list of instances
            value_subtree = ["Value", att_value]
            #    CASE 1: all class labels of the partition are the same => make a leaf node
            if len(att_partition) > 0 and self.all_same_class(att_partition):
                leaf_class_label = att_partition[0][-1]
                # finds the total amount of values
                total_att_instances_length = self.get_value_list_lengths(partitions)
                leaf_subtree = ["Leaf", leaf_class_label, len(att_partition), total_att_instances_length]
                value_subtree.append(leaf_subtree)
                tree.append(value_subtree)
            #    CASE 2: no more attributes to select (clash) => handle clash w/majority vote leaf node
            elif len(att_partition) > 0 and len(available_attributes) == 0:
                # finds majority value
                majority_value, value_length = self.majority_vote_lists(att_partition)
                # finds total amount of values
                total_att_instances_length = self.get_value_list_lengths(partitions)
                leaf_subtree = ["Leaf", majority_value, value_length, total_att_instances_length]
                value_subtree.append(leaf_subtree)
                tree.append(value_subtree)
            #    CASE 3: no more instances to partition (empty partition) => backtrack and replace attribute node with majority vote leaf node
            elif len(att_partition) == 0:
                # finds majority value
                majority_value, value_length = self.majority_vote_lists(current_instances)
                # finds total amount of values
                total_instances_length = self.get_value_list_lengths(partitions)
                tree = ["Leaf", majority_value, value_length, total_instances_length]
                # means you are changing your mind, overwrite tree with the majority vote leaf node backtrack
                break
            else:
                # if no cases apply must split again
                subtree = self.tdidt(att_partition, available_attributes.copy())
                value_subtree.append(subtree)
                tree.append(value_subtree)
        return tree

    def fit(self, X_train, y_train):
        """Fits a decision tree classifier to X_train and y_train using the TDIDT
        (top down induction of decision tree) algorithm.

        Args:
            X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since TDIDT is an eager learning algorithm, this method builds a decision tree model
                from the training data.
            Build a decision tree using the nested list representation described in class.
            On a majority vote tie, choose first attribute value based on attribute domain ordering.
            Store the tree in the tree attribute.
            Use attribute indexes to construct default attribute names (e.g. "att0", "att1", ...).
        """
        self.X_train = X_train
        self.y_train = y_train
        # lets stitch together X_train and y_train
        train = [X_train[i] + [y_train[i]] for i in range(len(X_train))]
        # make a copy of header, b/c python is pass by object reference
        # and tdidt will be removing attributes from the available_attributes
        self.header = self.build_header(X_train)
        self.attribute_domains = self.build_domain(X_train, self.header)
        available_attributes = self.header.copy()
        self.tree = self.tdidt(train, available_attributes)

    def tdidt_predict(self, tree, instance):
        """
            Parses through the tree recursively and returns the prediction
        """
        # base case: we are at a leaf node and can return the class prediction
        info_type = tree[0] # "Leaf" or "Attribute"
        if info_type == "Leaf":
            return tree[1]
        # if we are here, we are at an attribute
        # we need to match the instance's value for this attribute
        # to the appropriate subtree
        att_index = self.header.index(tree[1])
        for i in range(2, len(tree)):
            value_list = tree[i]
            # do we have a match with instance for this attribute?
            if value_list[1] == instance[att_index]:
                return self.tdidt_predict(value_list[2], instance)

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_pred = []
        for X in X_test:
            y_pred.append(self.tdidt_predict(self.tree, X))
        return y_pred

    def print_decision_rules_helper(self, tree, instance, first, attribute_names, class_name):
        """
            Recursively creates the decision rule for an instance
        """
        # base case: we are at a leaf node and can return the class prediction
        info_type = tree[0] # "Leaf" or "Attribute"
        if info_type == "Leaf":
            return " THEN " + class_name + " = " + str(tree[1])
        # if we are here, we are at an attribute
        # we need to match the instance's value for this attribute
        # to the appropriate subtree
        att_index = self.header.index(tree[1])
        if attribute_names is None:
            att_text = "att" + str(att_index)
        else:
            att_text = attribute_names[att_index]
        for i in range(2, len(tree)):
            value_list = tree[i]
            if value_list[1] == instance[att_index]:
                if first:
                        return "IF " + att_text + " == " + str(value_list[1])\
                        + self.print_decision_rules_helper(value_list[2], instance, False, attribute_names, class_name)
                else:
                        return " AND " + att_text + " == " + str(value_list[1])\
                        + self.print_decision_rules_helper(value_list[2], instance, False, attribute_names, class_name)

    def domain_permutations(self):
        """Takes the domains and creates all possible permutations out of it"""

        product_values = product(*[v if isinstance(v, (list, tuple)) else [v] for v in self.attribute_domains.values()])
        out = [dict(zip(self.attribute_domains.keys(), values)) for values in product_values]
        return out

    def print_decision_rules(self, attribute_names=None, class_name="class"):
        """Prints the decision rules from the tree in the format
        "IF att == val AND ... THEN class = label", one rule on each line.

        Args:
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes
                (e.g. "att0", "att1", ...) should be used).
            class_name(str): A string to use for the class name in the decision rules
                ("class" if a string is not provided and the default name "class" should be used).
        """
        rules = []
        permutes = self.domain_permutations()
        for dict in permutes:
            instance = list(dict.values())
            # gets result of instance
            output = self.print_decision_rules_helper(self.tree, instance, True, attribute_names, class_name)
            # if not in rules add it
            if output not in rules:
                rules.append(output)
        for r in rules:
            print(r)

    # BONUS method
    def visualize_tree(self, dot_fname, pdf_fname, attribute_names=None):
        """BONUS: Visualizes a tree via the open source Graphviz graph visualization package and
        its DOT graph language (produces .dot and .pdf files).

        Args:
            dot_fname(str): The name of the .dot output file.
            pdf_fname(str): The name of the .pdf output file generated from the .dot file.
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes
                (e.g. "att0", "att1", ...) should be used).

        Notes:
            Graphviz: https://graphviz.org/
            DOT language: https://graphviz.org/doc/info/lang.html
            You will need to install graphviz in the Docker container as shown in class to complete this method.
        """
        pass # TODO: (BONUS) fix this
