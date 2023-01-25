import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import copy
import random



def make_discrete(column):
    max = np.max(column)
    min = np.min(column)
    inv = (max - min)/5
    for i in range(len(column)):
        if(column[i] <= inv):
            column[i] = 1
        elif(inv < column[i] <= (2*inv)):
            column[i] = 2
        elif((2*inv) < column[i] <= (3*inv)):
            column[i] = 3
        elif((3*inv) < column[i] <= (4*inv)):
            column[i] = 4
        elif(column[i] > (4*inv)):
            column[i] = 5
        else:
            print("Error in make_discrete!")

def discretization(data):
    for column in data.T:
        if(type(column[0]) != int):
           continue
        elif(len(np.unique(column)) > 5):
            make_discrete(column)

def calc_total_entropy(train_data, label, class_list):
    total_row = train_data.shape[0] #the total size of the dataset
    total_entr = 0

    for c in class_list: #for each class in the label
        total_class_count = train_data[train_data[label] == c].shape[0] #number of the class
        total_class_entr = - (total_class_count/total_row)*np.log2(total_class_count/total_row) #entropy of the class
        total_entr += total_class_entr #adding the class entropy to the total entropy of the dataset

    return total_entr

def calc_entropy(feature_value_data, label, class_list):
    class_count = feature_value_data.shape[0]
    entropy = 0

    for c in class_list:
        label_class_count = feature_value_data[feature_value_data[label] == c].shape[0] #row count of class c
        entropy_class = 0
        if label_class_count != 0:
            probability_class = label_class_count/class_count #probability of the class
            entropy_class = - probability_class * np.log2(probability_class)  #entropy
        entropy += entropy_class
    return entropy

def calc_info_gain(feature_name, train_data, label, class_list):
    feature_value_list = train_data[feature_name].unique() #unqiue values of the feature
    total_row = train_data.shape[0]
    feature_info = 0.0

    for feature_value in feature_value_list:
        feature_value_data = train_data[train_data[feature_name] == feature_value] #filtering rows with that feature_value
        feature_value_count = feature_value_data.shape[0]
        feature_value_entropy = calc_entropy(feature_value_data, label, class_list) #calculcating entropy for the feature value
        feature_value_probability = feature_value_count/total_row
        feature_info += feature_value_probability * feature_value_entropy #calculating information of the feature value

    return calc_total_entropy(train_data, label, class_list) - feature_info #calculating information gain by subtracting

def find_most_informative_feature(train_data, label, class_list):
    feature_list = train_data.columns.drop(label) #finding the feature names in the dataset
                                            #N.B. label is not a feature, so dropping it
    max_info_gain = -1
    max_info_feature = None

    for feature in feature_list:  #for each feature in the dataset
        feature_info_gain = calc_info_gain(feature, train_data, label, class_list)
        if max_info_gain < feature_info_gain: #selecting feature name with highest information gain
            max_info_gain = feature_info_gain
            max_info_feature = feature

    return max_info_feature

def generate_sub_tree(feature_name, train_data, label, class_list):
    feature_value_count_dict = train_data[feature_name].value_counts(sort=False) #dictionary of the count of unqiue feature value
    tree = {} #sub tree or node

    for feature_value, count in feature_value_count_dict.iteritems():
        feature_value_data = train_data[train_data[feature_name] == feature_value] #dataset with only feature_name = feature_value

        assigned_to_node = False #flag for tracking feature_value is pure class or not
        for c in class_list: #for each class
            class_count = feature_value_data[feature_value_data[label] == c].shape[0] #count of class c

            if class_count == count: #count of (feature_value = count) of class (pure class)
                tree[feature_value] = c #adding node to the tree
                train_data = train_data[train_data[feature_name] != feature_value] #removing rows with feature_value
                assigned_to_node = True
        if not assigned_to_node: #not pure class
            tree[feature_value] = "?" #as feature_value is not a pure class, it should be expanded further,
                                      #so the branch is marking with ?

    return tree, train_data

def make_tree(root, prev_feature_value, train_data, label, class_list):
    if train_data.shape[0] != 0: #if dataset becomes enpty after updating
        max_info_feature = find_most_informative_feature(train_data, label, class_list) #most informative feature
        tree, train_data = generate_sub_tree(max_info_feature, train_data, label, class_list) #getting tree node and updated dataset
        next_root = None

        if prev_feature_value != None: #add to intermediate node of the tree
            root[prev_feature_value] = dict()
            root[prev_feature_value][max_info_feature] = tree
            next_root = root[prev_feature_value][max_info_feature]
        else: #add to root of the tree
            root[max_info_feature] = tree
            next_root = root[max_info_feature]

        for node, branch in list(next_root.items()): #iterating the tree node
            if branch == "?": #if it is expandable
                feature_value_data = train_data[train_data[max_info_feature] == node] #using the updated dataset
                make_tree(next_root, node, feature_value_data, label, class_list) #recursive call with updated dataset

def id3(train_data_m, label):
    train_data = train_data_m.copy() #getting a copy of the dataset
    tree = {} #tree which will be updated
    class_list = train_data[label].unique() #getting unqiue classes of the label
    make_tree(tree, None, train_data, label, class_list) #start calling recursion
    return tree

def predict(tree, instance):
    if not isinstance(tree, dict): #if it is leaf node
        return tree #return the value
    else:
        root_node = next(iter(tree)) #getting first key/feature name of the dictionary
        feature_value = instance[root_node] #value of the feature
        if feature_value in tree[root_node]: #checking the feature value in current tree node
            return predict(tree[root_node][feature_value], instance) #goto next feature
        else:
            return None

def evaluate(tree, test_data_m):
    res=[]
    flag = 1
    for index, row in test_data_m.iterrows(): #for each row in the dataset
        result = predict(tree, test_data_m.iloc[index]) #predict the row
        if(result is not None):
            res.append(result)
        else:
            res.append("None")
    return res

def dict_generator(indict, pre=None):
    pre = pre[:] if pre else []
    if isinstance(indict, dict):
        for key, value in indict.items():
            if isinstance(value, dict):
                for d in dict_generator(value, pre + [key]):
                    yield d
            elif isinstance(value, list) or isinstance(value, tuple):
                for v in value:
                    for d in dict_generator(v, pre + [key]):
                        yield d
            else:
                yield pre + [key, value]
    else:
        yield pre + [indict]

def sublist(sub_list, test_list):
    if(set(sub_list[:-2]).issubset(set(test_list[:-2]))):
        return True
    return False

def get_twigs(l):
    for i in range(len(l)):
        for j in range(len(l)):
            if(i != j and sublist(l[i], l[j])):
                l[i].clear()
    return list(filter(None, l))

def majority(l):
    yes = l.count("Yes")
    no = l.count("No")
    if(yes > no):
        return "Yes"
    elif(no > yes):
        return "No"
    else:
        return random.choice(["Yes", "No"])

def isLeaf(d):
    for i in list(d.values()):
        if(isinstance(i, dict)):
            return False
    return True

def cut(tree, twig_name):
    if(isinstance(tree, dict)):
        for key, value in tree.items():
            if(twig_name in value and isLeaf(tree[key][twig_name])):
                choice = majority(list(tree[key][twig_name].values()))
                tree[key] = choice
            else:
                cut(value, twig_name)
        return


def prune_tree(tree, last_acc, df, df_test, test):
    current_acc = last_acc
    while(current_acc >= last_acc):
        paths = []
        for i in dict_generator(tree):
            paths.append(i)
        gains = []
        twig_paths = get_twigs(paths)
        twigs = [i[-3] for i in twig_paths]
        for i in twigs:
            gains.append(calc_info_gain(i, df, "Attrition", df["Attrition"].unique()))
        min_index = gains.index(min(gains))
        old_tree = copy.deepcopy(tree)
        cut(tree, twigs[min_index])

        result2 = evaluate(tree, df_test)
        nones2 = []
        truth2 = test[:, 1].copy()

        for i in range(len(result2)):
            if(result2[i] == "None"):
                nones2.append(i)
        nones2.reverse()
        for j in nones2:
            result2.pop(j)
            truth2 = np.delete(truth2, j)
        current_acc = accuracy_score(truth2, result2)
        if(current_acc < last_acc):
            return old_tree
        last_acc = current_acc

def calculate_scores(key, tree, df_test, test):
    dict_res2 = {key: []}
    result2 = evaluate(tree, df_test)
    nones2 = []
    truth2 = test[:, 1].copy()
    for i in range(len(result2)):
        if(result2[i] == "None"):
            nones2.append(i)
    nones2.reverse()
    for j in nones2:
        result2.pop(j)
        truth2 = np.delete(truth2, j)
    dict_res2[list(dict_res2)[0]].append(accuracy_score(truth2, result2))
    dict_res2[list(dict_res2)[0]].append(np.average(precision_score(truth2, result2, average=None)))
    dict_res2[list(dict_res2)[0]].append(np.average(recall_score(truth2, result2, average=None)))
    dict_res2[list(dict_res2)[0]].append(np.average(f1_score(truth2, result2, average=None)))
    return dict_res2

if __name__ == '__main__':
    # PART 1
    # read the csv file
    df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv", encoding='cp1254')
    data = df.to_numpy()  # convert it to numpy array
    column_names = list(df.columns)
    column_names[0] = column_names[0][3:]

    # shuffle the data
    np.random.shuffle(data)

    # discretization of the data
    discretization(data)

    # create dictionary for results
    dict_res = {"fold0": [], "fold1": [], "fold2": [], "fold3": [], "fold4": []}

    # k-fold and train-test split
    Y = data[:,1]
    X = np.delete(data,1,1)
    kf = KFold(n_splits=5, random_state=None)
    fold_num = 0
    for train_index , test_index in kf.split(X):
        data_train, data_test = data[train_index,:], data[test_index,:]
        df_temp = pd.DataFrame(data_train, columns = column_names)
        df_test = pd.DataFrame(data_test, columns = column_names)
        tree = id3(df_temp, 'Attrition')
        result = evaluate(tree, df_test) #evaluating the test dataset
        nones = []
        truth = data_test[:, 1].copy()

        for i in range(len(result)):
            if(result[i] == "None"):
                nones.append(i)
        nones.reverse()
        for j in nones:
            result.pop(j)
            truth = np.delete(truth, j)
        dict_res[list(dict_res)[fold_num]].append(accuracy_score(truth, result))
        dict_res[list(dict_res)[fold_num]].append(np.average(precision_score(truth, result, average=None)))
        dict_res[list(dict_res)[fold_num]].append(np.average(recall_score(truth, result, average=None)))
        dict_res[list(dict_res)[fold_num]].append(np.average(f1_score(truth, result, average=None)))
        fold_num +=1

        # for rule generation
        # for i in dict_generator(tree):
        #     if(i[-1] == "Yes"):
        #         print(i)
    mux = pd.MultiIndex.from_product([['Accuracy', 'Precision', 'Recall', 'F1 Score']])
    df_res = pd.DataFrame.from_dict(dict_res, orient='index', columns = mux)
    print(df_res)

    # END OF PART 1

    # PART 2
    train2 = data[:882, :]
    validate2 = data[882:1176, :]
    test2 = data[1176:, :]


    df_train2 = pd.DataFrame(train2, columns = column_names)
    df_validate2 = pd.DataFrame(validate2, columns = column_names)
    tree2 = id3(df_train2, 'Attrition')
    result2 = evaluate(tree2, df_validate2) #evaluating the test dataset
    nones2 = []
    truth2 = validate2[:, 1].copy()

    dict_res2 = calculate_scores("validate", tree2, df_validate2, validate2)

    # df_res2 = pd.DataFrame.from_dict(dict_res2, orient='index', columns = mux)
    # print(df_res2)

    tree_prune = copy.deepcopy(tree2)
    last_accuracy = dict_res2["validate"][0]

    df_test2 = pd.DataFrame(test2, columns = column_names)
    pruned_tree = prune_tree(tree_prune, last_accuracy, df_train2, df_validate2, validate2)

    dict_res3_1 = calculate_scores("test preprune", tree2, df_test2, test2)
    dict_res3_2 = calculate_scores("test postprune", pruned_tree, df_test2, test2)
    print("old: ", dict_res3_1)
    print("new: ", dict_res3_2)
