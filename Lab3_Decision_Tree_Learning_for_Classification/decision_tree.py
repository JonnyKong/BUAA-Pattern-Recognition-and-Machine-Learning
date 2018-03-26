# -*- coding:utf-8 -*-
import os
import numpy as np
import graphviz as gv

# Construct global name references
name2type = {'Iris-setosa' : 0, 'Iris-versicolor' : 1, 'Iris-virginica': 2}
type2name = {0 : 'Iris-setosa', 1 : 'Iris-versicolor', 2 : 'Iris-virginica'}
attribute2name = {0 : 'Sepal Length', 1 : 'Sepal Width', 2 : 'Petal Length', 3 : 'Petal Width'}

class DecisionTree():
    # Branch nodes are classes and leaf nodes are integers
    class BranchNode():
        def __init__(self, attribute, threshold, left, right):
            self.attribute = attribute
            self.threshold = threshold
            self.left = left
            self.right = right

    # Construct decision tree with dataset including n-1 attributes and 1 flag
    # Build decision tree recursively
    def __init__(self, dataset):
        attributes = set([i for i in range(len(dataset[0]) - 1)])
        self.root = self.createTree(dataset, attributes)

    # Load data into a matrix with flag in numbers
    @staticmethod
    def load_data():
        fileName = 'bezdekIris.txt'
        filePath = os.path.join('data', fileName)
        data = []
        with open(filePath, 'r') as f:
            for line in f.readlines():
                line = line.split(sep = ',')
                tmp = [float(i) for i in line[:-1]]
                tmp.append(name2type[line[-1][:-1]])
                data.append(tmp)
        return data

    # Calculate entropy of certain dataset
    @staticmethod
    def calculateEntropy(dataset):
        flag2num = dict()
        total = len(dataset)
        entropy = 0
        for data in dataset:
            flag = data[-1]
            if flag in flag2num.keys():
                flag2num[flag] += 1
            else:
                flag2num[flag] = 1
        for flag in flag2num:
            p = flag2num[flag] / total
            entropy -= p * np.log2(p)
        return entropy

    # Calculate entropy gain W.R.T. a certain attribute and threshold
    @staticmethod
    def calculateEntropyGain(dataset, attribute, threshold):
        # Sort data according to attribute
        dataset.sort(key = lambda x: x[attribute])
        entropy = DecisionTree.calculateEntropy(dataset)
        # Find the threshold position
        thres_pos = 0
        while (dataset[thres_pos][attribute] < threshold):
            thres_pos += 1
        entropy -= DecisionTree.calculateEntropy(dataset[:thres_pos]) * thres_pos / len(dataset)
        entropy -= DecisionTree.calculateEntropy(dataset[thres_pos:]) * (len(dataset) - thres_pos) / len(dataset)
        return entropy

    # Calculate optimal threshold for a particular attribute
    @staticmethod
    def calculateThreshold(dataset, attribute):
        # Sort data according to attribute
        dataset.sort(key = lambda x : x[attribute])
        threshold_list = [data[attribute] for data in dataset]
        entropy_gain_max = 0
        # for threshold in threshold_list:
        #     entropy_gain_current = DecisionTree.calculateEntropyGain(dataset, attribute, threshold)
        #     if entropy_gain_current > entropy_gain_max:
        #         entropy_gain_max = entropy_gain_current
        #         final_threshold = threshold
        for i in range(1, len(threshold_list)):
            entropy_gain_current = DecisionTree.calculateEntropyGain(dataset, attribute, threshold_list[i])
            if entropy_gain_current > entropy_gain_max:
                entropy_gain_max = entropy_gain_current
                final_threshold = np.mean([threshold_list[i], threshold_list[i - 1]])
        return final_threshold

    # Select attribute with maximum entropy gain and its threshold
    @staticmethod
    def selectAttribute(dataset, attributes):
        maxEntropyGain = 0
        threshold_selected = 0
        for attribute in attributes:
            threshold = DecisionTree.calculateThreshold(dataset, attribute)
            entropyGain = DecisionTree.calculateEntropyGain(dataset, attribute, threshold)
            if entropyGain > maxEntropyGain:
                maxEntropyGain = entropyGain
                attribute_selected = attribute
                threshold_selected = threshold
        return (attribute_selected, threshold_selected)

    # Check majority flag in dataset and return
    @staticmethod
    def majorityFlag(dataset):
        flag2num = dict()
        for data in dataset:
            if data[-1] in flag2num.keys():
                flag2num[data[-1]] += 1
            else:
                flag2num[data[-1]] = 1
        maxCnt = 0
        for flag in flag2num.keys():
            if flag2num[flag] > maxCnt:
                maxCnt = flag2num[flag]
                maxFlag = flag
        return maxFlag

    # Create decision tree recursively
    @staticmethod
    def createTree(dataset, attributes):
        flagList = [data[-1] for data in dataset]
        # If all samples belong to same flag, return this flag
        if flagList.count(flagList[0]) == len(flagList):
            return flagList[0]
        # No more attributes, return majority flag
        if len(attributes) == 0:
            return DecisionTree.majorityFlag(dataset)
        # Selected attribute with maximum entropy gain
        (attributeSelected, threshold) = DecisionTree.selectAttribute(dataset, attributes)
        # Generate two branches with new dataset and attributes
        dataset_l = [data for data in dataset if data[attributeSelected] < threshold]
        dataset_h = [data for data in dataset if data[attributeSelected] >= threshold]
        # Remove current attribute
        attributes.remove(attributeSelected)
        if len(dataset_l) > 0:
            # Create sub-tree recursively
            left = DecisionTree.createTree(dataset_l, attributes)
        else:
            # Assign child node with majority flag of parent node
            left = DecisionTree.majorityFlag(dataset)
        if len(dataset_h) > 0:
            # Create sub-tree recursively
            right = DecisionTree.createTree(dataset_h, attributes)
        else:
            # Assign child node with majority flag of parent node
            right = DecisionTree.majorityFlag(dataset)
        node = DecisionTree.BranchNode(attributeSelected, threshold, left, right)
        attributes.add(attributeSelected)
        return node

    # Visualize the decision tree
    def visualize(self):
        # Depth first search decision tree, add node/edge to set
        def dfs(root, graph, nodeSet, edgeSetL, edgeSetR):
            assert isinstance(root, DecisionTree.BranchNode)
            title = attribute2name[root.attribute] + '>=' + str(root.threshold) + '?'
            graph.node(title)
            # Plot the left tree
            if isinstance(root.left, DecisionTree.BranchNode):
                dfs(root.left, graph, nodeSet, edgeSetL, edgeSetR)
                title_left = attribute2name[root.left.attribute] + '>=' + str(root.left.threshold) + '?'
                edgeSetL.add((title, title_left))
            else:
                nodeSet.add(type2name[root.left])
                edgeSetL.add((title, type2name[root.left]))
            # Plot the right tree
            if isinstance(root.right, DecisionTree.BranchNode):
                dfs(root.right, graph, nodeSet, edgeSetL, edgeSetR)
                title_right = attribute2name[root.right.attribute] + '>=' + str(root.right.threshold) + '?'
                edgeSetR.add((title, title_right))
            else:
                nodeSet.add(type2name[root.right])
                edgeSetR.add((title, type2name[root.right]))
        # Update styles to graph
        def apply_styles(graph, styles):
            graph.graph_attr.update(('graph' in styles and styles['graph']) or {})
            graph.node_attr.update(('nodes' in styles and styles['nodes']) or {})
            graph.edge_attr.update(('edges' in styles and styles['edges']) or {})
            return graph

        self.graph = gv.Graph(format = 'svg')
        nodeSet = set()
        edgeSetL = set()
        edgeSetR = set()
        dfs(self.root, self.graph, nodeSet, edgeSetL, edgeSetR)
        # Plot all elements in set
        for node in nodeSet:
            self.graph.node(node)
        for node in edgeSetL:
            self.graph.edge(*node, **{'label' : 'Yes'})
        for node in edgeSetR:
            self.graph.edge(*node, **{'label' : 'No'})
        # Choose graph styles
        styles = {
            'graph': {
                'label': 'Decision Tree for Iris Dataset',
                'fontname' : 'Helvetica',
                'fontsize': '16',
                'fontcolor': 'black',
                'bgcolor': '#D1EEEE',
            },
            'nodes': {
                'fontname': 'Lucida Grande',
                'shape': 'egg',
                'fontcolor': 'white',
                'color': 'black',
                'style': 'filled',
                'fillcolor': '#EE2C2C',
            },
            'edges': {
                'style': 'dashed',
                'color': 'black',
                'arrowhead': 'open',
                'fontname': 'Courier',
                'fontsize': '12',
                'fontcolor': 'black',
            }
        }
        self.graph = apply_styles(self.graph, styles)
        filename = self.graph.render(filename = os.path.join('img', 'graph'))

if __name__ == '__main__':
    dataset = DecisionTree.load_data()
    tree = DecisionTree(dataset)
    tree.visualize()