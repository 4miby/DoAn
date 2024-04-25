import numpy as np
from sklearn.ensemble import RandomForestClassifier
import  pandas as pd
class DroidRLEnvironment:
    def __init__(self, dataset, valid_subset_length):
        self.dataset = dataset
        self.valid_subset_length = valid_subset_length # Độ dài của tập subset
        self.features = dataset.columns[:-1] # Ta sẽ giả sử nhãn label nằm ở cuối
        self.actionSpace = {index:column for index, column in enumerate(self.features)}  # Đánh số cho từng feature
        self.state = []
        self.reward = 0
        self.classifier = RandomForestClassifier(max_leaf_nodes=32, random_state=10)  # Sử dụng random forest để phân loại

    # Reset lại môi trường và trả lại state rỗng
    def reset(self):
        self.state = []
        self.reward = 0
        return self.state

    # Thực hiện action được chọn
    def step(self, action):
        self.state.append(action)
        self.state.sort()
        self.reward = self.calculate_reward()
        done = self.is_done()
        return self.state, self.reward, done

    def split_dataset(self, train_ratio):
        num_samples = self.dataset.shape[0]
        num_train = int(num_samples * train_ratio)
        train_data = self.dataset[:num_train]
        test_data = self.dataset[num_train:]
        return train_data, test_data

    def calculate_reward(self):
        columns_selected = self.select_columns()
        train_data, _ = self.split_dataset(0.8)
        X_train = train_data.loc[:, columns_selected]
        y_train = train_data.iloc[:, -1]
        self.classifier.fit(X_train, y_train)  # Train the Random Forest classifier
        accuracy = self.evaluate_accuracy()  # Evaluate the accuracy on the test set
        if accuracy < 0.6 :
            accuracy = -1.0
        return accuracy

    def evaluate_accuracy(self):
        columns_selected = self.select_columns()
        _, test_data = self.split_dataset(0.8)
        X_test = test_data.loc[:, columns_selected]
        y_test = test_data.iloc[:, -1]
        accuracy = self.classifier.score(X_test, y_test)
        return accuracy

    def is_done(self):
        # Check if the agent has selected enough features according to the defined valid subset length
        return len(self.state) >= self.valid_subset_length

    def select_columns(self):
        columns = []
        for index in self.state:
            column = self.actionSpace.get(index)
            columns.append(column)
        return columns


    def print(self):
        print(self.reward, self.state)