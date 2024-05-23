import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
import  pandas as pd
class DroidRLEnvironment:
    def __init__(self, dataset, valid_subset_length):
        self.dataset = dataset.replace('?', np.nan)  # Thay thế '?' bằng NaN
        self.valid_subset_length = valid_subset_length  # Độ dài của tập con hợp lệ
        self.features = dataset.columns[:-1]  # Giả sử nhãn nằm ở cột cuối cùng
        self.actionSpace = {index: column for index, column in enumerate(self.features)}  # Đánh số cho từng tính năng
        self.state = []
        self.reward = 0
        self.classifier = RandomForestClassifier(max_leaf_nodes=32, random_state=10)  # Sử dụng Random Forest để phân loại
        self.imputer = SimpleImputer(strategy='mean')  # Sử dụng SimpleImputer để điền giá trị thiếu

    def reset(self):
        self.state = []
        self.reward = 0
        return self.state

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

        # Xử lý giá trị thiếu trong dữ liệu huấn luyện
        X_train = self.imputer.fit_transform(X_train)
        self.classifier.fit(X_train, y_train)  # Huấn luyện bộ phân loại Random Forest

        accuracy = self.evaluate_accuracy()  # Đánh giá độ chính xác trên tập kiểm tra
        if accuracy < 0.6:
            accuracy = -1.0
        return accuracy

    def evaluate_accuracy(self):
        columns_selected = self.select_columns()
        _, test_data = self.split_dataset(0.8)
        X_test = test_data.loc[:, columns_selected]
        y_test = test_data.iloc[:, -1]

        # Xử lý giá trị thiếu trong dữ liệu kiểm tra
        X_test = self.imputer.transform(X_test)
        accuracy = self.classifier.score(X_test, y_test)
        return accuracy

    def is_done(self):
        return len(self.state) >= self.valid_subset_length

    def select_columns(self):
        return [self.actionSpace[index] for index in self.state]

    def print(self):
        print(self.reward, self.state)