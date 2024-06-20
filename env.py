import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import  pandas as pd
class DroidRLEnvironment:
    def __init__(self, dataset, valid_subset_length):
        self.dataset = dataset.replace('?', np.nan)  # Thay thế '?' bằng NaN
        self.valid_subset_length = valid_subset_length  # Độ dài của tập con hợp lệ
        self.features = dataset.columns[:-1]  # Giả sử nhãn nằm ở cột cuối cùng
        self.actionSpace = {index: column for index, column in enumerate(self.features)}  # Đánh số cho từng tính năng
        self.state = []
        self.reward = 0
        self.classifier = SVC()

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
        # X_train = self.imputer.fit_transform(X_train)
        self.classifier.fit(X_train, y_train)  # Huấn luyện bộ phân loại Random Forest

        accuracy = self.evaluate_accuracy()  # Đánh giá độ chính xác trên tập kiểm tra
        return accuracy

    def evaluate_accuracy(self):
        columns_selected = self.select_columns()
        _, test_data = self.split_dataset(0.8)
        X_test = test_data.loc[:, columns_selected]
        y_test = test_data.iloc[:, -1]
        y_pred = self.classifier.predict(X_test)
        accuracy = accuracy_score(y_test,y_pred)
        return accuracy

    def is_done(self):
        return len(self.state) >= self.valid_subset_length

    def select_columns(self):
        return [self.actionSpace[index] for index in self.state]
