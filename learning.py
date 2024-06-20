import tensorflow as tf
import numpy as np
from keras._tf_keras.keras.models import load_model
from env import DroidRLEnvironment
import pandas as pd

# Tải mô hình đã lưu
loaded_model = load_model("SVM16.h5")

# Sử dụng mô hình để dự đoán hoặc đánh giá
dataset = pd.read_csv("data1.csv", index_col=0)
env = DroidRLEnvironment(dataset, valid_subset_length=16)


def act(state, model, action_size, num_actions):
    selected_actions = []
    for _ in range(num_actions):
        if len(state) == 0:
            state_tensor = tf.constant([[0]], dtype=tf.int32)
        else:
            state_tensor = tf.constant([state], dtype=tf.int32)

        if state_tensor.shape[1] == 0:
            state_tensor = tf.pad(state_tensor, [[0, 0], [0, 1]], "CONSTANT")

        act_values = model(state_tensor).numpy()[0]
        sorted_actions = np.argsort(act_values)[::-1]

        for action in sorted_actions:
            if action not in state:
                state.append(action)
                selected_actions.append(action)
                break

        state.sort()

    return selected_actions


# Ví dụ sử dụng mô hình đã tải để chọn hành động
state = []

selected_actions = act(state, loaded_model, len(env.features), env.valid_subset_length)
optimal_state = [env.actionSpace[idx] for idx in state]
env.state = state
# Tính toán và in ra reward

reward = env.calculate_reward()
print(f"Optimal Feature Subset: {optimal_state}")
print(f"Selected actions: {selected_actions}")
print(f"Final Reward: {reward}")
