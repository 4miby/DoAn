import pandas as pd
import numpy as np
from env import DroidRLEnvironment
import tensorflow as tf
from replay_memory import MemoryBuffer
from decision_network import  DecisionNetwork
import random
if __name__ == "__main__":
    # Khởi tạo môi trường và các tham số
    dataset = pd.read_csv("drebin-215-dataset-5560malware-9476-benign.csv")
    env = DroidRLEnvironment(dataset, valid_subset_length=30)
    input_dim = len(env.features)
    embedding_dim = 128
    hidden_dim = 256
    action_size = len(env.features)
    E = 1000  # Tổng số tập huấn luyện
    p = 0.99  # Hằng số cho việc cập nhật epsilon
    gamma = 0.99  # Tỷ lệ chiết khấu
    epsilon = 1.0  # Tỷ lệ khám phá
    epsilon_min = 0.01
    learning_rate = 0.001
    batch_size = 16
    M = 100  # Số tập warm-up cho bộ nhớ replay
    Network_Learn_Frequency = 10  # Tần suất cập nhật mạng chính
    Sync_Frequency = 30  # Tần suất đồng bộ hóa mạng mục tiêu

    # Khởi tạo mô hình
    decision_network = DecisionNetwork(input_dim, embedding_dim, hidden_dim, action_size)
    model = decision_network.build_model()
    target_model = decision_network.build_model()
    target_model.set_weights(model.get_weights())
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    memory = MemoryBuffer()

    def update_epsilon(episode, E, p):
        return 1 - (episode / E * p)


    def act(state, model, epsilon, action_size):
        if np.random.rand() <= epsilon:
            return random.randrange(action_size)

        # Xử lý trạng thái rỗng
        if len(state) == 0:
            state = [0]
        state = tf.constant([state], dtype=tf.int32)
        act_values = model(state).numpy()[0]

        # Chọn hành động không trùng lặp
        sorted_actions = np.argsort(act_values)[::-1]  # Sắp xếp hành động theo giá trị giảm dần
        for action in sorted_actions:
            if action not in state.numpy()[0]:
                return action
        return sorted_actions[0]  # Nếu không tìm thấy, trả về hành động giá trị cao nhất


    def replay(model, target_model, memory, batch_size, gamma, optimizer, action_size):
        if memory.get_num() < memory.get_min():
            return

        state, action, reward, next_state, done = memory.sample(batch_size)
        state = tf.constant(state, dtype=tf.int32)
        next_state = tf.constant(next_state, dtype=tf.int32)
        reward = tf.constant(reward, dtype=tf.float32)
        done = tf.constant(done, dtype=tf.float32)

        # Công thức 2: Tính toán Q(s', a') bằng mạng mục tiêu
        target_next = tf.reduce_max(target_model(next_state), axis=1)
        # Công thức 3: Tính toán giá trị mục tiêu cho trạng thái hiện tại
        target = reward + (1 - done) * gamma * target_next

        with tf.GradientTape() as tape:
            q_values = model(state)
            indices = tf.range(batch_size)
            q_values = tf.reduce_sum(tf.one_hot(action, action_size) * q_values, axis=1)
            loss = tf.keras.losses.mean_squared_error(target, q_values)

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # In ra reward trung bình của batch hiện tại
        avg_reward = np.mean(reward)
        print(f"Average reward: {avg_reward:.4f}")


    # Warm-up bộ nhớ replay
    for m in range(M):
        state = env.reset()
        for _ in range(env.valid_subset_length):
            action = random.randrange(action_size)
            next_state, reward, done = env.step(action)
            memory.add(state, action, reward, next_state, done)
            state = next_state
            if done:
                break

    # Vòng lặp huấn luyện
    for episode in range(E):
        state = env.reset()
        total_reward = 0
        all_states = []
        for f in range(env.valid_subset_length):
            previous_state = state.copy()
            action = act(state, model, epsilon, action_size)
            next_state, reward, done = env.step(action)
            memory.add(previous_state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            all_states.append(state.copy())

            if done:
                break

        if episode % Network_Learn_Frequency == 0:
            replay(model, target_model, memory, batch_size, gamma, optimizer, action_size)

        if episode % Sync_Frequency == 0:
            target_model.set_weights(model.get_weights())

        epsilon = update_epsilon(episode, E, p)
        epsilon = max(epsilon, epsilon_min)

        print(
            f"Episode: {episode + 1}/{E}, Epsilon: {epsilon:.4f}, Total Reward: {total_reward:.4f}, States: {all_states}")
    # Đánh giá bộ tính năng tối ưu
    state = []
    for f in range(env.valid_subset_length):
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
                state.sort()
                break

    optimal_state = [env.actionSpace[idx] for idx in state]
    final_reward = env.calculate_reward()
    print(f"Optimal Feature Subset: {optimal_state}, Final Reward: {final_reward}")
    print(state)
    # Lưu mô hình sau khi huấn luyện xong
    model.save("model.h5", save_format='tf')
    print(f"Model has been saved")