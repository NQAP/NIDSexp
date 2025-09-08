import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers, models, optimizers
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from collections import deque
import random
import os
import json
from tqdm import tqdm

# ------------------ Double DQN Agent ------------------
class DoubleDQNAgent:
    def __init__(self, state_dim, action_dim, gamma=0.99, lr=0.001, epsilon=1.0,
                 epsilon_min=0.01, epsilon_decay=0.995, batch_size=64, memory_size=10000,
                 target_update_freq=10):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.lr = lr
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        self.target_update_freq = target_update_freq
        self.train_step = 0

        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()

    def build_model(self):
        model = models.Sequential([
            layers.Input(shape=(self.state_dim,)),
            layers.Dense(64, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(self.action_dim, activation='linear')
        ])
        model.compile(optimizer=optimizers.Adam(self.lr), loss='mse')
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_dim)
        q_values = self.model.predict(state[np.newaxis, :], verbose=0)
        return np.argmax(q_values[0])

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        states, targets = [], []
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state[np.newaxis, :], verbose=0)[0]
            if done:
                target[action] = reward
            else:
                next_action = np.argmax(self.model.predict(next_state[np.newaxis, :], verbose=0)[0])
                target_val = self.target_model.predict(next_state[np.newaxis, :], verbose=0)[0][next_action]
                target[action] = reward + self.gamma * target_val
            states.append(state)
            targets.append(target)
        self.model.fit(np.array(states), np.array(targets), epochs=1, verbose=0)
        self.train_step += 1
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        if self.train_step % self.target_update_freq == 0:
            self.update_target_model()
    # -------- Save & Load --------
    def save(self, path_prefix):
        """Save model and agent state"""
        # 取得目錄
        directory = os.path.dirname(path_prefix)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)  # 建立目錄

        # 存模型
        self.model.save(f"{path_prefix}_model.h5")
        self.target_model.save(f"{path_prefix}_target_model.h5")
        
        # 存 agent state
        np.savez(f"{path_prefix}_agent_state.npz",
                epsilon=self.epsilon,
                train_step=self.train_step)

    def load(self, path_prefix):
        """Load model and agent state"""
        self.model = models.load_model(f"{path_prefix}_model.h5")
        self.target_model = models.load_model(f"{path_prefix}_target_model.h5")
        state = np.load(f"{path_prefix}_agent_state.npz")
        self.epsilon = float(state["epsilon"])
        self.train_step = int(state["train_step"])

# ------------------ Pipeline ------------------
def anomaly_detection_pipeline_binary(df, sample_size=20000, episodes=3, batch_size=64):
    # --- Step 1: Convert attack categories to binary labels ---
    df['attack_cat'] = df['attack_cat'].apply(lambda x: 0 if x == 'Normal' else 1)
    
    # --- Step 2: Train/test split ---
    X = df.drop(columns=['attack_cat']).values
    y = df['attack_cat'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # --- Optional: sample a subset for faster training ---
    if sample_size < len(X_train):
        idx = np.random.choice(len(X_train), sample_size, replace=False)
        X_train = X_train[idx]
        y_train = y_train[idx]
    
    state_dim = X_train.shape[1]
    action_dim = 2  # Normal / Attack
    agent = DoubleDQNAgent(state_dim, action_dim)
    
    # --- Step 3: Train Double DQN ---
    for epoch in tqdm(range(episodes)):
        idxs = np.random.permutation(len(X_train))
        for start in tqdm(range(0, len(X_train), 10)):
            end = start + 10
            batch_idx = idxs[start:end]
            for i in batch_idx:
                state = X_train[i]
                action = agent.act(state)
                reward = 1 if action == y_train[i] else -1
                next_state = X_train[i+1] if i < len(X_train)-1 else np.zeros_like(state)
                done = (i == len(X_train)-1)
                agent.remember(state, action, reward, next_state, done)
            agent.replay()
        print(f"Epoch {epoch+1}/{episodes} complete!")

    agent.save("./model/AIDS_agent/agent_0")
    
    # --- Step 4: Test ---
    y_pred = np.argmax(agent.model.predict(X_test, verbose=0), axis=1)
    report = classification_report(y_test, y_pred)
    print(report)

    num_to_label = {
        0: "Normal",
        1: "Attack"
    }

    # 將 key（數字 label）轉換成文字 label
    report_converted = {}
    for key, value in report.items():
        try:
            int_key = int(key)  # 這裡 key 可能是數字 label
            new_key = num_to_label[int_key]
        except:
            new_key = key  # 其他 key (如 'accuracy', 'macro avg', 'weighted avg')
        report_converted[new_key] = value
    print(report)

    # 儲存 CSV
    rows = []
    for cls, metrics in report_converted.items():
        if isinstance(metrics, dict):
            rows.append({
                "class": cls,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1-score": metrics["f1-score"],
                "support": int(metrics["support"])
            })
        else:
            rows.append({
                "class": cls,
                "precision": "",
                "recall": "",
                "f1-score": "",
                "support": metrics
            })

    df_report = pd.DataFrame(rows)
    df_report.to_csv("./results/AIDS_1.csv", index=False, encoding="utf-8-sig")
    print("CSV 已儲存為 ./results/AIDS_1.csv")
    print(df_report)

    return df_report

# ------------------ Sample Usage ------------------
# df = pd.read_csv("./extra_dataset/combined_2.csv")
# report = anomaly_detection_pipeline_binary(df, sample_size=20000, episodes=3)
# print(report)

