import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers, models, optimizers
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from collections import deque
import random
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
        self.model.save(f"{path_prefix}_model.h5")
        self.target_model.save(f"{path_prefix}_target_model.h5")
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
def anomaly_detection_pipeline(df, m=20, episodes=5):
    # --- Step 1: Concatenate datasets ---
    

    # --- Step 2: Map attack categories to numbers ---
    i = 2
    mapping = {}
    for cat in df['attack_cat'].unique():
        if cat == 'Normal':
            mapping[cat] = 0
        elif cat == 'DoS':
            mapping[cat] = 1
        else:
            mapping[cat] = i
            i += 1
    df['attack_cat'] = df['attack_cat'].map(mapping)

    # --- Step 3: Split training and testing ---
    df_dos = df[df['attack_cat'] == 1]
    m = df_dos.shape[0]
    df_non_dos = df[df['attack_cat'] != 1]
    df_normal_sample = df_non_dos[df_non_dos['attack_cat'] == 0].sample(n=m, random_state=42)
    df_train_final = df_non_dos.drop(df_normal_sample.index)
    df_test_final = pd.concat([df_dos, df_normal_sample], axis=0).reset_index(drop=True)

    # --- Step 4: Prepare data ---
    X_train = df_train_final.drop(columns=['attack_cat']).values
    y_train = df_train_final['attack_cat'].values
    X_test = df_test_final.drop(columns=['attack_cat']).values
    y_test = df_test_final['attack_cat'].values

    state_dim = X_train.shape[1]
    action_dim = len(np.unique(y_train))
    agent = DoubleDQNAgent(state_dim, action_dim)

    # --- Step 5: Train Double DQN ---
    total_times = episodes * len(X_train)
    for e in tqdm(range(total_times)):
        i = e % len(X_train)

        state = X_train[i]
        action = agent.act(state)
        reward = 1 if action == y_train[i] else -1
        next_state = X_train[i+1] if i < len(X_train)-1 else np.zeros_like(state)
        done = (i == len(X_train)-1)
        agent.remember(state, action, reward, next_state, done)
        if (e+1) % 10 == 0:
            agent.replay()
        
        if (e+1) % len(X_train) == 0:
            print (f"epoch {e//len(X_train)} complete!")
        
    # 訓練後存檔
    agent.save("dqn_agent")

    # 之後要載入
    # new_agent = DoubleDQNAgent(state_dim, action_dim)
    # new_agent.load("dqn_agent")

    # --- Step 6: Test ---
    y_pred = [agent.act(s) for s in X_test]
    report = classification_report(y_test, y_pred)
    return report

# ------------------ Sample Usage ------------------
df = pd.read_csv("./extra_dataset/combined_after_preprocessing.csv")

df_train, df_test = train_test_split(df, test_size=0.5)

report = anomaly_detection_pipeline(df_train, m=20, episodes=5)
print(report)

