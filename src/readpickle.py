import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_metrics_from_pickles(folder):
    """
    從資料夾中讀取所有 pickle 檔，並整理成一個 DataFrame
    """
    all_metrics = []
    steps = []

    for i in range(50):
        filename = folder + str(i*100) + "/metrics_" + str(i*100) + ".pkl"
        print(filename)
        step = i*100  # 從檔名抓 step 數字
        steps.append(step)

        with open(filename, "rb") as f:
            metrics = pickle.load(f)

        # 扁平化，只抓 scalar (array 另外存)
        row = {
            "step": step,
            "acsa_tr": metrics["acsaSaveTr"][-1],
            "gm_tr": metrics["gmSaveTr"][-1],
            "acc_tr": metrics["accSaveTr"][-1],
            "acsa_ts": metrics["acsaSaveTs"][-1],
            "gm_ts": metrics["gmSaveTs"][-1],
            "acc_ts": metrics["accSaveTs"][-1],
        }
        all_metrics.append(row)

    df = pd.DataFrame(all_metrics).sort_values("step").reset_index(drop=True)
    return df

def load_confmat_tpr(folder, max_steps=50, mode="train"):
    """
    將每個 step 的 Confusion Matrix 和 TPR 分別存成 DataFrame
    mode: "train" 或 "test"
    """
    conf_list = []
    tpr_list = []

    for i in range(max_steps):
        step = i * 100
        filename = os.path.join(folder+f"{step}", f"metrics_{step}.pkl")
        print (filename)
        if not os.path.exists(filename):
            continue

        with open(filename, "rb") as f:
            metrics = pickle.load(f)

        if mode == "train":
            conf = metrics["confMatSaveTr"][-1]
            tpr = metrics["tprSaveTr"][-1]
        else:
            conf = metrics["confMatSaveTs"][-1]
            tpr = metrics["tprSaveTs"][-1]

        # Flatten Confusion Matrix (row-wise)
        conf_flat = conf.flatten()
        conf_row = {"step": step}
        for idx, val in enumerate(conf_flat):
            conf_row[f"conf_{idx}"] = val
        conf_list.append(conf_row)

        # TPR
        tpr_row = {"step": step}
        for idx, val in enumerate(tpr):
            tpr_row[f"tpr_{idx}"] = val
        tpr_list.append(tpr_row)

    df_conf_tr = pd.DataFrame(conf_list)
    print(df_conf_tr.columns)  # 確認 step 欄位存在
    df_conf_tr = df_conf_tr.sort_values("step").reset_index(drop=True)

    df_tpr_tr = pd.DataFrame(tpr_list)
    print(df_tpr_tr.columns)
    df_tpr_tr = df_tpr_tr.sort_values("step").reset_index(drop=True)
    df_conf = pd.DataFrame(conf_list).sort_values("step").reset_index(drop=True)
    df_tpr = pd.DataFrame(tpr_list).sort_values("step").reset_index(drop=True)
    return df_conf, df_tpr

# 使用範例
folder = "./UBSW_NB15_Gamo_Ver2/gamo_models_"
df_conf_tr, df_tpr_tr = load_confmat_tpr(folder, mode="train")
df_conf_ts, df_tpr_ts = load_confmat_tpr(folder, mode="test")

print(df_conf_tr.head())
print(df_tpr_tr.head())

df_metrics = load_metrics_from_pickles(folder)

print(df_metrics.head())


def plot_metrics(df):
    metrics = ["acsa", "gm", "acc"]
    for m in metrics:
        plt.figure(figsize=(8,5))
        plt.plot(df["step"], df[f"{m}_tr"], label=f"{m.upper()} Train")
        plt.plot(df["step"], df[f"{m}_ts"], label=f"{m.upper()} Test")
        plt.xlabel("Step")
        plt.ylabel(m.upper())
        plt.title(f"{m.upper()} over Steps")
        plt.legend()
        plt.grid(True)
        plt.show()

# 繪圖
plot_metrics(df_metrics)

# -----------------------------
# 1️⃣ TPR 隨 step 變化
# -----------------------------
def plot_tpr(df_tpr, title="TPR per Class"):
    steps = df_tpr["step"].values
    class_cols = [col for col in df_tpr.columns if col.startswith("tpr_")]

    plt.figure(figsize=(12,6))
    for col in class_cols:
        plt.plot(steps, df_tpr[col].values, label=col)
    plt.xlabel("Step")
    plt.ylabel("TPR")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

# 訓練集與測試集 TPR
plot_tpr(df_tpr_tr, title="Train TPR per Class")
plot_tpr(df_tpr_ts, title="Test TPR per Class")

# -----------------------------
# 2️⃣ 混淆矩陣熱力圖
# -----------------------------
def plot_confusion_matrix_over_steps(df_conf, num_classes, step_interval=100, title="Confusion Matrix"):
    """
    每 step 畫一次熱力圖，可以設定 step_interval 篩選間隔
    """
    steps = df_conf["step"].values
    for idx, step in enumerate(steps):
        if step % step_interval != 0:
            continue
        conf_flat = df_conf.iloc[idx, 1:].values  # 排除 step 欄位
        conf_mat = conf_flat.reshape((num_classes, num_classes))
        
        plt.figure(figsize=(8,6))
        sns.heatmap(conf_mat, annot=True, fmt=".2f", cmap="Blues")
        plt.title(f"{title} at Step {step}")
        plt.xlabel("Predicted Class")
        plt.ylabel("True Class")
        plt.show()

# 訓練集和測試集混淆矩陣
num_classes = 8  # 根據你的分類數改
plot_confusion_matrix_over_steps(df_conf_tr, num_classes, step_interval=500, title="Train Confusion Matrix")
plot_confusion_matrix_over_steps(df_conf_ts, num_classes, step_interval=500, title="Test Confusion Matrix")