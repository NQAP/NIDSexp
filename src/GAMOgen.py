import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import load_model, Model
from keras.layers import Input, Dense, Softmax, Layer
from keras.layers import BatchNormalization, Concatenate

class SelfAttention(Layer):
    def __init__(self, input_dim, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)
        self.W_query = Dense(input_dim)
        self.W_key = Dense(input_dim)
        self.W_value = Dense(input_dim)
        self.softmax = Softmax(axis=-1)
    
    def call(self, x):
        query = self.W_query(x)
        key = self.W_key(x)
        value = self.W_value(x)
        
        attention_scores = tf.matmul(query, key, transpose_b=True) / tf.math.sqrt(tf.cast(x.shape[-1], tf.float32))
        attention_weights = self.softmax(attention_scores)
        out = tf.matmul(attention_weights, value)
        return out + x  # 殘差連接
    
    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config
    
def build_cfmu(noise_dim=32, label_dim=8):
    noise=Input(shape=(noise_dim,))
    labels=Input(shape=(label_dim,))
    gamoGenInput=Concatenate()([noise, labels])

    x=Dense(32)(gamoGenInput)
    x=SelfAttention(32)(x)
    
    x=Dense(64, activation='relu')(x)
    x=BatchNormalization(momentum=0.9)(x)
    x=Dense(32, activation='relu')(x)
    x=BatchNormalization(momentum=0.9)(x)

    gamoGen=Model([noise, labels], x, name="CFMU")
    gamoGen.summary()
    return Model([noise, labels], x, name="CFMU")

def load_models(num_classes, gen_prefix="./UBSW_NB15_Gamo/gamo_models_500/GenForClass_", gen_postfix="_500_Model"):
    """
    載入所有生成器和 cfmu_gen
    """
    gen = []
    for i in range(num_classes):
        model_path = f"{gen_prefix}{i}{gen_postfix}.h5"
        print(f"載入生成器: {model_path}")
        gen.append(load_model(model_path, compile=False, custom_objects={"SelfAttention": SelfAttention}, safe_mode=False))

    return gen


def generate_samples_per_class(gen, cfmu_gen, num_samples, latDim, feature_dim, target_class, c):
    """
    為指定類別生成樣本
    """
    if num_samples <= 0:
        return np.empty((0, feature_dim)), np.empty((0,), dtype=int)

    randNoise = np.random.normal(0, 1, (num_samples, latDim))
    fakeLabel = np.zeros((num_samples, c))
    fakeLabel[:, target_class] = 1
    cfmu = cfmu_gen.predict([randNoise, fakeLabel], verbose=0)
    fakePoints = gen[target_class].predict(cfmu, verbose=0)
    return fakePoints, np.full((num_samples,), target_class)


def generate_all_classes(gen, cfmu_gen, num_gen_dict, latDim, feature_dim, c, label_mapping=None, save_path="./extra_dataset/generated_data.csv"):
    """
    批量生成所有類別的樣本，並存成 CSV
    num_gen_dict: dict 或 list，key=class id, value=生成數量
                  例如 {0:500, 1:200, 2:300}
    """
    all_data = []
    all_labels = []

    for cls in range(c-1):
        num_samples = num_gen_dict.get(cls, 0) if isinstance(num_gen_dict, dict) else num_gen_dict[cls]
        print(f"類別 {cls} → 生成 {num_samples} 筆")
        fakePoints, fakeLabels = generate_samples_per_class(
            gen, cfmu_gen, num_samples, latDim, feature_dim, cls, c
        )
        all_data.append(fakePoints)
        all_labels.append(fakeLabels)

    all_data = np.vstack(all_data)
    all_labels = np.concatenate(all_labels)

    if label_mapping is not None:
        all_labels = np.vectorize(label_mapping.get)(all_labels)

    df = pd.DataFrame(all_data, columns=[f"f{i}" for i in range(feature_dim)])
    df["label"] = all_labels
    df.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"✅ 生成完成，存成 {save_path}，共 {df.shape[0]} 筆樣本")

    return df

if __name__ == "__main__":

    # 參數

    chi_m = 33393

    chi_dict = {
        "Exploits": 33393,
        "Fuzzers": 18184,
        "DoS": 12264,
        "Reconnaissance": 10491,
        "Analysis": 2000,
        "Backdoor": 1746,
        "Shellcode": 1133,
        "Worms": 130
    }



    c = 8  # 類別數
    latDim = 32
    feature_dim = 42
    num_gen_dict = {
        0: 33263,
        1: 32260,
        2: 31647,
        3: 31393,
        4: 22902,
        5: 21129,
        6: 15209,
        7: 0
    }
    label_mapping = {
        0: "Worms",
        1: "Shellcode",
        2: "Backdoor",
        3: "Analysis",
        4: "Reconnaissance",
        5: "DoS",
        6: "Fuzzers",
        7: "Exploits"
    }

    # 讀取模型
    gen = load_models(num_classes=c-1)

    # 批量生成
    df_generated = generate_all_classes(
        gen=gen,
        cfmu_gen=build_cfmu(),
        num_gen_dict=num_gen_dict,
        latDim=latDim,
        feature_dim=feature_dim,
        c=c,
        label_mapping=label_mapping,
        save_path="generated_data.csv"
    )
