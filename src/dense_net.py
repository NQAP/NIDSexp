import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Dense, RepeatVector, Lambda, Softmax, Layer, LeakyReLU, Attention, Reshape, Flatten
from keras.layers import BatchNormalization, Concatenate, Multiply, Dot
from keras.models import Model
from keras.utils import register_keras_serializable
from functools import partial
import numpy as np

class SelfAttention(Layer):
    def __init__(self, units, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)
        self.W_query = Dense(units)
        self.W_key = Dense(units)
        self.W_value = Dense(units)
        self.softmax = Softmax(axis=-1)
        self.units = units  # ⭐ 這行很重要，記住 units
    
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
    
# class DataAwareAttention(Layer):
#     def __init__(self, units, dataMinor, **kwargs):
#         super(DataAwareAttention, self).__init__(**kwargs)
#         self.W_query = Dense(units)
#         self.W_key = Dense(units)
#         self.W_value = Dense(units)
#         self.softmax = Softmax(axis=-1)
#         self.units = units
        
#         # 將 dataMinor 平均作為 context
#         self.data_context = Dense(units)(tf.constant(np.mean(dataMinor, axis=0, keepdims=True), dtype=tf.float32))

#     def call(self, x):
#         # x shape: (batch_size, input_dim)
#         query = self.W_query(x)                               # (batch_size, units)
#         key = self.W_key(x) + self.data_context  # (batch_size, units)
#         value = self.W_value(x) + self.data_context  # (batch_size, units)
        
#         attention_scores = tf.matmul(query, key, transpose_b=True) / tf.math.sqrt(tf.cast(tf.shape(x)[-1], tf.float32))
#         attention_weights = self.softmax(attention_scores)
#         out = tf.matmul(attention_weights, value)
#         return out + x
#     def get_config(self):
#         config = super().get_config()
#         config.update({"units": self.units})
#         return config
    
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


class DataMinorAttention(Layer):
    def __init__(self, units, dataMinor, **kwargs):
        super(DataMinorAttention, self).__init__(**kwargs)
        self.units = units
        # 將少數類別資料存成常數
        self.dataMinor = tf.constant(dataMinor, dtype=tf.float32)

    def build(self, input_shape):
        self.W_query = self.add_weight(shape=(input_shape[-1], self.units),
                                       initializer='glorot_uniform',
                                       trainable=True,
                                       name='W_query')
        self.W_key = self.add_weight(shape=(self.dataMinor.shape[-1], self.units),
                                     initializer='glorot_uniform',
                                     trainable=True,
                                     name='W_key')
        self.W_value = self.add_weight(shape=(self.dataMinor.shape[-1], self.units),
                                       initializer='glorot_uniform',
                                       trainable=True,
                                       name='W_value')
        super(DataMinorAttention, self).build(input_shape)

    def call(self, x):
        # x: (batch, input_dim)
        # key/value: 少數類別資料
        query = tf.matmul(x, self.W_query)                 # (batch, units)
        key = tf.matmul(self.dataMinor, self.W_key)        # (N_minor, units)
        value = tf.matmul(self.dataMinor, self.W_value)    # (N_minor, units)

        # 注意力分數
        scores = tf.matmul(query, key, transpose_b=True) / tf.math.sqrt(tf.cast(self.units, tf.float32))
        weights = tf.nn.softmax(scores, axis=-1)          # (batch, N_minor)
        out = tf.matmul(weights, value)                   # (batch, units)
        return out

    def get_config(self):
        config = super(DataMinorAttention, self).get_config()
        config.update({
            'units': self.units,
            'dataMinor': self.dataMinor.numpy()
        })
        return config
    
@register_keras_serializable(package="Custom")
class GenProcessFinal(Layer):
    def __init__(self, dataMinor, **kwargs):
        super().__init__(**kwargs)
        self.dataMinor = tf.constant(dataMinor, dtype=tf.float32)

    def call(self, x):
        # print (x.shape)
        # print(self.dataMinor.shape)
        result = tf.tensordot(x, self.dataMinor, axes=[[2],[0]])  # 或者 tensordot
        # print (result.shape)
        result = tf.reduce_mean(result, axis=1)
        # print (result.shape)
        return result

    def get_config(self):
        config = super().get_config()
        config.update({"dataMinor": self.dataMinor.numpy()})
        return config
    
    def compute_output_shape(self, input_shape):
    # 假設 input_shape=(batch, 137)，self.data.shape=(137,)
        return (input_shape[0], self.dataMinor.shape[-1])



def denseGamoGenCreate(input_dim, numMinor, dataMinor):
    ip = Input(shape=(input_dim,)) # (None,32)
    x = Dense(32)(ip)
    # x = Reshape((32,1))(x)
    # data = Reshape((42, numMinor))(data)
    # print(data.shape)
    # print(x.shape)
    x = SelfAttention(32)(x) # (None,32)
    # x = Flatten()(x)
    # print(x.shape)
    # x = Reshape((32,-1))(x)
    x = Dense(numMinor)(x)
    # print(x.shape)
    # print(x.shape)
    x = RepeatVector(42)(x)
    # 最後用自訂 Layer 處理 dataMinor（可以像之前的 GenProcessFinal）
    x = GenProcessFinal(dataMinor)(x)

    gen = Model(ip, x, name="GAMOGen")
    return gen

# @register_keras_serializable(package="Custom")
# class GenProcessFinal(Layer):
# 	def __init__(self, dataMinor, **kwargs):
# 		super().__init__(**kwargs)
# 		self.dataMinor = tf.constant(dataMinor, dtype=tf.float32)

# 	def call(self, inputs):
# 		result = tf.tensordot(inputs, self.dataMinor, axes=[[2], [0]])
# 		result = tf.reduce_mean(result, axis=1)
# 		return result

# 	def get_config(self):
# 		config = super().get_config()
# 		config.update({
# 			"dataMinor": self.dataMinor.numpy()
# 		})
# 		return config

# def denseGamoGenCreate(input_dim, numMinor, dataMinor):
# 	ip1 = Input(shape=(input_dim,))
# 	x = Dense(32)(ip1)
    
# 	# 假設你有 SelfAttention
# 	x = DataAwareAttention(32, dataMinor=dataMinor)(x)

# 	x = Dense(numMinor, activation='softmax')(x)
# 	x = RepeatVector(42)(x)

# 	genProcessFinal = GenProcessFinal(dataMinor)(x)

# 	genProcess = Model(ip1, genProcessFinal, name="GAMOGen")
# 	return genProcess

def denseDisCreate():
    imIn=Input(shape=(42,))
    labels=Input(shape=(8,))
    disInput=Concatenate()([imIn, labels])

    x=Dense(42, activation="softmax")(disInput)

    x=SelfAttention(42)(x)

    x=Dense(64)(x)
    x=LeakyReLU(alpha=0.1)(x)

    x=Dense(32)(x)
    x=LeakyReLU(alpha=0.1)(x)

    disFinal=Dense(1, activation='sigmoid')(x)

    dis=Model([imIn, labels], disFinal)
    dis.summary()
    return dis

def denseMlpCreate():

    imIn=Input(shape=(42,))

    x=Dense(42, activation="softmax")(imIn)

    x=SelfAttention(42)(x)

    x=Dense(64)(x)
    x=LeakyReLU(alpha=0.1)(x)

    x=Dense(32)(x)
    x=LeakyReLU(alpha=0.1)(x)

    mlpFinal=Dense(8, activation='softmax')(x)

    mlp=Model(imIn, mlpFinal)
    mlp.summary()
    return mlp
