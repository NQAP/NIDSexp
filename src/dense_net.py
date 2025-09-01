import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Dense, RepeatVector, Lambda, Softmax, Layer, LeakyReLU
from keras.layers import BatchNormalization, Concatenate, Multiply, Dot
from keras.models import Model
from keras.utils import register_keras_serializable
from functools import partial

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

# @register_keras_serializable()
# def gen_process_final_fn(x, dataMinor):
#     # tensordot
#     result = tf.tensordot(x, dataMinor_tf(dataMinor), axes=[[2], [0]])
#     # reduce_mean
#     result = tf.reduce_mean(result, axis=1)
#     return result

# @register_keras_serializable()
# def dataMinor_tf(x):
#     return tf.constant(x, dtype=tf.float32)


# def denseGamoGenCreate(input_dim, numMinor, dataMinor):
#     ip1=Input(shape=(input_dim,))
#     x=Dense(32)(ip1)
#     x=SelfAttention(32)(x)

#     x=Dense(numMinor, activation='softmax')(x)
#     x=RepeatVector(42)(x)

    
#     genProcessFinal = Lambda(
#         gen_process_final_fn, output_shape=(None, 42)
#     )(x, dataMinor)

#     genProcess=Model(ip1, genProcessFinal, name="GAMOGen")
#     return genProcess

@register_keras_serializable(package="Custom")
class GenProcessFinal(Layer):
	def __init__(self, dataMinor, **kwargs):
		super().__init__(**kwargs)
		self.dataMinor = tf.constant(dataMinor, dtype=tf.float32)

	def call(self, inputs):
		result = tf.tensordot(inputs, self.dataMinor, axes=[[2], [0]])
		result = tf.reduce_mean(result, axis=1)
		return result

	def get_config(self):
		config = super().get_config()
		config.update({
			"dataMinor": self.dataMinor.numpy()
		})
		return config

def denseGamoGenCreate(input_dim, numMinor, dataMinor):
	ip1 = Input(shape=(input_dim,))
	x = Dense(32)(ip1)
	# 假設你有 SelfAttention
	x = SelfAttention(32)(x)

	x = Dense(numMinor, activation='softmax')(x)
	x = RepeatVector(42)(x)


	genProcessFinal = GenProcessFinal(dataMinor)(x)

	genProcess = Model(ip1, genProcessFinal, name="GAMOGen")
	return genProcess

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
