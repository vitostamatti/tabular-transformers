from tensorflow import keras
import tensorflow as tf
import numpy as np




class NumericalFeatureEmbedding(keras.layers.Layer):
    """Transforms continuous features to tokens (embeddings).
    For one feature, the transformation consists of two steps:
    * the feature is multiplied by a trainable vector
    * another trainable vector is added
    Note that each feature has its separate pair of trainable vectors, i.e. the vectors
    are not shared between features.

    Args:
        num_features: the number of continuous (scalar) features
        dim_token: the size of one token
        use_bias: if `False`, then the transformation will include only multiplication.
            **Warning**: :code:`use_bias=False` leads to significantly worse results for
            Transformer-like (token-based) architectures.
        initialization: initialization policy for parameters. Must be one of
            :code:`['uniform', 'normal']`. Let :code:`s = d ** -0.5`. Then, the
            corresponding distributions are :code:`Uniform(-s, s)` and :code:`Normal(0, s)`.
    """

    def __init__(
        self,
        num_features: int,
        dim_token: int,
        use_bias: bool=True,
        initialization: str='uniform',
    ) -> None:

        super(NumericalFeatureEmbedding, self).__init__()

        self.num_features = num_features
        self.dim_token = dim_token
        self.use_bias = use_bias
        self.initialization = initialization

        d_sqrt_inv = 1 / np.sqrt(self.dim_token)
        if self.initialization=="uniform":
            self.initializer = tf.random_uniform_initializer(
                minval=-d_sqrt_inv, maxval=d_sqrt_inv, seed=None
            )
        elif self.initialization=="normal":
            self.initializer = tf.random_normal_initializer(
                mean=0.0, stddev=d_sqrt_inv, seed=None
            )

    def build(self, input_shape):  
        self.w = tf.Variable(initial_value=self.initializer(shape=(self.num_features, self.dim_token), 
            dtype=tf.float32), trainable=True)

        self.b = tf.Variable(self.initializer(shape=(self.num_features, self.dim_token), dtype=tf.float32),
         trainable=True) if self.use_bias else None

    def call(self, x):
        x = self.w[tf.newaxis] * x[..., tf.newaxis]
        if self.use_bias:
            x = x + self.b
        return x

    def get_config(self):
        config = super(NumericalFeatureEmbedding, self).get_config()
        config.update({
            "num_features": self.num_features,
            "dim_token": self.dim_token,
            "use_bias": self.use_bias,
            "initialization": self.initialization,
            })
        return config


class CategoricalFeatureEmbedding(keras.layers.Layer):
    """
    Transforms categorical features to tokens (embeddings).
    The module efficiently implements a collection of `keras.layers.Embedding` (with
    optional biases).

    Args:
        cardinalities: the number of distinct values for each feature. For example,
            :code:`cardinalities=[3, 4]` describes two features: the first one can
            take values in the range :code:`[0, 1, 2]` and the second one can take
            values in the range :code:`[0, 1, 2, 3]`.
        d_token: the size of one token.
        bias: if `True`, for each feature, a trainable vector is added to the
            embedding regardless of feature value. The bias vectors are not shared
            between features.
        initialization: initialization policy for parameters. Must be one of
            :code:`['uniform', 'normal']`. Let :code:`s = d ** -0.5`. Then, the
            corresponding distributions are :code:`Uniform(-s, s)` and :code:`Normal(0, s)`.
    """

    def __init__(
        self,
        cardinalities,
        dim_token:int,
        use_bias:bool=True,
        initialization:str='uniform',
    ) -> None:

        super(CategoricalFeatureEmbedding, self).__init__()
        assert cardinalities, 'cardinalities must be non-empty'
        assert dim_token > 0, 'd_token must be positive'
        
        self.cardinalities = cardinalities
        self.dim_token = dim_token
        self.use_bias = use_bias
        self.initialization =initialization

        self.category_offsets = tf.cumsum(tf.constant([0] + cardinalities[:-1], dtype=tf.int64), axis=0)
        

        d_sqrt_inv = 1 / np.sqrt(self.dim_token)
        if self.initialization=="uniform":
            self.initializer = tf.random_uniform_initializer(
                minval=-d_sqrt_inv, maxval=d_sqrt_inv, seed=None
            )
        elif self.initialization=="normal":
            self.initializer = tf.random_normal_initializer(
                mean=0.0, stddev=d_sqrt_inv, seed=None
            )
        
        self.embeddings = keras.layers.Embedding(
            input_dim=sum(cardinalities), output_dim=dim_token,
            embeddings_initializer=self.initializer
        )


    def build(self, input_shape):

        self.bias = tf.Variable(initial_value=self.initializer(shape=(len(self.cardinalities), self.dim_token), 
            dtype=tf.float32),
            trainable=True) if self.use_bias else None

        # self.embeddings.build(input_shape=input_shape)
        # weights = tf.Variable(initial_value=initializer(shape=(sum(self.cardinalities), self.dim_token), dtype=tf.float32),
        # trainable=True)

        # self.embeddings.set_weights([weights])


    def call(self, x):
        x = self.embeddings(x + self.category_offsets)
        if self.bias is not None:
            x = x + self.bias
        return x

    def get_config(self):
        config = super(NumericalFeatureEmbedding, self).get_config()
        config.update({
            "cardinalities": self.cardinalities,
            "dim_token": self.dim_token,
            "use_bias": self.use_bias,
            "initialization": self.initialization,
            })
        return config




class FeatureEmbedding(keras.layers.Layer):
    """
    Combines `NumericalFeatureEmbedding` and `CategoricalFeatureEmbedding`.
    The layer transforms continuous and categorical features to tokens (embeddings).

    Args:
        n_num_features: the number of continuous features.
        cat_cardinalities: the number of unique values for each feature.
        d_token: the size of one token.
    """

    def __init__(
        self,
        num_features: int,
        cardinalities,
        dim_token: int,
    ) -> None:

        super(FeatureEmbedding, self).__init__()
        assert num_features >= 0, 'n_num_features must be non-negative'
        assert (
            num_features and cardinalities
        ), 'n_num_features and cat_cardinalities must be positive/non-empty'
        
        self.initialization = 'uniform'

        self.num_embedding = NumericalFeatureEmbedding(
                num_features=num_features,
                dim_token=dim_token,
                use_bias=True,
                initialization=self.initialization,
            )
            
        self.cat_embedding = CategoricalFeatureEmbedding(
                cardinalities, dim_token, 
                use_bias=True, initialization=self.initialization
            )
            

    def call(self, x_num, x_cat):
        """Perform the forward pass.
        Args:
            x_num: continuous features. Must be presented if :code:`n_num_features > 0`
                was passed to the constructor.
            x_cat: categorical features (see `CategoricalFeatureTokenizer.forward` for
                details). Must be presented if non-empty :code:`cat_cardinalities` was
                passed to the constructor.
        """
    
        x = []
        x.append(self.num_embedding(x_num))
        x.append(self.cat_embedding(x_cat))

        return tf.concat(x, dim=1)

    def get_config(self):
        config = super(NumericalFeatureEmbedding, self).get_config()
        config.update({
            "num_features": self.num_features,
            "cardinalities": self.cardinalities,
            "dim_token": self.dim_token,
            })
        return config



class MLP(keras.layers.Layer):
    def __init__(self, dims, activation):
        super(MLP, self).__init__()

        dims_pairs = list(zip(dims[:-1], dims[1:]))
        layers = []
        for idx, (dim_in, dim_out) in enumerate(dims_pairs):
            is_last = idx >= (len(dims_pairs) - 1)
            linear = keras.layers.Dense(units=dim_out, input_shape = (dim_in,))
            layers.append(linear)

            if is_last:
                continue

            act = activation
 
            layers.append(act)

        self.mlp = keras.Sequential(layers)

    def call(self, inputs):
        return self.mlp(inputs)

    def get_config(self):
        config = super(MLP, self).get_config()
        config.update({
            "dims": self.dims,
            "activation": self.activation,
            })
        return config


class TransformerEncoder(keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, attn_dropout=0.1, ff_dropout=0.1):
        super(TransformerEncoder, self).__init__()

        self.attn = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim, dropout=attn_dropout)
        
        self.ffn = keras.Sequential(
            [keras.layers.Dense(ff_dim, activation="relu"), keras.layers.Dense(embed_dim)]
        )
        # batch-layer
        self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = keras.layers.Dropout(ff_dropout)
        self.dropout2 = keras.layers.Dropout(ff_dropout)

    def call(self, inputs, training):
        attn_output = self.attn(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super(TransformerEncoder, self).get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim":self.ff_dim,
            "attn_dropout":self.attn_dropout,
            "ff_dropout":self.ff_dropout
            })
        return config