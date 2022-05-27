import pandas as pd
import tensorflow as tf
from tensorflow import keras
from .layers import NumericalFeatureEmbedding, CategoricalFeatureEmbedding, MLP,TransformerEncoder


# Implement model classes to facilitate building
# input preprocessing and evaluation.
# We use Functional API of keras to build models

class TabularTransformer(object):

    def __init__(self):
        # embeddings parameters
        self.emb_cat_dim = 32
        self.emb_num_dim = 32

        # transformer parameters
        self.tr_depth = 6
        self.tr_heads = 8
        self.tr_ff_dim = 16
        self.tr_attn_dropout = 0.1
        self.tr_ff_dropout = 0.1
        
        # mlp parameters
        self.mlp_hidden_dims = (4,2)
        self.mlp_activation = keras.layers.ReLU()

    @classmethod
    def from_config(cls, config):
        return cls(**config)


    def build(self):
        """Construct the model 
        using functional API of keras.
        """
        self.model = None
        return self

    def _get_features_dtypes(self, X:pd.DataFrame):
        pass
    

    def _df_to_dataset(self, X:pd.DataFrame, y:pd.DataFrame=None, shuffle=True, batch_size=32):
        X = X.copy()
        y = y.values if y is not None else None
 
        X = {key: value[:,tf.newaxis] for key, value in X.items()}

        ds = tf.data.Dataset.from_tensor_slices((dict(X), y))
        if shuffle:
            ds = ds.shuffle(buffer_size=len(X.shape[0]))
        ds = ds.batch(batch_size)
        ds = ds.prefetch(batch_size)

        return ds

    def _get_normalization_layer(self, name, dataset):
        # Create a Normalization layer for the feature.
        normalizer = keras.layers.Normalization(axis=None)
        # Prepare a Dataset that only yields the feature.
        feature_ds = dataset.map(lambda x, y: x[name])
        # Learn the statistics of the data.
        normalizer.adapt(feature_ds)
        return normalizer

    def _get_category_encoding_layer(self, name, dataset, dtype, max_tokens=None):
        # Create a layer that turns strings into integer indices.
        if dtype == 'string':
            index = keras.layers.StringLookup(max_tokens=max_tokens)
        # Otherwise, create a layer that turns integer values into integer indices.
        else:
            index = keras.layers.IntegerLookup(max_tokens=max_tokens)

        # Prepare a `tf.data.Dataset` that only yields the feature.
        feature_ds = dataset.map(lambda x, y: x[name])
        # Learn the set of possible values and assign them a fixed integer index.
        index.adapt(feature_ds)
        return lambda feature: index(feature)


    def preprocess(self, X:pd.DataFrame, y:pd.DataFrame=None, 
                shuffle=True, batch_size=32, normalize=True):
        """Prepare inputs in pd.DataFrame format
        to train-eval-predict the Model. 
        """
        dataset = None
        return dataset

    
    def fit(self, X, y, eval_size=0.2, epochs=100, early_stop=True):
        # preprocess X, y
        # build model
        # compile model
        # fit model
        pass

    def evaluate(self, X, y):
        pass

    def predict(self, X):
        pass



    

    


def build_tab_transformer(categories, num_continuous, dim,
            dim_out, depth, heads, ff_dim=16, mlp_hidden_mults = (4, 2),
            mlp_act = None, attn_dropout = 0.1, ff_dropout = 0.1):

    num_categories = len(categories)  

    cat_input = keras.layers.Input(shape=(num_categories,), name='cat_inputs')
    num_input = keras.layers.Input(shape=(num_continuous,), name='num_inputs')

    x_num = keras.layers.LayerNormalization()(num_input)
        
    # --> categorical inputs
    num_unique_categories = sum(categories)
    total_categories = num_unique_categories 

    # embedding
    x_cat = keras.layers.Embedding(input_dim=total_categories, output_dim=dim)(cat_input)

    for _ in range(depth+1):
        x_cat = TransformerEncoder(embed_dim=dim, num_heads=heads, ff_dim=ff_dim, ff_dropout=ff_dropout, attn_dropout=attn_dropout)(x_cat)
    x_cat = keras.layers.Flatten()(x_cat)


    mlp_input = keras.layers.Concatenate()([x_num,x_cat])

    input_size = (dim * num_categories) + num_continuous
    l = input_size // 8

    hidden_dimensions = list(map(lambda t: l * t, mlp_hidden_mults))
    all_dimensions = [input_size, *hidden_dimensions, dim_out]

    if mlp_act:
        mlp = MLP(all_dimensions, activation=mlp_act)
    else:
        mlp = MLP(all_dimensions, activation=keras.layers.ReLU())

    logits = mlp(mlp_input)
    
    outputs=keras.activations.sigmoid(logits)

    model = keras.Model(inputs=[num_input, cat_input], outputs=outputs)

    return model


def build_tab_transformer_v2(categories, num_continuous, dim,
            dim_out,depth,heads,ff_dim,mlp_hidden_mults = (4, 2),
            mlp_act = None, attn_dropout = 0.1,ff_dropout = 0.1):

    num_categories = len(categories)  

    cat_input = keras.layers.Input(shape=(num_categories,), name='cat_inputs')
    num_input = keras.layers.Input(shape=(num_continuous,), name='num_inputs')

    x_num = keras.layers.LayerNormalization()(num_input)
        
    x_cat = CategoricalFeatureEmbedding(cardinalities=categories,dim_token=dim)(cat_input)
    for _ in range(depth+1):
        x_cat = TransformerEncoder(embed_dim=dim, num_heads=heads, ff_dim=ff_dim, ff_dropout=ff_dropout)(x_cat)
    x_cat = keras.layers.Flatten()(x_cat)

    mlp_input = keras.layers.Concatenate()([x_num,x_cat])

    input_size = (dim * num_categories) + num_continuous
    l = input_size // 8

    hidden_dimensions = list(map(lambda t: l * t, mlp_hidden_mults))
    all_dimensions = [input_size, *hidden_dimensions, dim_out]

    if mlp_act:
        mlp = MLP(all_dimensions, activation=mlp_act)
    else:
        mlp = MLP(all_dimensions, activation=keras.layers.ReLU())

    logits = mlp(mlp_input)
    
    outputs=keras.activations.sigmoid(logits)

    model = keras.Model(inputs=[num_input, cat_input], outputs=outputs)

    return model


def build_tab_transformer_v3(categories, num_continuous, dim,
            dim_out,depth,heads,mlp_hidden_mults = (4, 2),
            mlp_act = None, attn_dropout = 0.1,ff_dropout = 0.1):

    num_categories = len(categories)  

    cat_input = keras.layers.Input(shape=(num_categories,), name='cat_inputs')
    num_input = keras.layers.Input(shape=(num_continuous,), name='num_inputs')
    x_num = keras.layers.LayerNormalization()(num_input)


    x_num_emb = NumericalFeatureEmbedding(num_features=num_continuous,dim_token=dim)(num_input)
    x_cat_emb = CategoricalFeatureEmbedding(cardinalities=categories,dim_token=dim)(cat_input)
    x_emb = keras.layers.Concatenate()([x_num_emb,x_cat_emb])
    for _ in range(depth+1):
        x_emb = TransformerEncoder(dim, heads, dim, attn_dropout=attn_dropout, ff_dropout=ff_dropout)(x_emb)
    x_emb = keras.layers.Flatten()(x_emb)

    mlp_input = keras.layers.Concatenate()([x_num,x_emb])

    input_size = (dim * num_categories) + (dim * num_continuous) + num_continuous
    # l = input_size // 8
    l=input_size
    hidden_dimensions = list(map(lambda t: l * t, mlp_hidden_mults))
    # hidden_dimensions = list(mlp_hidden_mults)

    all_dimensions = [*hidden_dimensions, dim_out]

    if mlp_act:
        mlp = MLP(all_dimensions, activation=mlp_act)
    else:
        mlp = MLP(all_dimensions, activation=keras.layers.ReLU())

    logits = mlp(mlp_input)
    
    outputs=keras.activations.sigmoid(logits)

    model = keras.Model(inputs=[num_input, cat_input], outputs=outputs)

    return model





class TabTransformer(keras.Model):


    def __init__(self, 
            categories,
            num_continuous,
            embed_dim,
            ff_dim,
            dim_out,
            depth,
            heads,
            mlp_hidden_mults = (4, 2),
            mlp_act = None,
            attn_dropout = 0.1,
            ff_dropout = 0.1,
            normalize_continuous = True
            ):

        super(TabTransformer, self).__init__()
        
        # continuous inputs
        self.normalize_continuous = normalize_continuous
        if normalize_continuous:
            self.continuous_normalization = keras.layers.LayerNormalization()

        self.num_continuous = num_continuous

        # categorical inputs
        self.num_categories = len(categories)
        self.num_unique_categories = sum(categories)
        total_categories = self.num_unique_categories 

        # embedding
        self.embedding_layer = CategoricalFeatureEmbedding(cardinalities=categories, dim_token=embed_dim)
        
        # transformers
        self.transformers = []
        for _ in range(0,depth):
            self.transformers.append(TransformerEncoder(
                embed_dim=embed_dim, 
                num_heads=heads, 
                ff_dim=ff_dim, 
                attn_dropout=attn_dropout,
                ff_dropout=ff_dropout
                ))
        self.flatten_transformer_output = keras.layers.Flatten()

        # --> MLP
        self.pre_mlp_concatenation = keras.layers.Concatenate()

        input_size = (embed_dim * self.num_categories) + num_continuous
        l = input_size 
        # // 8

        hidden_dimensions = list(map(lambda t: l * t, mlp_hidden_mults))
        all_dimensions = [input_size, *hidden_dimensions, dim_out]

        if mlp_act:
            self.mlp = MLP(all_dimensions, activation=mlp_act)
        else:
            self.mlp = MLP(all_dimensions)

    def call(self, continuous_inputs, categorical_inputs):

        # --> continuous
        if self.normalize_continuous:
            continuous_inputs = self.continuous_normalization(continuous_inputs)

        # --> categorical
        categorical_inputs = self.embedding_layer(categorical_inputs)
        
        for transformer in self.transformers:
            categorical_inputs = transformer(categorical_inputs)
        contextual_embedding = self.flatten_transformer_output(categorical_inputs)

        # --> MLP
        mlp_input = self.pre_mlp_concatenation([continuous_inputs, contextual_embedding])

        return self.mlp(mlp_input)



