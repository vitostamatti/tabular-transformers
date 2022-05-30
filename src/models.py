import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow import keras
from keras.utils import np_utils

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from .layers import NumericalFeatureEmbedding, CategoricalFeatureEmbedding, MLP,TransformerEncoder
from .metrics import MultiClassMeanRecall, MultiClassMeanPrecision, MultiClassConfusionMatrix
# Implement model classes to facilitate building
# input preprocessing and evaluation.
# We use Functional API of keras to build models


class TabularTransformer(object):

    def __init__(
            self, 
            emb_dim=32, 
            tr_depth=6, 
            tr_heads=8, 
            tr_ff_dim=16, 
            tr_attn_dropout=0.1, 
            tr_ff_dropout=0.1,
            mlp_hidden_mults=[4,2],
            mlp_activation='relu'
        ):

        # embeddings parameters
        self.emb_dim = emb_dim

        # transformer parameters
        self.tr_depth = tr_depth
        self.tr_heads = tr_heads
        self.tr_ff_dim = tr_ff_dim
        self.tr_attn_dropout = tr_attn_dropout
        self.tr_ff_dropout = tr_ff_dropout
        
        # mlp parameters
        self.mlp_hidden_mults = mlp_hidden_mults
        self.mlp_activation = keras.layers.ReLU() if mlp_activation == 'relu' else mlp_activation

        self.model = None

    @classmethod
    def from_config(cls, config):
        return cls(**config)


    def _compile_model(self, learning_rate=1e-4):
        opt = keras.optimizers.Adam(learning_rate=learning_rate)
        
        if self.output_dim > 1:
            loss = 'categorical_crossentropy'
            # loss = tf.keras.losses.CategoricalCrossentropy()
            metrics = [
                keras.metrics.CategoricalAccuracy(name='accuracy'),
                MultiClassMeanRecall(num_classes=self.output_dim),
                MultiClassMeanPrecision(num_classes=self.output_dim),
                MultiClassConfusionMatrix(num_classes=self.output_dim)   
            ]
        else:
            # loss = tf.keras.losses.MeanSquaredError()
            loss = 'mean_squared_error'
            metrics = [
                keras.metrics.MAE(name='mae'),
                keras.metrics.MAPE(name='mape'),
                keras.metrics.MSE(name='mse'),
            ]

        self.model.compile(optimizer=opt, loss=loss, metrics=metrics)


    def _fit_model(self, train_ds, val_ds, epochs=1):
        early_stop_callback = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10
        )
        callbacks = [early_stop_callback]
        history = self.model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)
        return history

    def _build_model(self, inputs):
        """
        Construct the model using functional API of keras.
        Requiers information about input layers.
        """

        all_inputs = inputs['inputs']
        cat_features = tf.keras.layers.Concatenate(name='categorical_features')(inputs['cat_features'])
        num_features = tf.keras.layers.Concatenate(name='numerical_features')(inputs['num_features'])

        x_num = keras.layers.LayerNormalization(name='layer_norm_num')(num_features)

        x_num_emb = NumericalFeatureEmbedding(num_features=self.n_num_features,dim_token=self.emb_dim)(num_features) 
        x_cat_emb = CategoricalFeatureEmbedding(cardinalities=self.cardinalities,dim_token=self.emb_dim)(cat_features)

        x_tr = tf.concat([x_num_emb,x_cat_emb], axis=1)

        for _ in range(0,self.tr_depth):
            x_tr = TransformerEncoder(
                embed_dim=(self.emb_dim), 
                num_heads=self.tr_heads, 
                ff_dim=self.tr_ff_dim, 
                ff_dropout=self.tr_ff_dropout,
                attn_dropout=self.tr_attn_dropout
                )(x_tr)

        x_emb = keras.layers.Flatten(name="flatten_embeddings")(x_tr)

        mlp_input = keras.layers.Concatenate(name="concat_num_emb")([x_num,x_emb])

        input_size = (self.emb_dim * len(self.cardinalities)) + ((self.emb_dim * self.n_num_features)) + self.n_num_features
        l = input_size // 8

        hidden_dimensions = list(map(lambda t: l * t, self.mlp_hidden_mults))

        # dim out sale de y
        all_dimensions = [input_size, *hidden_dimensions, self.output_dim]

        if self.mlp_activation is not None:
            mlp = MLP(all_dimensions, activation=self.mlp_activation)
        else:
            mlp = MLP(all_dimensions, activation=keras.layers.ReLU())

        mlp_outputs = mlp(mlp_input)
        
        if self.output_dim > 1:
            outputs = keras.activations.softmax(mlp_outputs)
        else:
            outputs = mlp_outputs

        self.model = keras.Model(inputs=all_inputs, outputs=outputs)
        
        return self


    def _fit_transform_label(self, y:pd.DataFrame):
        if y.dtype in [np.int16,np.int32,np.int64]:
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(np.ravel(y))
            self.output_dim = len(self.label_encoder.classes_)
        else:
            self.label_encoder = None
            self.output_dim = 1

        return self._transform_label(y)

    def _transform_label(self, y):
        if self.label_encoder is not None:
            encoded_y=self.label_encoder.transform(np.ravel(y))
            return np_utils.to_categorical(encoded_y)
        else:
            return y

    def _get_normalization_layer(self, name, dataset):
        normalizer = keras.layers.Normalization(axis=None,name=f"normalization_{name}")
        feature_ds = dataset.map(lambda x, y: x[name])
        normalizer.adapt(feature_ds)
        return normalizer

    def _get_category_encoding_layer(self, name, dataset, dtype, max_tokens=None):
    
        if dtype == 'string':
            index = keras.layers.StringLookup(max_tokens=max_tokens, name=f"encoding_{name}")
      
        else:
            index = keras.layers.IntegerLookup(max_tokens=max_tokens, oov_token=9999, name=f"encoding_{name}")

        feature_ds = dataset.map(lambda x, y: x[name])
        index.adapt(feature_ds)
        return index

    def _get_input_layers(self, dataset):
        all_inputs = []
        encoded_num_features = []
        encoded_cat_features = []

        cardinalities = []
        n_num_features = 0

        features_dtypes = dataset.element_spec[0]

        for name in features_dtypes:
            if features_dtypes[name].dtype in [tf.float64, tf.float32]:
                input_layer = tf.keras.Input(shape=(1,), name=name)
                prepro_layer = self._get_normalization_layer(name, dataset)
                encoded_feature = prepro_layer(input_layer)
                encoded_num_features.append(encoded_feature)

                n_num_features = n_num_features+1

            elif features_dtypes[name].dtype in [tf.int32, tf.int64]:
                input_layer = tf.keras.Input(shape=(1,), name=name,dtype='int64')
                prepro_layer = self._get_category_encoding_layer(name=name,dataset=dataset,dtype='integer')
                encoded_feature = prepro_layer(input_layer)
                encoded_cat_features.append(encoded_feature)

                cardinalities.append(len(prepro_layer.get_vocabulary()))

            elif features_dtypes[name].dtype in [tf.string]:
                input_layer = tf.keras.Input(shape=(1,), name=name, dtype='string')
                prepro_layer = self._get_category_encoding_layer(name=name,dataset=dataset, dtype='string')
                encoded_feature = prepro_layer(input_layer)
                encoded_cat_features.append(encoded_feature)

                cardinalities.append(len(prepro_layer.get_vocabulary()))

            all_inputs.append(input_layer)

        self.cardinalities = cardinalities
        self.n_num_features = n_num_features

        return {"inputs":all_inputs, "num_features":encoded_num_features, "cat_features":encoded_cat_features}


    def _preprocess_predict(self, X:pd.DataFrame):
        inputs = {name: tf.convert_to_tensor([value]) for name, value in X.items()}
        return tf.data.Dataset.from_tensor_slices(dict(inputs))

    def _preprocess_evaluate(self, X:pd.DataFrame, y:pd.DataFrame, batch_size=32):
        X = {key: np.array(value)[:,tf.newaxis] for key, value in X.items()}
        y = self._transform_label(y.values)
        ds = tf.data.Dataset.from_tensor_slices((dict(X), y))
        ds = ds.batch(batch_size)
        return ds

    def _preprocess_fit(self, X:pd.DataFrame, y:pd.DataFrame, eval_size=0.2, shuffle=True, batch_size=32):
        X, y = X.copy(), y.copy()
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=eval_size, random_state=1213, shuffle=shuffle)
        
        y_train = self._fit_transform_label(y_train.values)
        y_val = self._transform_label(y_val.values)
        
        X_train = {key: np.array(value)[:,tf.newaxis] for key, value in X_train.items()}
        train_ds = tf.data.Dataset.from_tensor_slices((dict(X_train), y_train))

        X_val = {key: np.array(value)[:,tf.newaxis] for key, value in X_val.items()}
        val_ds = tf.data.Dataset.from_tensor_slices((dict(X_val), y_val))

        train_ds = train_ds.batch(batch_size)
        val_ds = val_ds.batch(batch_size)

        return train_ds, val_ds



    def fit(self, X, y, eval_size=0.1, learning_rate=1e-4, epochs=1,shuffle=True, batch_size=32):
        # convert to dataset
        train_ds, val_ds = self._preprocess_fit(X, y, eval_size=eval_size, shuffle=shuffle, batch_size=batch_size)
    
        if self.model is None:
            # prepare input layers
            inputs = self._get_input_layers(train_ds)
            # build model    
            self._build_model(inputs)

        # compile model
        self._compile_model(learning_rate=learning_rate)
        # fit model
        self._fit_model(train_ds, val_ds, epochs=epochs)

        return self

    def evaluate(self, X:pd.DataFrame, y:pd.DataFrame, batch_size=32):
        val_ds = self._preprocess_evaluate(X, y, batch_size=batch_size)
        
        return self.model.evaluate(val_ds)

    def predict(self, X:pd.DataFrame):
        inputs = self._preprocess_predict(X)
        return self.model.predict(inputs)
