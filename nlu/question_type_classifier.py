# BERT imports
from transformers import TFAutoModel, AutoConfig, AutoTokenizer
# model initiations imports
# import tensorflow as tf
from tensorflow.keras.layers import Input, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
# pandas for data import
import pandas as pd
# numpy
import numpy as np
# Text processing utils
from .language_processing import *
from sklearn.preprocessing import MultiLabelBinarizer
# Constant
from common.constant import *
# File utils
from utils.files import *


class QuestionTypeClassifier:
    def __init__(self, bert_name, config):
        self.config = config
        self.model = None
        self.input_sentence_length = None
        self.tokenizer, self.transformer_model, self.bert_config = self.load_bert(bert_name)
        self.label_binarizer = None
        self.type2id = {}
        self.id2type = {
            1: 'what',
            2: 'when',
            3: 'where',
            4: 'who',
            5: 'why',
            6: 'how',
            7: 'yesno'
        }

    def train(self, sequence_length=40,
              batch_size=32, epochs=50, rate=5e-5, epsilon=1e-8):
        ###################################
        # --------- Import data --------- #
        data_path = self.config[QUESTION_TYPE_DATA_PATH]
        self.input_sentence_length = sequence_length
        train_data = {}
        # Import data from csv
        data = pd.read_csv(data_path)
        # Select required columns
        data = data[['Questions', 'IntentType']]
        train_data['question'] = [q for q in data['Questions']]
        tp = [[int(t) for t in types.split(',')] for types in data['IntentType']]
        train_data['type'] = [','.join(self.id2type[i] for i in group) for group in tp]
        train_data = pd.DataFrame(train_data)

        # Remove a row if any of the three columns are missing
        train_data = train_data.dropna()
        # Shuffle datas
        train_data = train_data.sample(frac=1).reset_index(drop=True)

        # Tokenize the input
        x = batch_word_segmentation(train_data['question'].to_list())
        x = self.tokenizer(
            text=x,
            return_tensors='tf',
            add_special_tokens=True,  # add [CLS], [SEP]
            max_length=self.input_sentence_length,  # max length of the text that can go to BERT
            padding='max_length',  # add [PAD] tokens
            return_attention_mask=True,  # add attention mask to not focus on pad tokens
            truncation=True)

        self.label_binarizer = MultiLabelBinarizer()
        # Ready output data for the model
        y_type = [[t.strip().lower() for t in types.split(',')] for types in train_data['type']]
        y_type = self.label_binarizer.fit_transform(y_type)
        print(y_type)
        # get all question types
        types = self.label_binarizer.classes_
        print(types)
        self.type2id, self.id2type = self.types_map_generate(types)
        # # Split into train and test
        # data, data_test = train_test_split(data, train_size=20000, test_size=5000, stratify=data[['num']])

        ###################################
        # ------- Build the model ------- #
        model = self.build_model(len(types))

        ###################################
        # ------- Train the model ------- #
        # Set an optimizer
        optimizer = Adam(
            learning_rate=rate,
            epsilon=epsilon,
            decay=0.01,
            clipnorm=1.0)
        # Set loss and metrics
        loss = {'type': CategoricalCrossentropy(from_logits=True)}
        metric = {'type': CategoricalAccuracy('accuracy')}
        # Compile the model
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metric)
        # Fit the model
        model.fit(
            x={
                'input_ids': x['input_ids'],
                'token_type_ids': x['token_type_ids'],
                'attention_mask': x['attention_mask']
            },
            y={'type': y_type},
            # validation_split=0.2,
            batch_size=batch_size,
            epochs=epochs)
        self.model = model
        # Store map datas to file for future uses
        map_datas = {
            OBJ2IDX: self.type2id,
            IDX2OBJ: self.id2type
        }
        pickle_file(map_datas, QUESTION_TYPE_MAP_FILE_PATH)

    def load(self):
        datapath = self.config[QUESTION_TYPE_MODEL_PATH]
        # Max sentence length
        self.input_sentence_length = self.config[MAX_SENTENCE_LENGTH]
        # Intent maps
        type_maps = unpickle_file(self.config[QUESTION_TYPE_MAP_FILE_PATH])
        self.type2id = type_maps[OBJ2IDX]
        self.id2type = type_maps[IDX2OBJ]
        # Label Binarizer
        self.label_binarizer = MultiLabelBinarizer()
        self.label_binarizer.fit_transform([list(self.id2type.values())])
        # Pretrained model
        print('(QuestionTypeClassifier) Loading pretrained model from: ', datapath)
        self.model = self.build_model(len(self.type2id))
        self.model.load_weights(self.config[QUESTION_TYPE_MODEL_PATH])

        return self.type2id, self.id2type

    def predict(self, input_query):
        # print('Predict:')
        x = word_segmentation(input_query)
        # print(x)
        x = self.tokenizer(
            text=x,
            return_tensors='tf',
            add_special_tokens=True,  # add [CLS], [SEP]
            max_length=self.input_sentence_length,  # max length of the text that can go to BERT
            padding='max_length',  # add [PAD] tokens
            return_attention_mask=True,  # add attention mask to not focus on pad tokens
            truncation=True)
        input_dict = {
            'input_ids': x['input_ids'],
            'token_type_ids': x['token_type_ids'],
            'attention_mask': x['attention_mask']
        }
        # x_tensor = tf.constant(np.concatenate((ids, atm), axis=0))
        # print(x_tensor)
        preds = self.model.predict(input_dict)
        # print(preds['type'])
        threshold = float(self.config[PREDICT_THRESHOLD])
        preds = np.array([[1 if acc > threshold else 0 for acc in p] for p in preds['type']])
        preds = self.label_binarizer.inverse_transform(preds)
        # for predict in preds:
        # print('Predicted types: ', ', '.join(preds[0]))
        return list(preds[0])

    def build_model(self, intents_count):
        ###################################
        # ------- Build the model ------- #
        # TF Keras documentation: https://www.tensorflow.org/api_docs/python/tf/keras/Model
        # Load the MainLayer
        bert = self.transformer_model.layers[0]
        # Build your model input
        input_ids = Input(shape=(self.input_sentence_length,), name='input_ids', dtype='int32')
        token_ids = Input(shape=(self.input_sentence_length,), name='token_type_ids', dtype='int32')
        attention_masks = Input(shape=(self.input_sentence_length,), name='attention_mask', dtype='int32')
        inputs = {'input_ids': input_ids, 'token_type_ids': token_ids, 'attention_mask': attention_masks}
        # Load the Transformers BERT model as a layer in a Keras model
        bert_model = bert(inputs)[1]
        dropout = Dropout(self.bert_config.hidden_dropout_prob, name='pooled_output')
        pooled_output = dropout(bert_model)
        # Output layer
        output_layer = Dense(units=intents_count,
                             kernel_initializer=TruncatedNormal(stddev=self.bert_config.initializer_range),
                             name='question_type', activation='sigmoid')(pooled_output)
        outputs = {'type': output_layer}
        # And combine it all in a model object
        model = Model(inputs=inputs, outputs=outputs, name='QuestionType_BERT_MultiLabel')

        # Take a look at the model
        model.summary()

        return model

    @staticmethod
    def load_bert(bert_name):
        ###################################
        # --------- Setup BERT ---------- #
        # Load transformers config and set output_hidden_states to False
        config = AutoConfig.from_pretrained(bert_name)
        config.output_hidden_states = True
        # Load BERT tokenizer
        tokenizer = AutoTokenizer.from_pretrained(bert_name)
        # Load the Transformers BERT model
        transformer_model = TFAutoModel.from_pretrained(bert_name, config=config)
        return tokenizer, transformer_model, config

    @staticmethod
    def types_map_generate(types=None):
        if types is None:
            types = []
        t2i = {}
        id_count = 1
        for t in types:
            t2i[t] = id_count
            id_count += 1
        i2t = {v: k for k, v in t2i.items()}
        return t2i, i2t
