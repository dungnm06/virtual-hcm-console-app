# BERT imports
from transformers import TFBertModel, BertConfig, AutoTokenizer
# model initiations imports
import tensorflow as tf
from tensorflow.keras.layers import Input, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
# pandas for data import
import pandas as pd
# numpy
import numpy as np
# Text processing utils
from utils.text_processing import batch_word_segmentation
from utils.files import *
# Constant
from common.constant import *
# For intents dictionary creation
from collections import Counter
import random


class IntentClassifier:
    def __init__(self, vncorenlp, bert_name, config):
        self.config = config
        self.rdrsegmenter = vncorenlp
        self.model = None
        self.input_sentence_length = None
        self.tokenizer, self.transformer_model, self.bert_config = self.load_bert(bert_name)
        self.intent_to_idx = {}
        self.idx_to_intent = {}

    def train(self, sequence_length=40,
              batch_size=32, epochs=50, rate=5e-5, epsilon=1e-8):
        ###################################
        # --------- Import data --------- #
        # Import data from csv
        data_path = self.config[INTENT_DATA_PATH]
        data = pd.read_csv(data_path)
        # Select required columns
        data = data[['Intent', 'Questions']]
        # Remove a row if any of the three columns are missing
        data = data.dropna()
        # Train data prepare
        x = []
        y = []
        # Split into x and y list from dataset
        for questions, intent in zip(data['Questions'], data['Intent']):
            q = questions.split('#')
            for q1 in q:
                x.append(q1.strip())
                y.append(intent)
        # Intent mapping for future uses
        intents_count = Counter(y)
        self.intent_to_idx = {intent: i for i, intent in enumerate(intents_count)}
        self.idx_to_intent = {i: intent for i, intent in enumerate(intents_count)}
        # Intent list to index number for training
        y = [self.intent_to_idx[intent] for intent in y]
        # Shuffle train data
        temp = list(zip(x, y))
        random.shuffle(temp)
        x, y = zip(*temp)
        x = list(x)
        y = tf.constant(list(y))
        # Tokenize the input
        x = batch_word_segmentation(self.rdrsegmenter, x)
        x = self.tokenizer(
            text=x,
            return_tensors='tf',
            add_special_tokens=True,  # add [CLS], [SEP]
            max_length=sequence_length,  # max length of the text that can go to BERT
            padding='max_length',  # add [PAD] tokens
            return_attention_mask=True,  # add attention mask to not focus on pad tokens
            truncation=True)
        # # Split into train and test
        # data, data_test = train_test_split(data, train_size=20000, test_size=5000, stratify=data[['num']])

        ###################################
        # ------- Build the model ------- #
        # TF Keras documentation: https://www.tensorflow.org/api_docs/python/tf/keras/Model
        # Load the MainLayer
        bert = self.transformer_model.layers[0]
        # Build your model input
        input_ids = Input(shape=(sequence_length,), name='input_ids', dtype='int32')
        token_ids = Input(shape=(sequence_length,), name='token_type_ids', dtype='int32')
        attention_masks = Input(shape=(sequence_length,), name='attention_mask', dtype='int32')
        inputs = {'input_ids': input_ids, 'token_type_ids': token_ids, 'attention_mask': attention_masks}
        # Load the Transformers BERT model as a layer in a Keras model
        bert_model = bert(inputs)[1]
        dropout = Dropout(self.bert_config.hidden_dropout_prob, name='pooled_output')
        pooled_output = dropout(bert_model)
        # Output layer
        intent = Dense(units=len(intents_count),
                       kernel_initializer=TruncatedNormal(stddev=self.bert_config.initializer_range), name='intent')(
            pooled_output)
        outputs = {'intent': intent}
        # And combine it all in a model object
        model = Model(inputs=inputs, outputs=outputs, name='Intent_Classifier_BERT_MultiClass')

        # Take a look at the model
        model.summary()

        ###################################
        # ------- Train the model ------- #
        # Hyperparameters
        # Set an optimizer
        optimizer = Adam(
            learning_rate=rate,
            epsilon=epsilon,
            decay=0.01,
            clipnorm=1.0)
        # Set loss and metrics
        loss = {'intent': SparseCategoricalCrossentropy(from_logits=True)}
        metric = {'intent': SparseCategoricalAccuracy('accuracy')}
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
            y={'intent': y},
            batch_size=batch_size,
            epochs=epochs)
        self.model = model
        # Store map datas to file for future uses
        map_datas = {
            OBJ2IDX: self.intent_to_idx,
            IDX2OBJ: self.idx_to_intent
        }
        pickle_file(map_datas, self.config[INTENT_MAP_PATH])

    def load(self):
        datapath = self.config[INTENT_MODEL_PATH]
        # Pretrained model
        print('(IntentClassifier) Loading pretrained model from: ', datapath)
        self.model = tf.keras.models.load_model(datapath)
        # Max sentence length
        self.input_sentence_length = self.model.layers[0].output_shape[0][1]
        # Intent maps
        intent_maps = unpickle_file(self.config[INTENT_MAP_PATH])
        self.intent_to_idx = intent_maps[OBJ2IDX]
        self.idx_to_intent = intent_maps[IDX2OBJ]
        return self.intent_to_idx, self.idx_to_intent

    def predict(self, input_query):
        print('Predict:')
        x = batch_word_segmentation(self.rdrsegmenter, input_query)
        print(x)
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
        pred = self.model.predict(input_dict)
        print(pred)
        intent_idx = np.argmax(pred['intent'], axis=1)[0]
        pred_intent = self.idx_to_intent[intent_idx]
        print("Intent: ", pred_intent)
        return pred_intent

    @staticmethod
    def load_bert(bert_name):
        ###################################
        # --------- Setup BERT ---------- #
        # Load transformers config and set output_hidden_states to False
        config = BertConfig.from_pretrained(bert_name)
        config.output_hidden_states = True
        # Load BERT tokenizer
        tokenizer = AutoTokenizer.from_pretrained(bert_name)
        # Load the Transformers BERT model
        transformer_model = TFBertModel.from_pretrained(bert_name, config=config)
        return tokenizer, transformer_model, config
