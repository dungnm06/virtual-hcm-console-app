from utils.files import *
from utils.strings import *
from common.constant import *
from nlu.question_type_classifier import QuestionTypeClassifier
from nlu.intent_classifier import IntentClassifier
from nlu.model.intent import *
import pandas as pd


class VirtualHCMChatbot(object):
    def __init__(self, config_path):
        print("Stating bot, loading resources...")
        # Application config
        self.config = load_config(config_path)
        # Intent reconizer
        self.intent_reconizer = IntentClassifier(self.config[VNCORENLP], self.config[BERT], self.config)
        self.intent2idx, self.idx2intent = self.intent_reconizer.load()
        # Question type reconizer
        self.question_type_reconizer = QuestionTypeClassifier(self.config[VNCORENLP], self.config[BERT], self.config)
        self.type2idx, self.idx2type = self.question_type_reconizer.load()
        # For dialogue states tracking (list of tuples(intent,action))
        self.state_tracker = []
        # Load intent data
        self.intent_datas = load_from_data(self.config[INTENT_DATA_PATH])

    def decide_action(self, question):
        """Combines intent and question type recognition to decide bot action"""
        pass

    def chat(self, question):
        intent = self.intent_reconizer.predict(question)
        type = self.question_type_reconizer.predict(question)
