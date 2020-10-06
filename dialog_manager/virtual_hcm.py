from utils.files import *
from common.constant import *
from nlu.question_type_classifier import QuestionTypeClassifier
from nlu.intent_classifier import IntentClassifier
from nlu.model.intent import *
from vncorenlp import VnCoreNLP


class VirtualHCMChatbot(object):
    def __init__(self):
        print("Stating bot, loading resources...")
        # Application config
        self.config = load_config(CONFIG_PATH)
        # VNCoreNLP tokenizer
        rdrsegmenter = VnCoreNLP(self.config[VNCORENLP])
        # Intent reconizer
        self.intent_reconizer = IntentClassifier(rdrsegmenter, self.config[BERT], self.config)
        self.intent2idx, self.idx2intent = self.intent_reconizer.load()
        # Question type reconizer
        self.question_type_reconizer = QuestionTypeClassifier(rdrsegmenter, self.config[BERT], self.config)
        self.type2idx, self.idx2type = self.question_type_reconizer.load()
        # For dialogue states tracking (list of tuples(intent,action))
        self.state_tracker = []
        # Load intent data
        self.intent_datas = load_from_data(self.config[INTENT_DATA_PATH])

    def decide_action(self, question):
        """Combines intent and question type recognition to decide bot action"""
        pass

    def chat(self, question):
        intent_name = self.intent_reconizer.predict(question)
        types = self.question_type_reconizer.predict(question)
        intent = self.intent_datas[intent_name]
        response = intent.base_response
        for t in types:
            type_id = self.type2idx[t]
            if type_id in intent.intent_types:
                response += (' ' + intent.corresponding_datas[type_id])
        return response
