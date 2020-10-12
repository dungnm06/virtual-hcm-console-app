from utils.files import *
from nlu.question_type_classifier import QuestionTypeClassifier
from nlu.intent_classifier import IntentClassifier
from nlu.model.intent import *
from nlu.language_processing import is_same_intent
from .action_type import *


class VirtualHCMChatbot(object):
    def __init__(self):
        print("Stating bot, loading resources...")
        # Application config
        self.config = load_config(CONFIG_PATH)
        # Intent reconizer
        self.intent_reconizer = IntentClassifier(self.config[BERT], self.config)
        self.intent2idx, self.idx2intent = self.intent_reconizer.load()
        # Question type reconizer
        self.question_type_reconizer = QuestionTypeClassifier(self.config[BERT], self.config)
        # For dialogue states tracking (list of tuples(intent,types,action))
        self.state_tracker = []
        # Intent, types map
        self.type2idx, self.idx2type = self.question_type_reconizer.load()
        # Answer generator
        self.answer_generator = AnswerGenerator(self)
        # Load intent data
        self.intent_datas = load_from_data(self.config)

    def __regis_history(self, intent, types, action):
        self.state_tracker.append((intent, types, action))

    def get_last_state(self):
        return self.state_tracker[len(self.state_tracker)-1]

    def __decide_action(self, chat_input, intent):
        """Combines intent and question type recognition to decide bot action"""
        last_state = self.get_last_state()
        if intent.intent_id == last_state[0].intent_id and last_state[2] == AWAIT_CONFIRMATION:
            if chat_input.lower() == 'đúng':
                return CONFIRMATION_OK
            if chat_input.lower() == 'sai':
                return CONFIRMATION_NG
        if is_same_intent(intent, chat_input):
            return ANSWER
        else:
            return AWAIT_CONFIRMATION

    def chat(self, chat_input):
        intent_name = self.intent_reconizer.predict(chat_input)
        types = self.question_type_reconizer.predict(chat_input)
        intent = self.intent_datas[intent_name]
        action = self.__decide_action(intent, types)
        self.__regis_history(intent, types, action)
        return self.answer_generator.get_response(intent, types, action)


class AnswerGenerator:
    def __init__(self, chatbot):
        self.chatbot = chatbot

    def get_response(self, intent, types, action):
        if action == ANSWER:
            return self.answer(intent, types)
        elif action == AWAIT_CONFIRMATION:
            return self.confirmation(intent)
        elif action == CONFIRMATION_OK:
            last_state = self.chatbot.get_last_state()
            return self.answer(last_state[0], last_state[1])
        elif action == CONFIRMATION_NG:
            return self.confirmation_ng()

    @staticmethod
    def confirmation(intent):
        return 'Có phải bạn đang nói tới: ' + intent.name + ' ? (đúng, sai)'

    @staticmethod
    def confirmation_ng():
        return 'Hiện tại chức năng báo cáo chưa hoàn thiện, mời bạn hỏi lại câu mới!'

    def answer(self, intent, types):
        response = intent.base_response
        for t in types:
            type_id = self.chatbot.type2idx[t]
            if type_id in intent.intent_types:
                response += (' ' + intent.corresponding_datas[type_id])
        return response
