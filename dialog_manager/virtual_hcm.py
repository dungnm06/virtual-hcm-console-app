from utils.files import *
from nlu.question_type_classifier import QuestionTypeClassifier
from nlu.intent_classifier import IntentClassifier
from nlu.model.intent import *
from nlu.language_processing import analyze_sentence_components
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
        self.state_tracker.append((Intent(), None, INITIAL))
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

    @staticmethod
    def __decide_action(chat_input, intent, types, last_state):
        """Combines intent and question type recognition to decide bot action"""
        # print(last_state)
        if intent.intent_id == last_state[0].intent_id and last_state[2] == AWAIT_CONFIRMATION:
            if chat_input.lower() == 'đúng':
                return CONFIRMATION_OK
            else:
                return CONFIRMATION_NG
        else:
            if analyze_sentence_components(intent, chat_input):
                return ANSWER
            else:
                return AWAIT_CONFIRMATION

    def chat(self, chat_input):
        last_state = self.get_last_state()
        if last_state[2] != AWAIT_CONFIRMATION:
            intent_name = self.intent_reconizer.predict(chat_input)
            types = self.question_type_reconizer.predict(chat_input)
            intent = self.intent_datas[intent_name]
        else:
            intent = last_state[0]
            types = last_state[1]
        action = self.__decide_action(chat_input, intent, types, last_state)
        # print(action)
        self.__regis_history(intent, types, action)
        return self.answer_generator.get_response(intent, types, action, last_state)


class AnswerGenerator:
    def __init__(self, chatbot):
        self.chatbot = chatbot
        self.id2type = self.chatbot.config[QUESTION_TYPE_MAP_PREDEFINE]
        self.type2id = {v: k for k, v in self.id2type.items()}

    def get_response(self, intent, types, action, last_state):
        if action == ANSWER:
            return self.answer(intent, types)
        elif action == AWAIT_CONFIRMATION:
            return self.confirmation(intent)
        elif action == CONFIRMATION_OK:
            return self.answer(last_state[0], last_state[1])
        elif action == CONFIRMATION_NG:
            return self.confirmation_ng()

    @staticmethod
    def confirmation(intent):
        return 'Có phải bạn đang hỏi về: ' + intent.name + '? (đúng, sai)'

    @staticmethod
    def confirmation_ng():
        return 'Hiện tại chức năng báo cáo chưa hoàn thiện, mời bạn hỏi lại câu mới!'

    def answer(self, intent, types):
        response = intent.base_response
        # Get type data exists in intent
        existing_types = [int(self.type2id[t]) for t in types if int(self.type2id[t]) in intent.intent_types]
        # If any of user asking data types not exist in intent so just print all intent data
        if not existing_types:
            existing_types = intent.intent_types
        for t in existing_types:
            response += (' ' + intent.corresponding_datas[t])
        return response
