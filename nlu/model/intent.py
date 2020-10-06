import pandas as pd
from utils.strings import isInt

# Mapping
ID = 'ID'
NAME = 'Intent'
QUESTIONS = 'Questions'
RAW_DATA = 'Raw data'
BASE_RESPONSE = 'BaseResponse'
INTENT_TYPES = 'IntentType'
CORRESPONDING_DATAS = 'Corresponding Data'
CRITICAL_DATAS = 'CriticalData'
REFERENCE_DOC_ID = 'Reference Document ID'
REFERENCE_DOC_PAGE = 'Page'
SENTENCE_COMPONENTS = 'Components Of Questions'


class Intent:
    def __init__(self, intent_id=0, intent='', questions=None, raw_data='', base_response='',
                 intent_types=None, corresponding_datas=None, critical_datas=None,
                 reference_doc_id=0, reference_doc_page=0, sentence_components=None):
        # Default argument value is mutable
        # https://florimond.dev/blog/articles/2018/08/python-mutable-defaults-are-the-source-of-all-evil
        if intent_types is None:
            intent_types = []
        if sentence_components is None:
            sentence_components = []
        if critical_datas is None:
            critical_datas = []
        if corresponding_datas is None:
            corresponding_datas = {}
        if questions is None:
            questions = []
        # Assign attributes
        self.intent_id = intent_id
        self.intent = intent
        self.questions = questions
        self.raw_data = raw_data
        self.base_response = base_response
        self.intent_types = intent_types
        self.corresponding_datas = corresponding_datas
        self.critical_datas = critical_datas
        self.reference_doc_id = reference_doc_id
        self.reference_doc_page = reference_doc_page
        self.sentence_components = sentence_components


def load_from_data(datapath):
    intent_maps = {}
    intent_datas = pd.read_csv(datapath)
    for data in intent_datas:
        intent = Intent()
        # ID
        intent.intent_id = int(data[ID])
        # Intent name
        intent.intent = data[NAME]
        # Questions
        intent.questions = [q.strip() for q in data[QUESTIONS].split('#')]
        # Raw data
        intent.raw_data = data[RAW_DATA]
        # Intent types
        intent.intent_types = [int(t) for t in data[INTENT_TYPES].split(',')]
        # Corresponding datas
        cd = data[CORRESPONDING_DATAS].split('#')
        intent.corresponding_datas = {i: v for i, v in zip(intent.intent_types, cd)}
        # Critical datas
        intent.critical_datas = [tuple(i.split(',')) for i in data[CRITICAL_DATAS].split('#')]
        # Reference document id
        intent.reference_doc_id = data[REFERENCE_DOC_ID]
        # Reference document page
        intent.reference_doc_page = int(data[REFERENCE_DOC_PAGE]) if isInt(data[REFERENCE_DOC_PAGE]) else data[
            REFERENCE_DOC_PAGE]
        # Sentence components
        sentence_components = data[SENTENCE_COMPONENTS].split('#')
        type_value_pairs = {}
        for component in sentence_components:
            smaller_parts = component.split(',')
            type_value_pairs = {p.split(':')[0]: p.split(':')[1] for p in smaller_parts}
            for word_type in type_value_pairs:
                if word_type.lower() == 'ns':
                    noun_phrases = [
                        type_value_pairs[word_type][1:(len(type_value_pairs[word_type]) - 1)].split('+')]
                    noun_phrases = [{part.strip(':')[0]: part.strip(':')[1] for part in noun_phrases}]
                    type_value_pairs[word_type] = noun_phrases
        intent.sentence_components = type_value_pairs
        intent_maps[intent.intent] = intent
    return intent_maps
