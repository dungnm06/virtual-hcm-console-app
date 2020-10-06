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
    for idx, data in intent_datas.iterrows():
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
        cd = data[CRITICAL_DATAS]
        if not pd.isnull(cd):
            intent.critical_datas = [tuple(i.split(',')) for i in cd.split('#')]
        # Reference document id
        rdi = data[REFERENCE_DOC_ID]
        if not pd.isnull(rdi):
            intent.reference_doc_id = rdi
        # Reference document page
        rdp = data[REFERENCE_DOC_PAGE]
        if not pd.isnull(rdp):
            intent.reference_doc_page = int() if isInt(rdp) else rdp
        # Sentence components
        sc = data[SENTENCE_COMPONENTS]
        if not pd.isnull(sc):
            sentence_components = data[SENTENCE_COMPONENTS].split('#')
            type_value_pairs = {}
            for component in sentence_components:
                # print(component)
                smaller_parts = component.split(',')
                # print(smaller_parts)
                for p in smaller_parts:
                    split_idx = p.find(':')
                    type_value_pairs = {p[:split_idx]: p[(split_idx + 1):]}
                    # print(type_value_pairs)
                    for word_type in type_value_pairs:
                        if word_type.lower() == 'ns':
                            noun_phrases = type_value_pairs[word_type][1:(len(type_value_pairs[word_type]) - 1)].split(
                                '+')
                            # print(noun_phrases)
                            tmp_dict = {}
                            for part in noun_phrases:
                                spart = part.split(':')
                                tmp_dict[spart[0]] = spart[1]
                            # print(tmp_dict)
                            type_value_pairs[word_type] = tmp_dict
            intent.sentence_components = type_value_pairs

        intent_maps[intent.intent] = intent

    return intent_maps
