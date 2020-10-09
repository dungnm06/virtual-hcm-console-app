import pandas as pd
import json
from .synonym import *
from utils.strings import isInt
from common.constant import *

# Mapping
INTENT_ID = 'ID'
INTENT_NAME = 'Intent'
INTENT_QUESTIONS = 'Questions'
INTENT_RAW_DATA = 'Raw data'
INTENT_BASE_RESPONSE = 'BaseResponse'
INTENT_INTENT_TYPES = 'IntentType'
INTENT_CORRESPONDING_DATAS = 'Corresponding Data'
INTENT_CRITICAL_DATAS = 'CriticalData'
INTENT_REFERENCE_DOC_ID = 'Reference Document ID'
INTENT_REFERENCE_DOC_PAGE = 'Page'
INTENT_SENTENCE_COMPONENTS = 'Components Of Questions'
INTENT_SYNONYM_IDS = 'Synonyms ID'


class Intent:
    def __init__(self, intent_id=0, intent='', questions=None, raw_data='', base_response='',
                 intent_types=None, corresponding_datas=None, critical_datas=None,
                 reference_doc_id=0, reference_doc_page=0, sentence_components=None, synonym_sets=None):
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
        if synonym_sets is None:
            synonym_sets = {}
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
        self.synonym_sets = synonym_sets


def load_from_data(datapath):
    intent_maps = {}
    intent_datas = pd.read_csv(datapath[INTENT_MAP_PATH])
    f = open(datapath[INTENT_MAP_PATH], encoding=UTF8)
    synonym_dictionary = json.load(f)
    for idx, data in intent_datas.iterrows():
        intent = Intent()
        # ID
        intent.intent_id = int(data[INTENT_ID])
        # Intent name
        intent.intent = data[INTENT_NAME]
        # Questions
        intent.questions = [q.strip() for q in data[INTENT_QUESTIONS].split(HASH)]
        # Raw data
        intent.raw_data = data[INTENT_RAW_DATA]
        # Intent types
        intent.intent_types = [int(t) for t in data[INTENT_INTENT_TYPES].split(COMMA)]
        # Corresponding datas
        cd = data[INTENT_CORRESPONDING_DATAS].split(HASH)
        intent.corresponding_datas = {i: v for i, v in zip(intent.intent_types, cd)}
        # Critical datas
        cd = data[INTENT_CRITICAL_DATAS]
        if not pd.isnull(cd):
            for i in cd.split(HASH):
                if i:
                    group_data = []
                    for i1 in i.split(COMMA):
                        split_idx = i1.find(COLON)
                        if i1.startswith('MISC'):
                            group_data.append(('MISC', i1[(split_idx + 1):]))
                        else:
                            group_data.append((i1[:split_idx], i1[(split_idx + 1):]))
                    intent.critical_datas.append(group_data)
        # Reference document id
        rdi = data[INTENT_REFERENCE_DOC_ID]
        if not pd.isnull(rdi):
            intent.reference_doc_id = rdi
        # Reference document page
        rdp = data[INTENT_REFERENCE_DOC_PAGE]
        if not pd.isnull(rdp):
            intent.reference_doc_page = int() if isInt(rdp) else rdp
        # Sentence components
        sc = data[INTENT_SENTENCE_COMPONENTS]
        if not pd.isnull(sc):
            sentence_components = data[INTENT_SENTENCE_COMPONENTS].split(HASH)
            type_value_pairs = {}
            for component in sentence_components:
                # print(component)
                smaller_parts = component.split(COMMA)
                # print(smaller_parts)
                for p in smaller_parts:
                    split_idx = p.find(COLON)
                    type_value_pairs = {p[:split_idx]: p[(split_idx + 1):]}
                    # print(type_value_pairs)
                    for word_type in type_value_pairs:
                        # Noun phrases handling
                        if word_type.lower() == 'ns':
                            noun_phrases = type_value_pairs[word_type][1:(len(type_value_pairs[word_type]) - 1)].split(
                                PLUS)
                            # print(noun_phrases)
                            tmp_dict = {}
                            for part in noun_phrases:
                                spart = part.split(COLON)
                                tmp_dict[spart[0]] = spart[1]
                            # print(tmp_dict)
                            type_value_pairs[word_type] = tmp_dict
            intent.sentence_components = type_value_pairs
        # Synonym words dictionary
        synonym_ids = data[INTENT_SYNONYM_IDS]
        if not pd.isnull(synonym_ids):
            synonym_ids = synonym_ids.split(COMMA)
            for s in synonym_ids:
                synonym_set = SynonymSet()
                synonym_set.id = int(s)
                synonym_set.meaning = synonym_dictionary[s][SYNONYM_MEANING]
                synonym_set.words = synonym_dictionary[s][SYNONYM_WORDS]
                intent.synonym_sets[s] = synonym_set
        # Push to intents map
        intent_maps[intent.intent] = intent
    # Close file reading
    f.close()

    return intent_maps
