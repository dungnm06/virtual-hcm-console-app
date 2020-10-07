from nlu.model.intent import Intent
from vncorenlp import VnCoreNLP
from utils.files import load_config
from common.constant import *
from itertools import product

# Variables for language understanding tasks
config = load_config(CONFIG_PATH)
rdrsegmenter = VnCoreNLP(config[VNCORENLP])
ner_types = config[NAMED_ENTITY_TYPES]
critical_data_check_patterns = config[CRITICAL_DATA_PATTERNS]


def word_segmentation_no_join(text):
    return rdrsegmenter.tokenize(text)


def batch_word_segmentation(texts):
    segmented_text = []
    if type(texts) is list:
        for text in texts:
            segmented_text.extend(word_segmentation(text))
    elif type(texts) is str:
        segmented_text.extend(word_segmentation(texts))
    else:
        raise Exception("Invaild input type, only str or list of string")

    return segmented_text


def word_segmentation(text):
    word_segmented_text = rdrsegmenter.tokenize(text)
    return SPACE.join([SPACE.join(sentence) for sentence in word_segmented_text])


def words_unsegmentation(sentence):
    if type(sentence) is str:
        return SPACE.join([s.strip().replace(UNDERSCORE, SPACE) for s in sentence.split(SPACE)])
    elif type(sentence) is list:
        return [SPACE.join([s2.strip().replace(UNDERSCORE, SPACE) for s2 in s1.split(SPACE)]) for s1 in sentence]
    else:
        raise Exception("Invaild input type, only str or list of string")


def named_entity_reconize(sentence):
    # Can only handle simple sentence for now
    ner = rdrsegmenter.ner(sentence)[0]
    # Collect named entity words to list
    # Using BIO rule: B: begin, I: inside, O: outside
    named_entities_list = []
    tmp_word = ''
    current_type = ''
    for o in ner:
        word, tag = o
        if tag != 'O':
            pos, typ = tag.split('-')
            # Not supporting type
            if typ not in ner_types:
                continue
            # Begin of entity
            if pos == 'B' and not current_type:
                tmp_word += word + SPACE
                current_type = typ
            # Inside of entity
            elif pos == 'I' and current_type == typ:
                tmp_word += word + SPACE
            # Start new entity
            elif pos == 'B' and current_type:
                named_entities_list.append((current_type, tmp_word.strip()))
                tmp_word = word + SPACE
                current_type = typ
        # Outside of entity
        elif tag == 'O' and current_type:
            named_entities_list.append((current_type, tmp_word.strip()))
            tmp_word = ''
            current_type = ''
    return named_entities_list


def generate_similary_sentences(sentence_synonym_dict_pair):
    org_sentence, synonym_dicts = sentence_synonym_dict_pair
    return_val = []
    # synonym_dicts: eg: 1: 'sinh', 2: 'TenBac' (each dict is instance of SynonymSet obj)
    # ['Bác', 'sinh', 'năm', '1890']
    words_segmented_sentences = word_segmentation_no_join(org_sentence)
    for words_segmented_sentence in words_segmented_sentences:
        # synonym_replaceable_pos: [(0,2), (1,1)]
        synonym_replaceable_pos = get_synonym_replaceable_pos(words_segmented_sentence, synonym_dicts)
        # [2, 1]
        using_dicts = [srp[1] for i, srp in enumerate(synonym_replaceable_pos)]
        # Generate all posible combinations
        # eg: [('Bác', 'sinh'), ('Bác', 'ra đời'), ('Hồ_Chí_Minh', 'sinh'), ('Hồ_Chí_Minh', 'ra đời')]
        combinations = list(product(*(synonym_dicts[i].words for i in using_dicts)))
        # Create similary sentences
        for c in combinations:
            sentence = words_segmented_sentence[:]
            for i, srp in enumerate(synonym_replaceable_pos):
                sentence[srp[0]] = c[i]
            return_val.append(sentence)

    return return_val


def get_synonym_replaceable_pos(org_sentence, synonym_dicts):
    # synonym_dicts: eg: 1: 'sinh', 2: 'TenBac' (each dict is instance of SynonymSet obj)
    # ['Bác', 'sinh', 'năm', '1890']
    # return [(0,2), (1,1)] - tuple of (word_pos_in_sentence, synonym_id)
    synonyms_replaceable_pos = []
    for i, word in enumerate(org_sentence):
        for dictionary_id in synonym_dicts:
            if word in synonym_dicts[dictionary_id].words:
                synonyms_replaceable_pos.append((i, dictionary_id))
    return synonyms_replaceable_pos


def batch_generate_similary_sentences(sentence_synonym_dict_pairs):
    return [generate_similary_sentences(pair) for pair in sentence_synonym_dict_pairs]


def get_synonym_dicts(word, synonym_dicts):
    # TODO:
    #  Use word embedding for sentiment analyze for more accurate in getting right synonym set
    #  in case of multiple meaning word may belong to multiple synonym set
    #  Currently get all synonym sets thats have the word
    return [sd for sd in synonym_dicts if word in sd.words]


def analyze_critical_parts(intent, sentence):
    intent_critical_datas = intent.critical_datas
    # Obtain named entity in the sentence
    ner = named_entity_reconize(sentence)
    if len(ner) == 0 and len(intent_critical_datas) == 0:
        return True
    # Word POS tagging
    pos_tag = rdrsegmenter.pos_tag(ner)

    # Map for compare named entities in sentence with entities in intent
    ner_compare_map = {}
    for typ in ner_types:
        entities_in_sentence = [e[1] for e in ner if e[0] == typ]
        entities_in_intent = [e[1] for e in intent_critical_datas if e[0] == typ]
        # Number of entity not matching return not match
        if len(entities_in_sentence) != len(entities_in_intent):
            return False
        # Compare entities in sentence and intent
        for ie in entities_in_intent:
            # TODO
            pass
        ner_compare_map[typ] = (entities_in_sentence, entities_in_intent)

    return True


def is_same_intent(intent, sentence):
    flag = True
    flag = analyze_critical_parts(intent, sentence)

    return flag
