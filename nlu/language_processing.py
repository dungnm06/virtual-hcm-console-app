from nlu.model.intent import Intent
from vncorenlp import VnCoreNLP
from utils.files import load_config
from common.constant import *
from itertools import product

# VNCoreNLP tokenizer
rdrsegmenter = VnCoreNLP(load_config(CONFIG_PATH)[VNCORENLP])


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
    return ' '.join([' '.join(sentence) for sentence in word_segmented_text])


def words_unsegmentation(sentence):
    if type(sentence) is str:
        return ' '.join([s.strip().replace('_', ' ') for s in sentence.split(' ')])
    elif type(sentence) is list:
        return [' '.join([s2.strip().replace('_', ' ') for s2 in s1.split(' ')]) for s1 in sentence]
    else:
        raise Exception("Invaild input type, only str or list of string")


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


def check_critical_parts(critical_parts, sentence):
    return False


def is_same_intent(intent, sentence):
    flag = True
    flag = check_critical_parts(intent.critical_datas, sentence)

    return flag
