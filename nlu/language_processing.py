from nlu.model.intent import Intent
from vncorenlp import VnCoreNLP
from utils.files import load_config
from common.constant import *
from itertools import product

# Variables for language understanding tasks
config = load_config(CONFIG_PATH)
rdrsegmenter = VnCoreNLP(config[VNCORENLP])
ner_types = config[NAMED_ENTITY_TYPES]
critical_data_ng_patterns = config[CRITICAL_DATA_NG_PATTERNS]
exclude_pos_tag = config[EXCLUDE_POS_TAG]
exclude_words = config[EXCLUDE_WORDS]


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


def pos_tagging(sentence):
    # Can only handle simple sentence for now
    return rdrsegmenter.pos_tag(sentence)[0]


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


def generate_similary_sentences(sentence_synonym_dict_pair, word_segemented=False):
    org_sentence, synonym_dicts = sentence_synonym_dict_pair
    return_val = []
    # synonym_dicts: eg: 1: 'sinh', 2: 'TenBac' (each dict is instance of SynonymSet obj)
    # ['Bác', 'sinh', 'năm', '1890']
    words_segmented_sentences = []
    if not word_segemented:
        words_segmented_sentences = word_segmentation_no_join(org_sentence)
    else:
        words_segmented_sentences = [org_sentence]
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
    return [synonym_dicts[sid] for sid in synonym_dicts if word in synonym_dicts[sid].words]


def analyze_critical_parts(intent, sentence):
    # Word POS tagging
    pos_tag = rdrsegmenter.pos_tag(sentence)
    # Obtain named entity in the sentence
    ner = named_entity_reconize(sentence)
    # Data for the process
    intent_critical_datas = intent.critical_datas
    tokenized_sentence = word_segmentation(sentence)
    tokenized_sentence_list = tokenized_sentence.split()
    normalized_sentence_list = [w[0] for w in pos_tag if w[0] not in exclude_words and w[1] not in exclude_pos_tag]
    normalized_sentence = ' '.join(normalized_sentence_list)

    # Nothing to analyze
    if len(ner) == 0 and len(intent_critical_datas) == 0:
        return True

    # Sentence structure
    sentence_struct = [p[1] for p in pos_tag]
    # Compare named entities in sentence with entities in intent
    check_flag = True
    for typ in ner_types:
        if not check_flag:
            break
        entities_in_sentence = [(e[1], tokenized_sentence_list.index(e[1]), tokenized_sentence_list.index(e[1])) for e
                                in ner if e[0] == typ]
        if len(entities_in_sentence) > 0:
            print(entities_in_sentence)
        else:
            print('No', typ, 'entity in sentence')
        entities_in_intent = []
        for c in intent_critical_datas:
            for c1 in c:
                if c1[0] == typ:
                    entities_in_intent.append(c)
                    break
        if len(entities_in_intent) > 0:
            print(entities_in_intent)
        else:
            print('No', typ, 'entity in intent')

        # Check by extracted NER by VNCoreNLP
        eit_1 = entities_in_intent[:]
        eis_1 = entities_in_sentence[:]
        for eit in eit_1:
            # get components structure
            struct = [c[0] for c in eit]
            # get main component index
            main_pos = struct.index(typ)
            # synonyms of main component
            synonyms = get_synonym_dicts(eit[main_pos][1], intent.synonym_sets)
            # get corresponse part in the sentence
            # (word, entity start pos, entity end pos)
            corresponse_part = None
            for e in eis_1:
                for syn in synonyms:
                    if e[0] in syn.words:
                        corresponse_part = e
                        break
            # In case VNCoreNLP pos-tagging not working right, compare string instead
            if not corresponse_part:
                tokenized_sentence_list_lower = [w.lower() for w in tokenized_sentence_list]
                # sentence_list_lower = [w.lower() for w in sentence_list]
                if typ == 'MISC':
                    content = [part.split(':')[1] for part in eit[main_pos][1].split('+') if
                               part.split(':')[0] not in exclude_pos_tag]
                    content = [c for c in content if c not in exclude_words]
                    print(content)
                    similaries = generate_similary_sentences((content, intent.synonym_sets), word_segemented=True)
                    print(similaries)
                    for sim in similaries:
                        sim = sim.lower()
                        if sim in normalized_sentence.lower():
                            print('Found MISC')
                            sim_list = sim.split()
                            start_idx = tokenized_sentence_list_lower.index(sim_list[0])
                            end_idx = tokenized_sentence_list_lower.index(sim_list[len(sim_list) - 1])
                            corresponse_part = (sim, start_idx, end_idx)
                            break
                else:
                    for syn in synonyms:
                        for word in syn.words:
                            if word.lower() in sentence.lower():
                                corresponse_part = (
                                    word,
                                    tokenized_sentence_list_lower.index(word.lower()),
                                    tokenized_sentence_list_lower.index(word.lower())
                                )
            # Critical part still not found -> not same intent
            if not corresponse_part:
                check_flag = False
                break
            # Critical data existing, but need to check grammar structure too
            for pattern in critical_data_ng_patterns:
                pattern_struct = pattern[0].split(MINUS)
                start_pos = corresponse_part[1]
                end_pos = corresponse_part[2]
                rel_pos_main = pattern_struct.index('main')
                rel_pos_end = len(pattern_struct) - 1
                tmp = start_pos - rel_pos_main
                left_pos = tmp if tmp >= 0 else 0
                tmp = end_pos + (rel_pos_end - rel_pos_main)
                right_pos = tmp if tmp <= (len(tokenized_sentence_list) - 1) else (len(tokenized_sentence_list) - 1)
                # walk to left
                struct_ok = True
                for i in range(start_pos - left_pos):
                    # If Preposition so skip
                    if pattern_struct[rel_pos_main - (i + 1)] == 'E' and sentence_struct[start_pos - (i + 1)] == 'E':
                        continue
                    # Out of sentence, grammar error
                    elif sentence_struct[rel_pos_main - i] == 'E' and (start_pos - (i + 1)) < 0:
                        check_flag = False
                        break
                    # NG pattern matching, change flag to False but keep walking to the end
                    elif ((pattern_struct[rel_pos_main - (i + 1)] != 'any' and pattern_struct[rel_pos_main - (i + 1)] ==
                           sentence_struct[start_pos - (i + 1)])
                          or (pattern_struct[rel_pos_main - (i + 1)] == 'any')):
                        struct_ok = False
                    # NG pattern not match, sentence is ok
                    elif pattern_struct[rel_pos_main - (i + 1)] != 'any' and pattern_struct[rel_pos_main - (i + 1)] != \
                            sentence_struct[start_pos - (i + 1)]:
                        struct_ok = True
                        break
                if not struct_ok:
                    check_flag = struct_ok
                    break
                # walk to right
                for i in range(right_pos - end_pos):
                    # If Preposition so skip
                    if pattern_struct[rel_pos_main + (i + 1)] == 'E' and sentence_struct[end_pos + (i + 1)] == 'E':
                        continue
                    # Out of sentence, grammar error
                    elif sentence_struct[start_pos + i] == 'E' and (end_pos + (i + i)) > (len(tokenized_sentence_list) - 1):
                        check_flag = False
                        break
                    # NG pattern matching, change flag to False but keep walking to the end
                    elif ((pattern_struct[rel_pos_main + (i + 1)] != 'any' and pattern_struct[rel_pos_main + (i + 1)] ==
                           sentence_struct[end_pos + (i + 1)])
                          or (pattern_struct[rel_pos_main + (i + 1)] == 'any')):
                        struct_ok = False
                    # NG pattern not match, sentence is ok
                    elif pattern_struct[rel_pos_main + (i + 1)] != 'any' and pattern_struct[rel_pos_main + (i + 1)] != \
                            sentence_struct[end_pos + (i + 1)]:
                        struct_ok = True
                        break
                if not struct_ok:
                    check_flag = struct_ok
                    break

    return check_flag


def is_same_intent(intent, sentence):
    flag = True
    flag = analyze_critical_parts(intent, sentence)

    return flag
