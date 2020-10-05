import re
import numpy as np


def batch_word_segmentation(segmenter, texts):
    segmented_text = []
    if type(texts) is list:
        for text in texts:
            segmented_text.extend(word_segmentation(segmenter, text))
    elif type(texts) is str:
        segmented_text.extend(word_segmentation(segmenter, texts))
    else:
        raise Exception("Invaild input type, only str or list of string")
  
    return segmented_text


def word_segmentation(segmenter, text):
    word_segmented_text = segmenter.tokenize(text)
    return [' '.join(sentence) for sentence in word_segmented_text]


def text_prepare(text):
    """Performs tokenization and simple preprocessing."""
    replace_by_space_re = re.compile(r'[/(){}\[\]|@,;]')
    good_symbols_re = re.compile('[^0-9a-z #+_]')

    text = text.lower()
    text = replace_by_space_re.sub(' ', text)
    text = good_symbols_re.sub('', text)
    text = ' '.join([x for x in text.split() if x])

    return text.strip()


def load_embeddings(embeddings_path):
    """Loads pre-trained word embeddings from tsv file.
    Args:
      embeddings_path - path to the embeddings file.
    Returns:
      embeddings - dict mapping words to vectors;
      embeddings_dim - dimension of the vectors.
    """

    # Hint: you have already implemented a similar routine in the 3rd assignment.
    # Note that here you also need to know the dimension of the loaded embeddings.
    # When you load the embeddings, use numpy.float32 type as dtype

    embeddings = {}
    dimension, initialized = 0, False
    for line in open(embeddings_path):
        word, *ex = line.split('\t')
        embeddings[word] = np.array(ex, dtype="float32")
        if not initialized:
            initialized = True
            dimension = embeddings[word].shape[0]

    return embeddings, dimension


def question_to_vec(question, embeddings, dim):
    """ Transforms a string to an embedding by averaging word embeddings.
        question: a string
            embeddings: dict where the key is a word and a value is its' embedding
            dim: size of the representation

            result: vector representation for the question
    """

    list_words = question.split()
    question_vec = np.zeros(dim)
    no_count_words = 0
    for word in list_words:
        if word in embeddings:
            w_vec = embeddings[word]
            question_vec += w_vec
        else:
            no_count_words += 1
    if len(list_words) - no_count_words > 0:
        question_vec /= (len(list_words) - no_count_words)

    return question_vec

