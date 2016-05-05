import nltk as nl
import pandas as pd

LEXICON_NRC = pd.read_csv("data/NRC-emotion-lexicon-wordlevel-alphabetized-v0.92.csv",
                          names=["word", "emotion", "score"])
LEXICON_AFINN = pd.read_csv("data/AFINN-111.csv", names=["word", "valence"])

COLLABEL = ["anger", "anticipation", "disgust", "fear", "joy", "negative", "positive",
            "sadness", "surprise", "trust"]




def count_lines(path_to_file):
    """
    Count the number of lines of a text file
    The result of opening a file is an iterator, which can be converted to a sequence, which has a length
    :type path_to_file: str
    """
    with open(path_to_file) as f:
        return len(list(f))


def get_nrc_values(text):
    """
    Get the nrc values from a string
    :rtype: dict
    :param text: either a string or a collection of words
    :return: nrc_value: score of the emotions and sentiments for the text
    """
    nrc_value = {key: 0 for key in COLLABEL}
    try:
        tokens = nl.word_tokenize(text)
        for token in tokens:
            if LEXICON_NRC["word"].isin([token]).any():
                # TODO : find something else for this test
                for emotion in nrc_value.keys():
                    nrc_value[emotion] += int(
                        LEXICON_NRC[(LEXICON_NRC.word == token) & (LEXICON_NRC.emotion == emotion)]["score"])
    except TypeError:
        for word in text:
            if LEXICON_NRC["word"].isin([word.content.encode('UTF-8')]).any():
                # TODO : find something else for this test
                for emotion in nrc_value.keys():
                    nrc_value[emotion] += int(
                        LEXICON_NRC[
                            (LEXICON_NRC.word == word.content.encode('UTF-8')) & (LEXICON_NRC.emotion == emotion)][
                            "score"])
                    # print "match"

    return nrc_value


def get_valence_values(string):
    """
    Get the valence values from a string
    :param string:
    :return: score
    """
    valence_value = 0
    tokens = nl.word_tokenize(string)
    for token in tokens:
        if LEXICON_AFINN["word"].isin([token]).any():
            # TODO : find something else for this test
            valence_value += int(LEXICON_AFINN[LEXICON_AFINN.word == token]["valence"])
            # print "match"
    return valence_value
