import nltk
import re
import spacy
import string
from multi_rake import Rake


RE_PUNC = re.compile('[%s]' % re.escape(string.punctuation))


def main():
    text = ""
    lang = spacy.load('pt')
    input_file = open('1f86ecdf-4659-4a98-84bf-78d0ef9d3512.txt')
    for line in input_file:
        phrase = RE_PUNC.sub('', line)
        phrase = phrase.lower()
        doc = lang(line)
        for token in doc:
            pos = token.pos_
            if pos == 'VERB':
                phrase = phrase.replace(str(token), token.lemma_)
        text += line
        text += "-- " + phrase
    print(text)

    rake = Rake(
        min_chars=2,
        max_words=1,
        min_freq=1,
        language_code='pt',
        stopwords=nltk.corpus.stopwords.words('portuguese'),  # {'and', 'of'}
        lang_detect_threshold=50,
        max_words_unknown_lang=2,
        generated_stopwords_percentile=80,
        generated_stopwords_max_len=3,
        generated_stopwords_min_freq=2,
    )
    keywords = rake.apply(text)
    # output_text = ''
    for keyword in keywords:
        print(keyword)
        # output_text += keyword + '\n'
    # open('output.txt', 'w').write(output_text)


if __name__ == "__main__":
    main()