from multi_rake import Rake
from gensim.models import KeyedVectors
import nltk
import re
import spacy
import string

RE_PUNC = re.compile('[%s]' % re.escape(string.punctuation))


class KeywordsGenerator:
    def __init__(self, lang, model, input_file):
        """
        lang(string): language of the input_file
        model(string): path of the word embedding model
        input_file(string): input text to extract keywords from
        aug_supported_lang's: languages tha are supported to apply word augmenter
        """
        self.lang = lang
        try:
            self.nlp = spacy.load(lang)
        except OSError:
            print(lang, 'model not installed. Run `python -m spacy download pt_core_news_sm` in the command line to'
                  'install, then `python -m spacy link pt_core_news_sm pt --force` to correctly link.')
        self.model = KeyedVectors.load_word2vec_format("word2vec.vec", binary=False)
        self.input_file = open(input_file)
        self.aug_supported_langs = ['pt', 'en']

    def augment_keywords(self, keywords):
        new_keywords = []
        for keyword in keywords:
            new_keywords.append(keyword)
            try:
                keyword_similar_list = self.model.most_similar(keyword, topn=5)
            except KeyError:
                keyword_similar_list = []
            for keyword_similar in keyword_similar_list:
                if keyword_similar[1] > 0.6:
                    new_keywords.append(keyword_similar[0])
    
        return new_keywords

    def get_keywords(self, is_augmented=True):
        text = ""
        for line in self.input_file:
            phrase = RE_PUNC.sub('', line)
            phrase = phrase.lower()
            doc = self.nlp(line)
            for token in doc:
                pos = token.pos_
                if pos == 'VERB':
                    phrase = phrase.replace(str(token), token.lemma_)
            text += line
            text += "-- " + phrase

        stopwords = nltk.corpus.stopwords.words('portuguese')

        rake = Rake(
            min_chars=2,
            max_words=1,
            min_freq=1,
            language_code='pt',
            stopwords=stopwords,  # {'and', 'of'}
            lang_detect_threshold=50,
            max_words_unknown_lang=2,
            generated_stopwords_percentile=80,
            generated_stopwords_max_len=3,
            generated_stopwords_min_freq=2,
        )
        keywords_tuple = rake.apply(text)
        keywords = [keyword[0] for keyword in keywords_tuple]
        if is_augmented and self.lang in self.aug_supported_langs:
            keywords = self.augment_keywords(keywords)
        return keywords


if __name__ == "__main__":
    generator = KeywordsGenerator('pt', 'word2vec.vec', 'data/792b3931-3da3-4f1c-94d0-db0a4028f4e4.txt')
    generator.get_keywords()
