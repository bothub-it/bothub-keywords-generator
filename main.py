from multi_rake import Rake
import stopwordsiso as stopwords
import re
import string

RE_PUNC = re.compile('[%s]' % re.escape(string.punctuation))


class KeywordsGenerator:
    def __init__(self, lang, input_file):
        """
        lang(string): language of the input_file
        model(string): path of the word embedding model
        input_file(string): input text to extract keywords from
        aug_supported_lang's: languages tha are supported to apply word augmenter
        """
        self.lang = lang
        self.aug_supported_langs = ['pt', 'en']
        if not stopwords.has_lang(lang) and lang not in self.aug_supported_langs:
            print('language not supported')
            return
        self.stopwords = stopwords.stopwords([lang, 'en'])
        self.input_file = open(input_file)

    @staticmethod
    def augment_keywords(keywords):
        new_keywords = []
        sentence_suggestion = SentenceSuggestion()
        for keyword in keywords:
            new_keywords.append(keyword)
            try:
                keyword_similar_list = sentence_suggestion.most_similar(keyword, topn=6)
            except KeyError:
                keyword_similar_list = []
            for keyword_similar in keyword_similar_list:
                if keyword_similar[1] > 0.6:
                    new_keywords.append(keyword_similar[0])

        return new_keywords

    def get_keywords(self, is_augmented=True):
        text = ''
        for line in self.input_file:
            phrase = RE_PUNC.sub('', line)
            phrase = phrase.lower()
            text += phrase

        text = str.join(' ', text.splitlines())

        rake = Rake(
            min_chars=2,
            max_words=1,
            min_freq=1,
            language_code='pt',
            stopwords=self.stopwords,  # {'and', 'of'}
            lang_detect_threshold=50,
            max_words_unknown_lang=2,
            generated_stopwords_percentile=80,
            generated_stopwords_max_len=3,
            generated_stopwords_min_freq=2,
        )

        if not rake:
            text_tokens = text.split(' ')
            tokens = [word for word in text_tokens]
            filtered_words = [word for word in tokens if word not in self.stopwords]
            keywords = filtered_words
        else:
            keywords_tuple = rake.apply(text)
            keywords = [keyword[0] for keyword in keywords_tuple]

        if is_augmented and self.lang in self.aug_supported_langs:
            keywords = self.augment_keywords(keywords)
        return keywords


if __name__ == "__main__":
    generator = KeywordsGenerator('pt', 'data/792b3931-3da3-4f1c-94d0-db0a4028f4e4.txt')
    print(generator.get_keywords(False))
