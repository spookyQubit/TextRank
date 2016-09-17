__author__ = 'shantanu'

import re
import numpy as np
import argparse
from nltk.tokenize.punkt import PunktSentenceTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


text = "Automatic summarization is the process of reducing a text document with a computer program in " \
    "order to create a summary that retains the most important points of the original document. " \
    "Technologies that can make a coherent summary take into account variables such as length, " \
    "writing style and syntax. Automatic data summarization is part of machine learning and data mining. " \
    "The main idea of summarization is to find a representative subset of the data, which contains the " \
    "information of the entire set. Summarization technologies are used in a large number of sectors in " \
    "industry today. An example of the use of summarization technology is search engines such as Google. " \
    "Other examples include document summarization, image collection summarization and video summarization. " \
    "Document summarization, tries to automatically create a representative summary or abstract of the entire document, " \
    "by finding the most informative sentences. Similarly, in image summarization the system finds the most " \
    "representative and important (or salient) images. Similarly, in consumer videos one would want to " \
    "remove the boring or repetitive scenes, and extract out a much shorter and concise version of the video. " \
    "This is also important, say for surveillance videos, where one might want to extract only important " \
    "events in the recorded video, since most part of the video may be uninteresting with nothing going on. " \
    "As the problem of information overload grows, and as the amount of data increases, the interest in automatic " \
    "summarization is also increasing."

doc = "We introduce contextual values as a generalization of the eigenvalues of an observable that takes " \
    "into account both the system observable and a general measurement procedure. This technique leads " \
    "to a natural definition of a general conditioned average that converges uniquely to the quantum weak " \
    "value in the minimal disturbance limit. As such, we address the controversy in the literature " \
    "regarding the theoretical consistency of the quantum weak value by providing a more general theoretical " \
    "framework and giving several examples of how that framework relates to existing experimental and " \
    "theoretical results."


doc = "Accumulation of intracellular double-stranded RNA (dsRNA) usually marks viral " \
    "infections or de-repression of endogenous retroviruses and repeat elements. The innate " \
    "immune system, the first line of defense in mammals, is therefore equipped to sense " \
    "dsRNA and mount a protective response. The largest family of dsRNA sensors are " \
    "oligoadenylate synthetases (OAS) which produce a second messenger, 2-5A, in " \
    "response to dsRNA. This 2-5A activates an endoribonuclease, RNase L, which cleaves " \
    "single-stranded cellular and viral RNAs. OAS/RNase L is not only essential for coping " \
    "with bacterial and viral infections but also a major regulator of cell cycle progression, " \
    "differentiation, and apoptosis, processes often misregulated in cancers. We seek to " \
    "understand the dynamics and molecular basis of signaling in the OAS/RNase L " \
    "pathway. To this end we have developed a three-pronged approach to: a) identify " \
    "dsRNAs that accumulate b) monitor 2-5A levels real-time in live cells and c) map direct " \
    "RNA cleavages by RNase L. These approaches collectively provide a complete " \
    "molecular framework to examine dsRNA signaling in various infections and disease " \
    "states."

#doc = "I am going to school. I like to work. like to play. Hello you there."


class PorterTokenizer(object):
    def __init__(self):
        self.porter = PorterStemmer()

    def __call__(self, *args, **kwargs):
        return [self.porter.stem(word) for word in args[0].split()]


class TextRank(object):
    def __init__(self, document, d=0.85, preprocessor=None, stop=None, tokenizer=None):
        self.document = document
        self.d = d
        self.preprocessor = preprocessor
        self.stop = stop
        self.tokenizer = tokenizer

        self.sentences = self.get_sentences()
        self.no_of_sentences = len(self.sentences)

        self.tfdif_matrix, self.vocab = self.get_tfidf_vestors_and_vocab()

        self.directed_graph_weights_sentences = []
        self.ranks_of_sentences = []
        self.ranked_sentences = []
        self.ranked_sentences_idx = []

        self.directed_graph_weights_words = []
        self.ranks_of_words = []
        self.ranked_words = []

    def get_sentences(self):
        sentence_tokenizer = PunktSentenceTokenizer()
        return sentence_tokenizer.tokenize(self.document)

    def get_tfidf_vestors_and_vocab(self):
        tfidf = TfidfVectorizer(preprocessor=self.preprocessor, # currently always None
                                stop_words=self.stop,
                                tokenizer=self.tokenizer)
        mat = tfidf.fit_transform(self.sentences)
        vocab = tfidf.vocabulary_
        return [mat, vocab]

    def generate_graph_weights(self, scope="sentences"):
        if scope=="sentences":
            self.directed_graph_weights_sentences = self.tfdif_matrix * self.tfdif_matrix.T
            self.directed_graph_weights_sentences = self.directed_graph_weights_sentences.toarray()
        elif scope=="words":
            self.directed_graph_weights_words = self.tfdif_matrix.T * self.tfdif_matrix
            self.directed_graph_weights_words = self.directed_graph_weights_words.toarray()

    def generate_ranks(self, scope="sentences"):
        A = self.directed_graph_weights_sentences if scope=="sentences" else self.directed_graph_weights_words
        matrix_size = A.shape[0]
        for id in range(matrix_size):
            A[id, id] = 0
            col_sum = np.sum(A[:,id])
            if col_sum != 0:
                A[:, id] /= col_sum
            A[:, id] *= -self.d
            A[id, id] = 1
        B = (1-self.d) * np.ones((matrix_size, 1))
        if scope == "sentences":
            self.ranks_of_sentences = np.linalg.solve(A, B)
        else:
            self.ranks_of_words = np.linalg.solve(A, B)

    def generate_scores(self, scope="sentences"):
        if scope == "sentences":
            #for i,s in enumerate(self.sentences):
            #    print i
            #    print s
            self.ranked_sentences_idx = sorted(((self.ranks_of_sentences[i][0], i) for i,s in enumerate(self.sentences)), reverse=True)
            self.ranked_sentences = sorted(((self.ranks_of_sentences[i][0], s) for i,s in enumerate(self.sentences)), reverse=True)
        else:
            sorted_vocab_list = sorted(self.vocab, key=lambda k: self.vocab[k])
            self.ranked_words = sorted(((self.ranks_of_words[i][0], w) for i,w in enumerate(sorted_vocab_list)), reverse=True)

    def get_ranks(self, scope="sentences"):
        if scope=="sentences":
            return self.ranks_of_sentences
        else:
            return self.ranks_of_words

    def get_ranked_sentences(self):
        self.generate_graph_weights("sentences")
        self.generate_ranks("sentences")
        self.generate_scores("sentences")
        return self.ranked_sentences

    def get_ranked_sentences_idx(self):
        return self.ranked_sentences_idx

    def get_ranked_words(self):
        self.generate_graph_weights("words")
        self.generate_ranks("words")
        self.generate_scores("words")
        return self.ranked_words


def get_document(file_path):
    with open(file_path, 'r') as f:
        return f.read()


def print_format_table():
    """
    prints table of formatted text format options
    """
    for style in range(8):
        for fg in range(30,38):
            s1 = ''
            for bg in range(40,48):
                format = ';'.join([str(style), str(fg), str(bg)])
                #s1 += '\x1b[%sm %s \x1b[0m' % (format, format)
                s1 += '\x1b[{}m {} \x1b[0m'.format(format, format)
            print(s1)
        print('\n')


def color_sentences(sentences, ranked_sentences_idx):
    print "Hello"
    print ranked_sentences_idx
    p = ''

    #color_strength = 0
    #for rank, sentence_idx in ranked_sentences_idx:
    #    color_strength += 1
    #    p += '\x1b[{}m {} \x1b[0m'.format('1;30;4'+str(color_strength), sentences[sentence_idx])

    for i in range(len(sentences)):
        color_strength = 0
        for rank, sentence_idx in ranked_sentences_idx:
            color_strength += 1
            if i == sentence_idx:
                p += '\x1b[{}m {} \x1b[0m'.format('1;30;4'+str(color_strength), sentences[i])
                p += '\n'
    print p




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", help="path to filename")
    args = parser.parse_args()
    file_path = args.file
    #document = get_document(file_path)

    document = doc

    text_rank = TextRank(document=document,
                         tokenizer=PorterTokenizer(),
                         stop=stopwords.words('english'))
    ranked_sentences = text_rank.get_ranked_sentences()

    #color_sentences(text_rank.get_sentences(), text_rank.get_ranked_sentences_idx())

    ranked_words = text_rank.get_ranked_words()
    print ranked_sentences
    print ranked_words



    #print_format_table()
