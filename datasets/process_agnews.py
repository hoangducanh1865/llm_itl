"""
process text as BOW for topic model to learn
"""
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import WordNetLemmatizer
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import gensim.downloader as api
import numpy as np
from scipy import sparse
from operator import itemgetter
from scipy import io as sio
from datasets import load_dataset
import random


############## construct stop words ##############
nltk.download('stopwords')
nltk.download('wordnet')
# Load English stop words from both package
stop_words_nltk = list(stopwords.words('english'))
stop_words_sklearn = list(ENGLISH_STOP_WORDS)
# union
stop_words = list(set(stop_words_nltk) | set(stop_words_sklearn))
stop_words.extend(['doe', 'ha', 'le', 'u', 'wa'])
##################################################

def sparse2dense(input_matrix):
    if sparse.isspmatrix(input_matrix):
        input_matrix = input_matrix.toarray()
    input_matrix = input_matrix.astype('float32')
    return input_matrix


class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):
        token_pattern = re.compile(r"(?u)\b\w\w+\b")
        return [self.wnl.lemmatize(t) for t in doc.split() if len(t) >= 2 and re.match("[a-z].*", t)
                and re.match(token_pattern, t)]


def data_stats(data_file):
    data = sio.loadmat(data_file)
    print('Num Train Docs: ', data['bow_train'].shape[0])
    print('Num Test Docs: ', data['bow_test'].shape[0])
    print('Num labels: ', len(np.unique(data['label_train'])))
    print('Voc size: ', data['bow_train'].shape[1])

    train_np = sparse2dense(data['bow_train'])
    test_np = sparse2dense(data['bow_test'])
    all_bow = np.concatenate((train_np, test_np), axis=0)

    print('Avg length BOW: ', round(np.average(all_bow.sum(1)), 0))

    train_text = data['doc_train']
    test_text = data['doc_test']
    train_text = [str(item[0]) for item in train_text]
    test_text = [str(item[0]) for item in test_text]
    all_text = train_text + test_text

    lengths = [len(d.split()) for d in all_text]
    print('Avg length Text: ', round(sum(lengths)/len(lengths), 0))


if __name__ == '__main__':
    # run this after you create the ``.mat'' data file
    # data_stats('AGNews.mat')
    # quit()

    dataset_path = 'fancyzhx/ag_news'
    sample_size = 20000
    test_size = 0.2
    seed = 1
    random.seed(seed)

    # load raw data
    dataset = load_dataset(dataset_path, cache_dir='hf_data_cache', trust_remote_code=True)['train']

    # sample
    n_total = len(dataset)
    all_idx = [i for i in range(n_total)]
    idx_sub = random.sample(all_idx, sample_size)
    n_test = int(len(idx_sub) * test_size)

    idx_test = random.sample(idx_sub, n_test)  # test idx
    idx_train = [idx for idx in idx_sub if idx not in idx_test]  # train idx

    train_docs = itemgetter(*idx_train)(dataset['text'])
    train_labels = itemgetter(*idx_train)(dataset['label'])

    test_docs = itemgetter(*idx_test)(dataset['text'])
    test_labels = itemgetter(*idx_test)(dataset['label'])

    n_train = len(train_docs)
    docs = train_docs + test_docs

    # convert to BOW
    print('First fit ...')
    vectorizer = CountVectorizer(input='content', analyzer='word', stop_words=stop_words,
                                 tokenizer=LemmaTokenizer(), max_df=0.8, min_df=5, max_features=20000)
    vectorizer.fit_transform(docs)
    vocab_list = vectorizer.get_feature_names_out().tolist()

    # remove single character in voc
    for term in vocab_list:
        if len(term) == 1:
            vocab_list.remove(term)

    # keep only words in glove-50
    print('Process voc ...')
    model_glove = api.load("glove-wiki-gigaword-50")
    voc = []
    for i in range(len(vocab_list)):
        if vocab_list[i] in model_glove.wv.vocab:               # if word in glove
            voc.append(vocab_list[i])                           # keep in voc

    # rebuild BOW based on voc
    print('Second fit ...')
    vectorizer = CountVectorizer(input='content', analyzer='word', stop_words=stop_words,
                                 tokenizer=LemmaTokenizer(), max_df=0.8, min_df=5, max_features=20000,
                                 vocabulary=voc)

    data_crs_mat = vectorizer.fit_transform(docs)
    data_crs_mat = sparse.csr_matrix(data_crs_mat, dtype=np.float32)

    bow_train = data_crs_mat[0:n_train, :]
    bow_test = data_crs_mat[n_train:, :]

    # remove empty rows (all zero BOWs)
    nonzero_idx_train = np.where(bow_train.sum(1) != 0)[0]
    bow_train = bow_train[nonzero_idx_train]
    train_docs = itemgetter(*nonzero_idx_train)(train_docs)

    nonzero_idx_test = np.where(bow_test.sum(1) != 0)[0]
    bow_test = bow_test[nonzero_idx_test]
    test_docs = itemgetter(*nonzero_idx_test)(test_docs)

    # format and save
    voc_np = np.array(voc, dtype=np.object_)
    voc_np = voc_np.reshape((1, len(voc)))

    train_text_np = np.array(train_docs, dtype=np.object_)
    train_text_np = train_text_np.reshape((len(train_docs), 1))

    test_text_np = np.array(test_docs, dtype=np.object_)
    test_text_np = test_text_np.reshape((len(test_docs), 1))

    train_labels_np = np.array(train_labels).reshape((len(train_labels), 1))
    test_labels_np = np.array(test_labels).reshape((len(test_labels), 1))

    sio.savemat('AGNews.mat', {'bow_train': bow_train,
                           'bow_test': bow_test,
                           'voc': voc_np,
                           'label_train': train_labels_np,
                           'label_test': test_labels_np,
                           'doc_train': train_text_np,
                           'doc_test': test_text_np})
