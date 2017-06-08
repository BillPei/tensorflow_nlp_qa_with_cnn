import string
import numpy as np
import csv
from nltk import word_tokenize
import random
filename = '/Users/zoesh/Desktop/Takehome/employee_channel/Acme_dataset.csv'
train_filename = '/Users/zoesh/Desktop/Takehome/employee_channel/train_question.csv'
test_filename = '/Users/zoesh/Desktop/Takehome/employee_channel/test_question.csv'

def build_vocab():
    code = int(0)
    vocab = {}
    vocab['UNKNOWN'] = code
    code += 1
    translator = str.maketrans('', '', string.punctuation)
    with open(filename,encoding="latin1") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for x in reader:
            combin = x[1] + ' ' + x[2]
            combin = combin.translate(translator)
            for token in word_tokenize(combin.lower()):
                if not token in vocab:
                    vocab[token] = code
                    code+=1
    return vocab

def read_ans_list():
    alist = []
    with open(train_filename,encoding="latin1") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for x in reader:
            alist.append(x[2])
    return np.array(alist)

def read_raw_file():
    raw =[]
    with open(train_filename,encoding="latin1") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for x in reader:
            raw.append(x)
    return np.array(raw)

def encode_sent(vocab, sentence):
    x = []
    translator = str.maketrans('', '', string.punctuation)
    words = sentence.lower().translate(translator).split(' ')
    for i in range(0, 200):
        if ((len(words))<=i) or (words[i] not in vocab):
            x.append(vocab['UNKNOWN'])
        else:
            x.append(vocab[words[i]])
    return x

def load_data(vocab, alist, raw):
    question = []
    answer = []
    rand_answer = []
    for i in range(0, len(alist)):
        items = raw[random.randint(0, len(raw) - 1)]
        nega = alist[random.randint(0, len(alist) - 1)]
        question.append(encode_sent(vocab, items[1]))
        answer.append(encode_sent(vocab, items[2]))
        rand_answer.append(encode_sent(vocab, nega))
    return np.array(question), np.array(answer), np.array(rand_answer)

def load_test_list():
    test_list = []
    with open(test_filename,encoding="latin1") as csvfile:
        reader = csv.reader(csvfile)
        #skip header
        next(reader)
        for x in reader:
            test_list.append(x)
    return np.array(test_list)

def load_test_data(test_list, vocab, index, batch):
    x_train_1 = []
    x_train_2 = []
    x_train_3 = []
    for i in range(0, batch):
        true_index = index + i
        if (true_index >= len(test_list)):
            true_index = len(test_list) - 1
        items = test_list[true_index]
        #questions
        x_train_1.append(encode_sent(vocab, items[1]))
        #answer
        x_train_2.append(encode_sent(vocab, items[2]))
        #same answer as x_train_2
        x_train_3.append(encode_sent(vocab, items[2]))
    return np.array(x_train_1), np.array(x_train_2), np.array(x_train_3)

def batch_iter(vocab, alist, raw, batch_size, num_epochs, shuffle=True):
    for epoch in range(num_epochs):
        #load and shuffle the data at each epoch
        (x_train_1,x_train_2,x_train_3) = load_data(vocab, alist, raw)
        data_size = len(x_train_1)
        num_batches_per_epoch = int(data_size/batch_size)
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_x_train_1 = x_train_1[shuffle_indices]
        shuffled_x_train_2 = x_train_2[shuffle_indices]
        shuffled_x_train_3 = x_train_3[shuffle_indices]
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = (batch_num + 1) * batch_size
            yield shuffled_x_train_1[start_index:end_index],\
                    shuffled_x_train_2[start_index:end_index],\
                    shuffled_x_train_3[start_index:end_index]
    


