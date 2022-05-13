import numpy
from collections import Counter

def read_data_file(path):
    with open(path) as msg_f:
        data = msg_f.read().splitlines()
        return data


def tokenize_msg(messages):
    tokenized = []
    for i in messages:
        split_space = i.split(' ')
        lower = [i.lower() for i in split_space]
        tokenized.append(lower)

    return(tokenized)


class word_analysis:
    def __init__(self, train_messages, reference_messages, corec_messages, copynet_messages):
        self.train_messages = train_messages
        self.reference_messages = reference_messages
        self.corec_messages = corec_messages
        self.copynet_messages = copynet_messages


    def count_low_frequency_word(self):
        

        reference_tokenized = tokenize_msg(self.reference_messages)
        train_tokenized = tokenize_msg(self.train_messages)


        target_all_words = list(numpy.concatenate(reference_tokenized).flat)
        train_all_words = list(numpy.concatenate(train_tokenized).flat)


        target_frequencies = Counter(target_all_words)
        train_frequencies = Counter(train_all_words)

        updated_frequencies = {}
        for key, value in target_frequencies.items():
            updated_frequencies.update({key:train_frequencies[key]})

        # make a list of low frequency words, count less than 5
        low_frequency_words = []

        for key, value in updated_frequencies.items():
            if value <= 5:
                low_frequency_words.append(key)

        return low_frequency_words


    def corec_vs_copy_low_frequency(self):

        low_frequency_words = self.count_low_frequency_word()
        total = len(low_frequency_words)

        corec_tokenized = tokenize_msg(self.corec_messages)
        copy_tokenized = tokenize_msg(self.copynet_messages)

        corec_all_words = list(numpy.concatenate(corec_tokenized).flat)
        copy_all_words = list(numpy.concatenate(copy_tokenized).flat)

        copy = 0
        for i in low_frequency_words:
            if i in copy_all_words:
                copy += 1

        corec = 0
        for i in low_frequency_words:
            if i in corec_all_words:
                corec += 1

        return(total, copy,corec)
        


    def low_freq_msgs(self):

        tokenized_target = tokenize_msg(self.reference_messages)
        low_frequency_words = self.count_low_frequency_word()

        low_frequency_indeces = []

        for i in range(len(tokenized_target)):
            for j in tokenized_target[i]:
                if j in low_frequency_words:
                    low_frequency_indeces.append(i)

        low_frequency_indeces = list(set(low_frequency_indeces))

        low_frequency_target = []
        for i in low_frequency_indeces:
            low_frequency_target.append(self.reference_messages[i])


        low_frequency_copy = []
        for i in low_frequency_indeces:
            low_frequency_copy.append(self.copynet_messages[i])


        low_frequency_corec = []
        for i in low_frequency_indeces:
            low_frequency_corec.append(self.corec_messages[i])

        return(low_frequency_target, low_frequency_copy, low_frequency_corec)



    def oov_words_count(self):

        oov_words = []

        reference_tokenized = tokenize_msg(self.reference_messages)
        train_tokenized = tokenize_msg(self.train_messages)


        target_all_words = list(numpy.concatenate(reference_tokenized).flat)
        train_all_words = list(numpy.concatenate(train_tokenized).flat)

        for i in target_all_words:
            if i not in train_all_words:
                if i not in oov_words:
                    oov_words.append(i)
        return oov_words


    def corec_vs_copy_oov(self):

        oov_words = self.oov_words_count()
        total = len(oov_words)

        corec_tokenized = tokenize_msg(self.corec_messages)
        copy_tokenized = tokenize_msg(self.copynet_messages)

        corec_all_words = list(numpy.concatenate(corec_tokenized).flat)
        copy_all_words = list(numpy.concatenate(copy_tokenized).flat)

        copy = 0
        for i in oov_words:
            if i in copy_all_words:
                copy += 1
        

        corec = 0
        for i in oov_words:
            if i in corec_all_words:
                corec += 1

        return (total, copy, corec)
        



