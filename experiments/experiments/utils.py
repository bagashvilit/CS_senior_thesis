from lib2to3.pgen2.tokenize import tokenize
from experiments.pycocoevalcap.meteor.meteor import Meteor
from experiments.pycocoevalcap.rouge.rouge import Rouge
from experiments.pycocoevalcap.bleu.bleu import Bleu
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cbook import boxplot_stats
import seaborn as sns
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


def text_similarity(predictions, golden_targets):
    # This source code is adapted from Wang et al 
    hypothesis = predictions

    res = {k: [v.strip().lower()] for k, v in enumerate(hypothesis)}

    references = golden_targets

    tgt = {k: [v.strip().lower()] for k, v in enumerate(references)}

    score_Bleu, scores_Bleu = Bleu().compute_score(tgt, res)
    score_Meteor, scores_Meteor = Meteor().compute_score(tgt, res)
    score_Rouge, scores_Rouge = Rouge().compute_score(tgt, res)


    return (score_Meteor, scores_Meteor,score_Rouge,scores_Rouge)

def box_plot(copynet_scores, corec_scores, name):

    scores = {"0":copynet_scores, "1":corec_scores}
    scores_frame = pd.DataFrame(scores)

    # hide the median line
    medianlineprops = dict(linestyle='--', color='black')

    sns.set(rc = {'figure.figsize':(8,6)})
    sns.set_style("white")
    box_plot = sns.boxplot(data=scores_frame, color='grey',medianprops=medianlineprops)

    ax = box_plot.axes
    lines = ax.get_lines()
    plt.yticks(fontsize=14)
    plt.xticks([0,1], ["CopyNet","CoRec"], fontsize=14)
    categories = ax.get_xticks()

    for cat in categories:

        stats = boxplot_stats(scores_frame[str(cat)])

        median =  round(stats[0]["med"],2)

        ax.text(
            cat, 
            median, 
            f'{median}', 
            ha='center', 
            va='center', 
            fontweight='bold', 
            size=10,
            color='black',
            bbox=dict(facecolor='#d3d3d3'))

    box_plot.figure.savefig("experiments/plots/{}".format(name))
    plt.clf()


def count_low_frequency_word(train, reference):
    

    reference_tokenized = tokenize_msg(reference)
    train_tokenized = tokenize_msg(train)


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


def corec_vs_copy_low_frequency(train_messages, reference_messages, corec_messages, copynet_messages):

    low_frequency_words = count_low_frequency_word(train_messages, reference_messages)

    corec_tokenized = tokenize_msg(corec_messages)
    copy_tokenized = tokenize_msg(copynet_messages)

    corec_all_words = list(numpy.concatenate(corec_tokenized).flat)
    copy_all_words = list(numpy.concatenate(copy_tokenized).flat)

    copy = 0
    for i in low_frequency_words:
        if i in copy_all_words:
            copy += 1
    print("Count of low frequency words by CopyNet: ",copy)

    corec = 0
    for i in low_frequency_words:
        if i in corec_all_words:
            corec += 1

    print("Count of low frequency words by CoRec: ", corec)



def low_freq_msgs(corec_messages, copynet_messages, reference_messages, train_messages):

    tokenized_target = tokenize_msg(reference_messages)
    low_frequency_words = count_low_frequency_word(train_messages, reference_messages)

    low_frequency_indeces = []

    for i in range(len(tokenized_target)):
        for j in tokenized_target[i]:
            if j in low_frequency_words:
                low_frequency_indeces.append(i)

    low_frequency_indeces = list(set(low_frequency_indeces))

    low_frequency_target = []
    for i in low_frequency_indeces:
        low_frequency_target.append(reference_messages[i])


    low_frequency_copy = []
    for i in low_frequency_indeces:
        low_frequency_copy.append(copynet_messages[i])


    low_frequency_corec = []
    for i in low_frequency_indeces:
        low_frequency_corec.append(corec_messages[i])

    print(len(low_frequency_target),'\n', len(low_frequency_copy),'\n', len(low_frequency_corec))
    print(len(list(set(low_frequency_words))))
    
    return(low_frequency_target, low_frequency_copy, low_frequency_corec)


