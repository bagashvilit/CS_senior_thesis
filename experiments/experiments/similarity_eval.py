from CoRec.CoRec.evaluation.pycocoevalcap.meteor.meteor import Meteor
from CoRec.CoRec.evaluation.pycocoevalcap.rouge.rouge import Rouge
from CoRec.CoRec.evaluation.pycocoevalcap.bleu.bleu import Bleu
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cbook import boxplot_stats
import seaborn as sns
import pingouin as pg

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


def similarity_score(copynet_messages, corec_messages,  reference_messages):

    corec_Meteor, corec_scores_Meteor, corec_Rouge, corec_scores_Rouge = text_similarity(corec_messages,  reference_messages)
    copy_Meteor, copy_scores_Meteor, copy_Rouge, copy_scores_Rouge = text_similarity(copynet_messages, reference_messages)

    print("\nCoRec Meteor: ", corec_Meteor, "CoRec Rouge: ", corec_Rouge)
    print("CopyNet Meteor: ", copy_Meteor, "CopyNet Rouge: ", copy_Rouge)

    box_plot(copy_scores_Rouge, corec_scores_Rouge, "rouge.png")
    box_plot(copy_scores_Meteor, corec_scores_Meteor, "meteor.png")

    print("\nMan-Whitney U statistical test results for ROUGE scores")
    print(pg.mwu(corec_scores_Rouge, copy_scores_Rouge, alternative='two-sided'))

    print("Man-Whitney U statistical test results for Meteor scores")
    print(pg.mwu(corec_scores_Meteor, copy_scores_Meteor, alternative='two-sided'))