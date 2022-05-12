from experiments import utils
from experiments.utils import box_plot
import pingouin as pg
import typer
import numpy

app = typer.Typer()

reference_messages = utils.read_data_file("experiments/data/reference.msg")
copynet_messages = utils.read_data_file("experiments/data/copynet.msg")
corec_messages = utils.read_data_file("experiments/data/corec.msg")
train_messages = utils.read_data_file("experiments/data/train.msg")

@app.command()
def similarity():

    corec_Meteor, corec_scores_Meteor, corec_Rouge, corec_scores_Rouge = utils.text_similarity(corec_messages,  reference_messages)
    copy_Meteor, copy_scores_Meteor, copy_Rouge, copy_scores_Rouge = utils.text_similarity(copynet_messages, reference_messages)

    print("CoRec Meteor: ", corec_Meteor, "CoRec Rouge: ", corec_Rouge)
    print("CopyNet Meteor: ", copy_Meteor, "CopyNet Rouge: ", copy_Rouge)

    box_plot(copy_scores_Rouge, corec_scores_Rouge, "rouge.png")
    box_plot(copy_scores_Meteor, corec_scores_Meteor, "meteor.png")

    print("Man-Whitney U statistical test results for ROUGE scores")
    print(pg.mwu(corec_scores_Rouge, copy_scores_Rouge, alternative='two-sided'))

    print("Man-Whitney U statistical test results for Meteor scores")
    print(pg.mwu(corec_scores_Meteor, copy_scores_Meteor, alternative='two-sided'))



@app.command()
def count():

    utils.corec_vs_copy_low_frequency(train_messages, reference_messages, corec_messages, copynet_messages)



@app.command()
def similaritylow():

    low_frequency_target, low_frequency_copy, low_frequency_corec = utils.low_freq_msgs(corec_messages, copynet_messages, reference_messages, train_messages)

    corec_Meteor, corec_scores_Meteor, corec_Rouge, corec_scores_Rouge = utils.text_similarity(low_frequency_corec,  low_frequency_target)
    copy_Meteor, copy_scores_Meteor, copy_Rouge, copy_scores_Rouge = utils.text_similarity(low_frequency_copy, low_frequency_target)

    print("CoRec Meteor: ", corec_Meteor, "CoRec Rouge: ", corec_Rouge)
    print("CopyNet Meteor: ", copy_Meteor, "CopyNet Rouge: ", copy_Rouge)

    box_plot(copy_scores_Rouge, corec_scores_Rouge, "rouge.png")
    box_plot(copy_scores_Meteor, corec_scores_Meteor, "meteor.png")

    print("Man-Whitney U statistical test results for ROUGE scores")
    print(pg.mwu(corec_scores_Rouge, copy_scores_Rouge, alternative='two-sided'))

    print("Man-Whitney U statistical test results for Meteor scores")
    print(pg.mwu(corec_scores_Meteor, copy_scores_Meteor, alternative='two-sided'))
    
