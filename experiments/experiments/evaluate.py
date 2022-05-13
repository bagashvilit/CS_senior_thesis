from experiments import utils

import typer
import numpy
from experiments import similarity_eval
from experiments.utils import word_analysis

app = typer.Typer()

reference_messages = utils.read_data_file("CoRec/CoRec/data/top1000/cleaned.test.msg")
copynet_messages = utils.read_data_file("experiments/data/copynet.msg")
corec_messages = utils.read_data_file("CoRec/CoRec/result/CoRec/cleaned.test.msg")
train_messages = utils.read_data_file("CoRec/CoRec/data/top1000/cleaned.train.msg")

evaluator = word_analysis(train_messages, reference_messages, corec_messages, copynet_messages)

@app.command()
def similarity(low: bool=False):

    if low:
        low_frequency_target, low_frequency_copy, low_frequency_corec = evaluator.low_freq_msgs()
        similarity_eval.similarity_score(low_frequency_copy, low_frequency_corec, low_frequency_target)
    else:
        similarity_eval.similarity_score(copynet_messages, corec_messages, reference_messages)



@app.command()
def count():

    print("Count of correctly predicted low frequency words")
    total_low, copy_low, corec_low = evaluator.corec_vs_copy_low_frequency()
    print("Count of low frequency words in reference messages: ", total_low)
    print("Count of low frequency words by CopyNet: ",copy_low)
    print("Count of low frequency words by CoRec: ", corec_low)

    print("\nCount of correctly predicted oov words")
    total_oov, copy_oov, corec_oov = evaluator.corec_vs_copy_oov()
    print("Total count of oov words in reference messages: ", total_oov)
    print("Count of low frequency words by CopyNet: ",copy_oov)
    print("Count of low frequency words by CoRec: ", corec_oov)




    
