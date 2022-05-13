from experiments import utils

import typer
import numpy

from experiments.utils import word_analysis

app = typer.Typer()

reference_messages = utils.read_data_file("experiments/data/reference.msg")
copynet_messages = utils.read_data_file("experiments/data/copynet.msg")
corec_messages = utils.read_data_file("experiments/data/corec.msg")
train_messages = utils.read_data_file("experiments/data/train.msg")

evaluator = word_analysis(train_messages, reference_messages, corec_messages, copynet_messages)

@app.command()
def similarity(low: bool=False):

    if low:
        low_frequency_target, low_frequency_copy, low_frequency_corec = evaluator.low_freq_msgs()
        utils.similarity_score(low_frequency_copy, low_frequency_corec, low_frequency_target)
    else:
        utils.similarity_score(copynet_messages, corec_messages, reference_messages)



@app.command()
def count():

    print("Count of correctly predicted low frequency words")
    evaluator.corec_vs_copy_low_frequency()


    print("\nCount of correctly predicted oov words")
    evaluator.corec_vs_copy_oov()



    
