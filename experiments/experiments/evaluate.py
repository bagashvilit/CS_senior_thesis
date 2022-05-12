from experiments import utils
from experiments.utils import box_plot

import typer
import numpy

app = typer.Typer()

reference_messages = utils.read_data_file("experiments/data/reference.msg")
copynet_messages = utils.read_data_file("experiments/data/copynet.msg")
corec_messages = utils.read_data_file("experiments/data/corec.msg")
train_messages = utils.read_data_file("experiments/data/train.msg")

@app.command()
def similarity(low: bool=False):

    if low:
        low_frequency_target, low_frequency_copy, low_frequency_corec = utils.low_freq_msgs(corec_messages, copynet_messages, reference_messages, train_messages)
        utils.similarity_score(low_frequency_copy, low_frequency_corec, low_frequency_target)
    else:
        utils.similarity_score(copynet_messages, corec_messages, reference_messages)



@app.command()
def count():

    utils.corec_vs_copy_low_frequency(train_messages, reference_messages, corec_messages, copynet_messages)




    
