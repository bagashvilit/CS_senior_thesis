from experiments import __version__
from experiments import utils
import pytest
from experiments.utils import word_analysis

def test_version():
    assert __version__ == '0.1.0'

train_messages = ["add new dependencies", "update dependencies", "update readme", "update tests", "update update"]
reference_messages = ["Add new dependencies", "Update project dependencies"]
copynet_messages = ["add dependencies", "update project dependencies"]
corec_messages = ["add dependencies", "update dependencies"]

evaluator = word_analysis(train_messages, reference_messages, corec_messages, copynet_messages)

@pytest.mark.parametrize(
    "input_message,tokens",
    [
        (["fix typo"], [["fix","typo"]]),
        (["Add missing test dependency"], [["add", "missing", "test", "dependency"]])

    ],
)
def test_tokenize_msg(input_message, tokens):
    assert utils.tokenize_msg(input_message) == tokens


def test_count_low_frequency_word():
    assert len(evaluator.count_low_frequency_word()) == 5


def test_oov_words_count():
    assert len(evaluator.oov_words_count()) == 1

def test_corec_vs_copy_low_frequency():
    assert evaluator.corec_vs_copy_low_frequency() == (5, 4, 3)


def test_corec_vs_copy_oov():
    assert evaluator.corec_vs_copy_oov() == (1, 1, 0)