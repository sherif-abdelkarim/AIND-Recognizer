import warnings
from asl_data import SinglesData
import math

def compute_score(model, Y, Ylengths):
  try:
    return model.score(Y, Ylengths)
  except:
    return -math.inf

def recognize(models: dict, test_set: SinglesData):
  """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
  warnings.filterwarnings("ignore", category=DeprecationWarning)
  probabilities = [{word: compute_score(model,Y,Ylengths) for word, model in models.items()} for key,[Y, Ylengths] in test_set.get_all_Xlengths().items()]
  guesses = [max(probability, key=probability.get) for probability in probabilities]
  return (probabilities, guesses)