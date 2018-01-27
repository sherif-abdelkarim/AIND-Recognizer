from my_recognizer import recognize
from asl_utils import show_errors
from asl_utils import train_all_words
from my_model_selectors import SelectorCV, SelectorBIC, SelectorDIC
from preprocesser import *

features_list = [features_ground, features_polar, features_norm, features_delta, features_custom] # change as needed
#features_list = [features_custom]
model_selectors = [SelectorCV, SelectorBIC, SelectorDIC] # change as needed
# TODO Recognize the test set and display the result with the show_errors method
for features in features_list:
    for model_selector in model_selectors:
        print(features)
        print(model_selector)
        training = asl.build_training(features)
        models = train_all_words(training, model_selector)
        test_set = asl.build_test(features)
        probabilities, guesses = recognize(models, test_set)
        show_errors(guesses, test_set)
