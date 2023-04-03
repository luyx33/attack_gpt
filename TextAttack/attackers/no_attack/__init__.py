from typing import List, Optional
import numpy as np

from ...metric import UniversalSentenceEncoder
from ..classification import ClassificationAttacker, Classifier, ClassifierGoal
from ...text_process.tokenizer import Tokenizer, get_default_tokenizer
from ...attack_assist.substitute.word import WordSubstitute, get_default_substitute
from ...utils import get_language, check_language, language_by_name
from ...exceptions import WordNotInDictionaryException
from ...tags import Tag
from ...attack_assist.filter_words import get_default_filter_words

class TestAttacker(ClassificationAttacker):
    @property
    def TAGS(self):
        return { self.__lang_tag, Tag("get_pred", "victim"), Tag("get_prob", "victim") }

    def __init__(self,
            import_score_threshold : float = -1,
            sim_score_threshold : float = 0.5,
            sim_score_window : int = 15,
            tokenizer : Optional[Tokenizer] = None,
            substitute : Optional[WordSubstitute] = None,
            filter_words : List[str] = None,
            token_unk = "<UNK>",
            lang = None,
            model_dir = ''
        ):
        """
        Is BERT Really Robust? A Strong Baseline for Natural Language Attack on Text Classification and Entailment. Di Jin, Zhijing Jin, Joey Tianyi Zhou, Peter Szolovits. AAAI 2020.
        `[pdf] <https://arxiv.org/pdf/1907.11932v4>`__
        `[code] <https://github.com/jind11/TextFooler>`__

        Args:
            import_score_threshold: Threshold used to choose important word. **Default:** -1.
            sim_score_threshold: Threshold used to choose sentences of high semantic similarity. **Default:** 0.5
            im_score_window: length used in score module. **Default:** 15
            tokenizer: A tokenizer that will be used during the attack procedure. Must be an instance of :py:class:`.Tokenizer`
            substitute: A substitute that will be used during the attack procedure. Must be an instance of :py:class:`.WordSubstitute`
            lang: The language used in attacker. If is `None` then `attacker` will intelligently select the language based on other parameters.
            token_unk: The token id or the token name for out-of-vocabulary words in victim model. **Default:** ``"<UNK>"``
            filter_words: A list of words that will be preserved in the attack procesudre.

        :Classifier Capacity:
            * get_pred
            * get_prob
        
        """
        lst = []
        if tokenizer is not None:
            lst.append(tokenizer)
        if substitute is not None:
            lst.append(substitute)
        if len(lst) > 0:
            self.__lang_tag = get_language(lst)
        else:
            self.__lang_tag = language_by_name(lang)
            if self.__lang_tag is None:
                raise ValueError("Unknown language `%s`" % lang)
        self.model_dir = model_dir
        
        if substitute is None:
            substitute = get_default_substitute(self.__lang_tag, self.model_dir)
        self.substitute = substitute

        if tokenizer is None:
            tokenizer = get_default_tokenizer(self.__lang_tag)
        self.tokenizer = tokenizer

        self.sim_predictor = UniversalSentenceEncoder()

        check_language([self.tokenizer, self.substitute, self.sim_predictor], self.__lang_tag)

        self.import_score_threshold = import_score_threshold
        self.sim_score_threshold = sim_score_threshold
        self.sim_score_window = sim_score_window

        if filter_words is None:
            filter_words = get_default_filter_words(self.__lang_tag)
        self.filter_words = set(filter_words)

        self.token_unk = token_unk

    def attack(self, victim: Classifier, sentence : str, goal: ClassifierGoal):
        """
        * **clsf** : **Classifier** .
        * **x_orig** : Input sentence.
        """
        x_orig = sentence.lower()
        pred = victim.get_pred([x_orig])[0]
        # pred1 = victim.get_pred([x_orig])
        # pred2 = victim.get_pred([x_orig])
        # print("pred---------",pred,pred1,pred2)
        if goal.check(x_orig, pred):
            return x_orig
        else:
            return None
            

    def get_neighbours(self, word, pos):
        try:
            return list(
                filter(
                    lambda x: x != word,
                    map(
                        lambda x: x[0],
                        self.substitute(word, pos),
                    )
                )
            )
        except WordNotInDictionaryException:
            return []


    def pos_filter(self, ori_pos, new_pos_list):
        same = [True if ori_pos == new_pos or (set([ori_pos, new_pos]) <= set(['noun', 'verb']))
                else False for new_pos in new_pos_list]
        return same