import warnings
from copy import deepcopy

import stanza
import torch
from conllu import parse
from sacremoses import MosesDetokenizer
from tqdm import tqdm

from lib.qanli.rule import AnswerSpan, Question
from src.eval.qa_loader import (W_QUESTION_STRUCTURAL_TYPES,
                                YES_NO_STRUCTURAL_TYPES)
from src.util.util import text_to_y


warnings.filterwarnings("ignore", category=FutureWarning)

detokenizer = MosesDetokenizer()
stanza.download("en")
nlp = stanza.Pipeline("en")


def to_tokens(language_string):
    doc = nlp(language_string)
    # assert len(doc.sentences) == 1
    conllu_format = []
    for word in doc.sentences[0].words:
        conllu_format.append(
            f"{word.id}\t{word.text}\t_\t{word.upos}\t{word.xpos}\t_\t{word.head}\t{word.deprel}\t_\t_"
        )
    return parse("\n".join(conllu_format))[0].tokens


def get_question_to_statements_function(structural_type, all_answers, config):
    r"""@return function that takes a question and returns a tensor on the device encoding that statement."""

    statement_fmt = config.get("eval.qa.statements", "rule")
    if statement_fmt == "question_and_answerWord":
        return lambda question: (
            text_to_y([question + " " + a for a in all_answers], config)
        )

    elif statement_fmt == "naive":
        return lambda question: text_to_y(
            [f"Question: {question} Answer: {a}" for a in all_answers], config
        )

    elif statement_fmt == "rule":
        if structural_type in W_QUESTION_STRUCTURAL_TYPES:
            s_gen = queryStatementGenerator
        elif structural_type in YES_NO_STRUCTURAL_TYPES:
            s_gen = yesNoStatementGenerator
        else:
            return None

        # generate answer tokens (only once!)
        a_token_list = [
            to_tokens(a)
            for a in tqdm(all_answers, desc=f"Tokenizing answers for {structural_type}")
        ]

        def getStatements_vec_on_device(question):
            # instantiate the generator with question tokens
            generator = s_gen(question, config)
            # run using question and answer token.
            return generator.gen_and_clip_list(a_token_list)

        return getStatements_vec_on_device
    else:
        raise ValueError(f"Unknown statement format {statement_fmt}")


class queryStatementGenerator:
    def __init__(self, question, config):
        # generate question tokens only once
        questionTokens = to_tokens(question)
        self.q = Question(questionTokens)
        if not self.q.isvalid:
            raise SyntaxError("Question is not valid.")
        self.config = config

    def _generate(self, answerTokens):
        a = AnswerSpan(answerTokens)
        q = deepcopy(self.q)
        q.insert_answer_default(a)
        return detokenizer.detokenize(q.format_declr(), return_str=True)

    def gen_and_clip_list(self, answerTokensList):
        statements = [self._generate(a) for a in answerTokensList]
        return text_to_y(statements, self.config)


class yesNoStatementGenerator:
    def __init__(self, question, config):
        self.q = question
        self.config = config

    def gen_and_clip_list(self, _):
        statement_yes = [f"{self.q} This is true."]
        yes_encoding = text_to_y(statement_yes, self.config)
        no_encoding = -yes_encoding
        return torch.cat(
            [yes_encoding, no_encoding]
        )  # uses enforced order from qa-loader: yes -> 0, no -> 1
