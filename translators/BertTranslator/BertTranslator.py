from typing import Literal

import torch
from transformers import BertTokenizerFast, BertForMaskedLM

import lexicon
from translators.BaseTranslator import BaseTranslator

# sentence = f"A whiter shade of pale."
# word = "pale"
# masked_sentence = sentence.replace(word, tokenizer.mask_token)

# sequence = f"{sentence} {tokenizer.sep_token} {masked_sentence}"

# inputs = tokenizer.encode(sequence, return_tensors="pt")

# print(inputs)
# print(tokenizer.batch_decode(inputs))
# ## First mask token
# mask_token_index = (inputs == tokenizer.mask_token_id).nonzero(as_tuple=True)[1][0]

# outputs = model(inputs)
# ## Get logits for the first batch in the index of match_token_index for all vocab words
# mask_token_logits = outputs.logits[0, mask_token_index, :]
# top_5_options = torch.topk(mask_token_logits, 5).indices
# print(top_5_options)
# print(tokenizer.batch_decode(top_5_options))

# for token in top_5_options:
#     print(masked_sentence.replace(tokenizer.mask_token, tokenizer.decode([token])))

ALLOWED_MODELS = ["bert-base-uncased"]


class Translator(BaseTranslator):
    tokenizer: BertTokenizerFast
    model: BertForMaskedLM

    def __init__(self, model_name: str, lexicon=lexicon.load("most_common_1000")):
        if model_name not in ALLOWED_MODELS:
            raise ValueError(f"model_name must be one of {ALLOWED_MODELS}")
        self.tokenizer = BertTokenizerFast.from_pretrained(model_name)
        self.lexicon = [
            token
            for tokens in [enc.ids for enc in self.tokenizer.encode_batch(lexicon)]
            for token in tokens
        ]
        self.model = BertForMaskedLM.from_pretrained(model_name)

    def __identify_masked_tokens(self, tokens):
        pass

    def translate(self, sentence: str):
        input_tokens = self.tokenizer.encode(sentence)
