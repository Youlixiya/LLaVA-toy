# Copyright (c) Meta Platforms, Inc. and affiliates.
# Source from: https://github.com/facebookresearch/llama/blob/main/llama/tokenizer.py

import os
import numpy as np
import torch
from logging import getLogger
from typing import List

from sentencepiece import SentencePieceProcessor


logger = getLogger()


class TextTokenizer:
    
    box_token = '<box>'
    """Tokenizing and encoding/decoding text using SentencePiece."""

    def __init__(self, model_path=None):
        """
        Initializes the Tokenizer with a SentencePiece model.

        Args:
            model_path (str): The path to the SentencePiece model file.
        """
        if model_path is None:
            model_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "text_tokenizer.model"
            )
        # reload tokenizer
        assert os.path.isfile(model_path), model_path
        self.sp_model = SentencePieceProcessor(model_file=model_path)
        logger.info(f"Reloaded SentencePiece model from {model_path}")
        # BOS / EOS token IDs
        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.pad_id()
        self.box_id: int = self.sp_model.vocab_size()
        # self.n_words +=1
        self.pad_id += self.n_words if self.pad_id < 0 else 0
        logger.info(f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id}")
        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        """
        Encodes a string into a list of token IDs.

        Args:
            s (str): The input string to be encoded.
            bos (bool): Whether to prepend the beginning-of-sequence token.
            eos (bool): Whether to append the end-of-sequence token.

        Returns:
            List[int]: A list of token IDs.
        """
        assert type(s) is str
        if self.box_token in s:
            s_list = s.split(self.box_token)
            t = []
            for i, s in enumerate(s_list):
                t += self.sp_model.encode(s)
                if i != len(s_list)-1:
                    t.append(self.box_id)
        else:
            t = self.sp_model.encode(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        """
        Decodes a list of token IDs into a string.

        Args:
            t (List[int]): The list of token IDs to be decoded.

        Returns:
            str: The decoded string.
        """
        if self.bos_id in t:
            t = np.array(t)
            mask = (t == self.box_id)
            split_index = np.where(mask)[0]
            split_t = np.split(t, split_index)
            result = ''
            for i, t in enumerate(split_t):
                if i == 0:
                    result += self.sp_model.decode(t.tolist())
                else:
                    result += self.box_token
                    result += self.sp_model.decode(t[1:].tolist())
            return result
        else:
                # elif i == (len(split_t) -1)
        
            return self.sp_model.decode(t)

    def tokenize(self, texts, context_length=None, return_attn_masks=False):
        """Encode a list of string.

        Parameters
        ----------
        texts : Union[str, List[str]]
            The input text(s).
        context_length : int, optional
            The max token length.

        Returns
        -------
        List[List[int]]
            The encoded token indices.

        """
        if isinstance(texts, str):
            texts = [texts]
        max_len = 0
        tokens = []
        for text in texts:
            token = self.encode(text, bos=True, eos=True)
            max_len = len(token) if len(token) > max_len else max_len
            tokens.append(token)
        # tokens = [self.encode(text, bos=True, eos=True) for text in texts]
        if return_attn_masks:
            attn_masks = torch.ones((len(tokens), max_len))
        else:
            attn_masks = None
        for i in range(len(tokens)):
            token = tokens[i]
            pad_num = max_len - len(token)
            token += [self.pad_id] * pad_num
            tokens[i] = token
            if return_attn_masks:
                if pad_num > 0:
                    attn_masks[i, -pad_num:] = 0
        
        if context_length is None:
            return torch.LongTensor(tokens), attn_masks
        truncated_tokens = []
        for k, t in enumerate(tokens):
            if len(t) > context_length:
                t = t[:context_length]
                t[-1] = self.eos_id            
            truncated_tokens.append(t)
        if return_attn_masks:
            attn_masks = attn_masks[:, :context_length]
        return torch.tensor(truncated_tokens), attn_masks

    def detokenize(self, tokens):
        """Decode a list of string.

        Parameters
        ----------
        tokens : Union[List[List[int]], numpy.ndarray]
            The input tokens.

        Returns
        -------
        List[str]
            The decoded text strings.

        """
        if hasattr(tokens, "tolist"):
            tokens = tokens.tolist()
        texts = []
        for i in range(len(tokens)):
            t = tokens[i][1:]
            try:
                eot_idx = t.index(self.eos_id)
                t = t[:eot_idx]
            except ValueError:
                pass
            texts.append(self.decode(t))
        return texts
