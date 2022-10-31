import numpy as np
from datetime import datetime
from pathlib import Path
from typing import List, Union
from types import MethodType
from panphon import FeatureTable
from dataclasses import dataclass
from panphon import FeatureTable
from epitran import Epitran

@dataclass
class AlignContext:
    allo_ipas: List[str]
    ft: FeatureTable
    epi: Epitran

def createContext() -> AlignContext:
    # globals
    ipa_text_path = Path(__file__).parent / Path("../../data/allosaurus_ipas.txt")
    cedict_path = Path(__file__).parent / Path("../../data/cedict_1_0_ts_utf-8_mdbg.txt")
    allo_ipas = ipa_text_path.read_text(encoding="UTF-8").split()
    ft = FeatureTable()   
    epi = Epitran('cmn-Hant', cedict_file=str(cedict_path))  # type: ignore
    epi.epi.transliterate_char = MethodType(transliterate_char, epi.epi)  # type: ignore
    epi.transliterate_char = epi.epi.transliterate_char  # type: ignore
    return AlignContext(allo_ipas, ft, epi)

def softmax(x):
    return np.exp(x)/np.exp(x).sum(axis=1)[:,np.newaxis]

def to_seconds(time_str: Union[int, float, str]):
    if (isinstance(time_str, float) or 
        isinstance(time_str, int)):
        return time_str
    else:            
        ref = datetime(1900,1,1)
        delta = datetime.strptime(time_str, "%H:%M:%S.%f")-ref
        return delta.seconds + delta.microseconds/1000000

#https://github.com/dmort27/epitran/blob/a30eef02327af0f5f1d161fa427f9e56545b3b64/epitran/epihan.py
def transliterate_char(self, text):
    tokens = self.cedict.tokenize(text)
    ipa_tokens = []
    for token in tokens:
        if token in self.cedict.hanzi:
            (pinyin, _) = self.cedict.hanzi[token]            
            ipa = [self.rules.apply(pinyin_x)
                   for pinyin_x in pinyin]
            ipa = [x.replace(u',', u'') for x in ipa]
            ipa_tokens.extend(ipa)
        else:
            ipa_tokens.append(token)
    return ipa_tokens

def extract_dark_phones(phone_logits: np.ndarray, allo_ipas: List[str]):
    max_idxs = phone_logits.argsort()[::-1]
    if max_idxs[0] != 0:
        # return [ipa_tokens[i] for i in max_idxs[:1]]
        return None
    else:
        return allo_ipas[max_idxs[1]]