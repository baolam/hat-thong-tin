from typing import List, Dict
from torch import nn
from torch import Tensor
from ..atom.single.StandardUnit import StandardUnit
from ..atom.single.WordUnit import WordUnit


# Một số loại từ (xây dựng cho tiếng việt)
QUERY_TAGS = [ "danh_tu", "dong_tu", "tinh_tu", 
    "so_tu", "luong_tu", "pho_tu", "dai_tu", "chi_tu", "quan_he_tu",
    "tro_tu", "than_tu", "tinh_thai_tu" ]
    

class GrammarUnit(StandardUnit):
    def __init__(self, gamma: int, beta: int, hidden: int,
        struct_grams : List[str]):
        super().__init__(gamma, beta, hidden)
        self.struct_grams = struct_grams

    def add_exp(self):
        pass

    def add_relate(self):
        pass

    def best_kunit(self):
        pass

    def forward(self, x: Tensor) -> Tensor:
        __, e = super().forward(x, is_existed = True)
        return e
    

class LanguageModel(nn.Module):
    # Đường dẫn cho vấn đề ngôn ngữ tiếng việt
    # https://github.com/winstonleedev/tudien
    def __init__(self, gamma : int, beta : int,
        query_tags : List[str] = QUERY_TAGS):
        super().__init__()
        # Các tham số của đơn vị từ
        self.gamma = gamma
        self.beta = beta
        # Là các loại từ dùng để xây dựng cấu trúc ngữ pháp
        self.query_tags = query_tags
        # Lưu trữ các cấu trúc ngữ pháp
        self.grammars : List[GrammarUnit] = nn.ModuleList()
        self.tag_words : Dict[str, List[str]] = {}
        self.__grammar_words()
        # Lưu trữ các từ
        self.words : Dict[str, WordUnit] = nn.ModuleDict()

    def __grammar_words(self):
        for tag in self.query_tags:
            self.tag_words[tag] = []

    def add_tag(self, word : str, tag : str):
        # Loại từ không tồn tại
        if self.tag_words.get(tag) is None:
            return
        # Từ đã tồn tại
        __words = self.tag_words[tag]
        for _word in __words:
            if word == _word:
                return
            
        self.tag_words[tag].append(word)

    def add_word(self, hidden : int ,word : str, tag : str):
        unit = WordUnit(self.gamma, self.beta, hidden, word)

        if self.words[word] is None:
            self.words[word] = unit
            self.add_tag(word, tag)

    def add_grammar(self, hidden : int, struct_grams : List[str]):
        grammar = GrammarUnit(self.gamma, self.beta, hidden, struct_grams)
        # Chắc chắn struct_grams đã khác nhau
        self.grammars.append(grammar)