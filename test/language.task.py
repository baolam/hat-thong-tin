import sys
sys.path.append("./")

from src.task.LanguageTask import GrammarUnit
grammar = GrammarUnit(32, 64, 16, [])

from torch import rand
x = rand((5, 64))

y = grammar.forward(x)
print(y)