from random import randint, choice, shuffle, random, sample, choices
from decimal import Decimal
from abc import ABC, abstractmethod

from typing import Callable
from pylatex import Math, Command, Tabular, MiniPage, Document, VerticalSpace, StandAloneGraphic, TikZ, Axis, Plot, TikZDraw
from pylatex.base_classes import CommandBase
from pylatex.utils import NoEscape
import math
import numpy as np
import sympy as sp

from classes.math_objects import *


class DocInjector:
    def __init__(self, body: Callable[[Document], None]):
        """
        Allows injecting a function into Document.
        Avoiding using this unless necessary (e.g. to use an environment).

        :param body: The function to be injected.
        """
        self.body = body

    def inject(self, doc: Document):
        """
        Injects the function into Document.
        :param doc: The Document instance.
        """
        self.body(doc)


class ProblemBase(ABC):
    def __init__(self, num_quest: int, vspace: str):
        """
        Initializes a problem.
        :param num_quest: The number of questions to be generated.
        :param vspace: The minimum vertical space to be added below each problem.
        """
        self.num_quest = num_quest
        self.vspace = vspace

    @abstractmethod
    def get_problem(self) -> list[Math | NoEscape | CommandBase | DocInjector]:
        """
        Returns a practice problem.
        Decrements num_quest in the object.
        Call .inject() for DocInjector with Document passed in,
        and append other types directly to the Document.
        """
        pass   # child class should implement this


class WordProblem(ProblemBase):
    def __init__(self, num_quest: int, vspace: str, name: str, ranges: list[tuple[int, int]]):
        """
        Initializes a word problem. Make sure word_problems.txt is in the same directory as where this was called.
        
        This class supports two types of placeholders:
          - Numeric placeholders (@): Replaced with random integers from specified ranges.
          - String placeholders (`): Replaced with randomly selected strings from predefined options.
            The same number of placeholder appearing multiple times will be replaced with the same value.
        
        :param num_quest: The number of questions to be generated.
        :param vspace: The minimum vertical space to be added below each problem.
        :param name: The name of the problem to be used to choose problems.
        :param ranges: The ranges of values to be used for numeric (@) parameters only.
                       They should be in the order corresponding to the number of @'s
                       (i.e. the first element specifies @, the second element specifies @@, etc.)
                       String placeholders (`) are handled separately from the file definition.
        """
        super().__init__(num_quest, vspace)
        self.ranges = ranges
        self.string_options = []  # list of options (list) for string placeholders
        
        with open("word_problems.txt", 'r') as f:
            line = f.readline()
            found = False
            while line and not found:
                if line.startswith('!'):
                    # problem start detected
                    problem_name = line[1:].rstrip()  # remove the '!' marker
                    if problem_name == name:
                        # problem found
                        found = True
                        self.problem = f.readline().rstrip()
                        num_params = f.readline().rstrip().split(' ')   # num_numeric, num_string
                        if int(num_params[0]) != len(ranges):
                            raise ValueError(f"The number of ranges ({len(ranges)}) doesn't agree with the number of problem parameters ({num_params[0]})")
                        # make sure the problem contains all placeholders
                        assert self.problem.count('@') == len(ranges), "the number of placeholders does not agree with the number of parameters"

                        if len(num_params) == 2 and int(num_params[1]) != 0:
                            # has string parameters, extract string parameter values
                            num_string = int(num_params[1])
                            # make sure the problem contains all placeholders
                            for n in range(1, num_string + 1):
                                assert self._find_placeholder(self.problem, '`', n), f"{num_string} string parameters expected, but could not find {'`'*n}"
                            n = 1
                            while n <= num_string:
                                next_line = f.readline()
                                placeholder, pattern = next_line.strip().split('=')
                                assert placeholder == '`' * n, f"placeholder value for {'`'*n} is either missing or in incorrect order"
                                self.string_options.append(pattern.split('|'))
                                n += 1
                        elif len(num_params) > 2:
                            # more than 2 parameters, illegal
                            raise ValueError(f"too many parameter counts ({len(num_params)}) for problem {problem_name}")
                    else:
                        line = f.readline()
                else:
                    line = f.readline()
            
            if not found:
                raise ValueError(f"{name} not found in the file")

    @staticmethod
    def _find_placeholder(text: str, placeholder: str, count: int) -> list[int]:
        """
        Finds starting indices of a placeholder repeated a certain number of times.
        For example, when count is 2, @@ is detected but @@@ is not.
        """
        assert count > 0, "count must be positive"
        indices = []
        target = placeholder * count
        i = 0
        while i < len(text):
            if text[i:i + count] == target:
                before = text[i-1 : i]   # use slice to avoid overflow index error
                after = text[i+count : i+count+1]
                # ensure not part of a larger cluster
                if before != placeholder and after != placeholder:
                    indices.append(i)
                    i += count
                else:
                    i += 1
            else:
                i += 1
        return indices

    @staticmethod
    def _replace_placeholders(text: str, placeholder: str, count: int, replacement: str) -> str:
        indices = WordProblem._find_placeholder(text, placeholder, count)

        if not indices:
            return text

        result = []
        last = 0

        for i in indices:
            result.append(text[last:i])
            result.append(replacement)
            last = i + count

        result.append(text[last:])
        return "".join(result)


    def get_problem(self) -> list[NoEscape]:
        """
        Generates a problem instance with randomized values.
        
        Steps:
        1. Replace string placeholders (#n) with randomly selected strings
        2. Replace numeric placeholders (@) with random integers from ranges
        
        :returns: A list containing the problem text as NoEscape.
        """
        result = self.problem

        # replace numeric placeholders
        for i, prange in enumerate(self.ranges):
            result = result.replace('@', str(randint(prange[0], prange[1])), count=1)
        # replace string placeholders
        for i, repl in enumerate(self.string_options):
            count = i + 1
            result = self._replace_placeholders(result, '`', count, choice(repl))

        self.num_quest -= 1
        return [NoEscape(result)]
