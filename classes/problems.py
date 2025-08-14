from random import randint, choice, shuffle, random, sample, choices
from decimal import Decimal
from abc import ABC, abstractmethod

from typing import Callable
from pylatex import Math, Command, Tabular, MiniPage, Document, VerticalSpace, StandAloneGraphic, TikZ, Axis, Plot
from pylatex.base_classes import Environment, CommandBase
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


class BinaryOperation(ProblemBase, ABC):
    def __init__(self, num_quest: int, operand: str | Command, neg=False):
        """
        Initializes a binary operation problem.

        :param num_quest: The number of questions to be generated.
        :param operand: The operand to be used, such as + or \\times.
        :param neg: neg: If True, at least one of the operands will be negative.
        """
        super().__init__(num_quest, "0cm")
        self.operand = operand
        self.neg = neg

    @abstractmethod
    def generator(self) -> BaseMathEntity:
        """
        Randomly generates an operand.
        """
        pass   # child class should implement this

    def generate_random_operands(self) -> tuple[BaseMathEntity, BaseMathEntity]:
        """
        Randomly generates two operands for the problem.
        If neg is True, at least one of the operand will be negative.

        :return: The two generated operands in a tuple.
        """
        ops = [self.generator() for _ in range(2)]
        if self.neg:
            neg_i = randint(0, 1)
            other_i = 1 - neg_i
            ops[neg_i] = -ops[neg_i]
            if randint(0, 1) == 1:
                ops[other_i] = -ops[other_i]
        return ops[0], ops[1]

    def get_problem(self) -> list[Math]:
        o1, o2 = self.generate_random_operands()
        self.num_quest -= 1
        return [Math(inline=True, data=o1.get_latex() + [self.operand] + o2.get_latex() + ["="])]


class IntegerBinaryOperation(BinaryOperation):
    def __init__(self, num_quest: int, operand: str | Command, orange: tuple[int, int], neg=False):
        """
        Initializes an integer binary operation problem.

        :param num_quest: The number of questions to be generated.
        :param operand: The operand to be used, such as + or \\times.
        :param orange: The range for the operands, (begin, end) inclusive.
                       If the range includes a negative number, using neg=True may not produce
                       negative operands because a negative operand might be negated again.
        :param neg: If True, at least one of the operands will be negative.
        """
        super().__init__(num_quest, operand, neg)
        self.orange = orange

    def generator(self) -> Number:
        return Number(randint(self.orange[0], self.orange[1]))


class FractionBinaryOperation(BinaryOperation):
    def __init__(self, num_quest: int, operand: str | Command, nrange: tuple[int, int], drange: tuple[int, int],
                 no1=True, neg=False):
        """
        Initializes a fraction binary operation problem.

        :param num_quest: The number of questions to be generated.
        :param operand: The operand to be used, such as + or \\times.
        :param nrange: The range for numerator, (begin, end) inclusive.
        :param drange: The range for denominator, (begin, end) inclusive.
        :param no1: If True, the numerator and the denominator are always different.
        :param neg: If True, at least one of the operands will be negative.
        """
        super().__init__(num_quest,operand, neg)
        self.nrange = nrange
        self.drange = drange
        self.no1 = no1

    def generator(self) -> Fraction:
        return self.generate_random_fraction(self.nrange, self.drange, self.no1)

    @staticmethod
    def generate_random_fraction(nrange, drange, no1=True) -> Fraction:
        """
        Generates a fraction with random numerator and denominator.
        :param nrange: The range for numerator, (begin, end) inclusive.
        :param drange: The range for denominator, (begin, end) inclusive.
        :param no1: If True, the numerator and the denominator are always different.
        """
        n = randint(nrange[0], nrange[1])
        d = randint(drange[0], drange[1])
        if no1:
            while d == n:
                d = randint(drange[0], drange[1])
        return Fraction(n, d)


class FractionAddition(FractionBinaryOperation):
    def __init__(self, num_quest: int, nrange: tuple[int, int], drange: tuple[int, int],
                 no1=True, neg=False):
        """
        Initializes a fraction addition problem.
        :param num_quest: The number of questions to be generated.
        :param nrange: The range for numerator, (begin, end) inclusive.
        :param drange: The range for denominator, (begin, end) inclusive.
        :param no1: If True, the numerator and the denominator are always different.
        :param neg: If True, at least one of the operands will be negative.
        """
        super().__init__(num_quest, '+', nrange, drange, no1, neg)


class FractionSubtraction(FractionBinaryOperation):
    def __init__(self, num_quest: int, nrange: tuple[int, int], drange: tuple[int, int],
                 no1=True, neg=False):
        """
        Initializes a fraction subtraction problem.
        :param num_quest: The number of questions to be generated.
        :param nrange: The range for numerator, (begin, end) inclusive.
        :param drange: The range for denominator, (begin, end) inclusive.
        :param no1: If True, the numerator and the denominator are always different.
        :param neg: If True, at least one of the operands will be negative.
        """
        super().__init__(num_quest, '-', nrange, drange, no1, neg)


class FractionMultiplication(FractionBinaryOperation):
    def __init__(self, num_quest: int, nrange: tuple[int, int], drange: tuple[int, int],
                 no1=True, neg=False):
        """
        Initializes a fraction subtraction problem.
        :param num_quest: The number of questions to be generated.
        :param nrange: The range for numerator, (begin, end) inclusive.
        :param drange: The range for denominator, (begin, end) inclusive.
        :param no1: If True, the numerator and the denominator are always different.
        :param neg: If True, at least one of the operands will be negative.
        """
        super().__init__(num_quest, Command('times'), nrange, drange, no1, neg)


class FractionDivision(FractionBinaryOperation):
    def __init__(self, num_quest: int, nrange: tuple[int, int], drange: tuple[int, int],
                 no1=True, neg=False):
        """
        Initializes a fraction subtraction problem.
        :param num_quest: The number of questions to be generated.
        :param nrange: The range for numerator, (begin, end) inclusive.
        :param drange: The range for denominator, (begin, end) inclusive.
        :param no1: If True, the numerator and the denominator are always different.
        :param neg: If True, at least one of the operands will be negative.
        """
        super().__init__(num_quest, Command('div'), nrange, drange, no1, neg)


class WordProblem(ProblemBase):
    def __init__(self, num_quest: int, vspace: str, name: str, ranges: list[tuple[int, int]]):
        """
        Initializes a word problem. Make sure word_problems.txt is in the same directory as where this was called.
        :param num_quest: The number of questions to be generated.
        :param vspace: The minimum vertical space to be added below each problem.
        :param name: The name of the problem to be used to choose problems.
        :param ranges: The ranges of values to be used for problem parameters.
                       They should be in the order the placeholders appear in the problem.
        """
        super().__init__(num_quest, vspace)
        self.ranges = ranges
        with open("word_problems.txt", 'r') as f:
            line = f.readline()
            found = False
            while line and not found:
                if line[0] == '!':
                    # problem start detected
                    line = line[1:].rstrip()   # remove the problem marker
                    if line == name:
                        # problem found
                        found = True
                        self.problem = f.readline().rstrip()
                        if int(f.readline().rstrip()) != len(ranges):
                            raise ValueError("The number of ranges doesn't agree with the number of problem parameters")
                line = f.readline()
            if not found:
                raise ValueError("{} not found in the file".format(name))

    def get_problem(self) -> list[NoEscape]:
        result = self.problem

        for prange in self.ranges:
            result = result.replace('@', str(randint(prange[0], prange[1])), 1)

        self.num_quest -= 1
        return [NoEscape(result)]


class GraphingProblem(ProblemBase, ABC):
    def __init__(self, num_quest: int):
        """
        Initializes a graphing problem.
        It requires graphing_grid.png in the document_output folder.

        :param num_quest: The number of questions to be generated.
        """
        super().__init__(num_quest, '0cm')

    @abstractmethod
    def get_random_function(self) -> NoEscape:
        """
        Returns a function with randomized parameters.
        """
        pass   # child class should implement this

    def get_problem(self) -> list[Math | NoEscape | CommandBase]:
        result = [
            Math(data=["f(x)=", self.get_random_function()], inline=True),
            NoEscape("\n"),
            VerticalSpace("2em"),
            NoEscape("\n"),
            StandAloneGraphic("graphing_grid.png", "width=0.3\\linewidth")
        ]

        self.num_quest -= 1
        return result


class LinearGraphingProblem(GraphingProblem):
    def __init__(self, num_quest: int, num_range: tuple[int, int], *types: str):
        """
        Initializes a graphing problem for a linear function.
        It requires graphing_grid.png in the document_output folder.

        The possible types are the following (there's a 50% chance the coefficient a will be a fraction):
        - si: ax + b, a != 0
        - sp: a(x - h) + k, a, h != 0

        :param num_quest: The number of questions to be generated.
        :param num_range: The range for the parameters, (begin, end) inclusive.
                          This must contain at least two integers where at least one is non-zero.
        :param types: The possible types of the problems. Refer to docstring for the list of all options.
        """
        super().__init__(num_quest)
        if num_range[1] - num_range[0] < 2:
            raise ValueError("Fewer than two integers found in the number range")

        possible_types = ("si", "sp")
        for t in types:
            if t not in possible_types:
                raise ValueError(f"Problem type {t} is not a valid type")
        if not types:
            types = possible_types

        self.num_range = num_range
        self.types = types

    def get_random_function(self):
        prob = choice(self.types)
        result = NoEscape()

        if random() < 0.5:
            # a is integer
            a = randint(self.num_range[0], self.num_range[1])
            while a == 0:
                a = randint(self.num_range[0], self.num_range[1])

            # wrap in BaseMathClass for the dumps() call later
            if a == -1:
                a = TextWrapper(["-"])
            elif a == 1:
                a = TextWrapper()
            else:
                a = Number(a)
        else:
            # a is fraction
            a = Fraction(1, 1)
            while a.denom == 1:
                num = randint(self.num_range[0], self.num_range[1])
                while num == 0:
                    num = randint(self.num_range[0], self.num_range[1])
                denom = randint(self.num_range[0], self.num_range[1])
                while denom == 0:
                    denom = randint(self.num_range[0], self.num_range[1])
                a = Fraction(abs(num), abs(denom), (num * denom) // abs(num * denom), big=False).simplified()

        match prob:
            case "si":
                b = randint(self.num_range[0], self.num_range[1])
                poly = UnsafePolynomial(a.dumps() + "x", Number(b))
                result = poly.dumps()
            case "sp":
                h = randint(self.num_range[0], self.num_range[1])
                while h == 0:
                    h = randint(self.num_range[0], self.num_range[1])
                k = randint(self.num_range[0], self.num_range[1])
                mini_poly = SingleVariablePolynomial("x", [
                    {"coefficient": 1, "exponent": 1},
                    {"coefficient": -h, "exponent": 0}
                ], wrap=True)
                poly = UnsafePolynomial(a.dumps() + mini_poly.dumps(), Number(k))
                result = poly.dumps()

        return result


class SquareRootProblem(ProblemBase):
    def __init__(self, num_quest: int, operand_range: tuple[int, int], frac=False, no_duplicate=True):
        """
        Initializes a square root problem, which involves evaluating the square root of a perfect square.
        The operand can either be a non-negative integer or a fraction of non-negative perfect square integers.

        :param num_quest: The number of questions to be generated.
        :param operand_range: The range of the square root of the operand, (begin, end) inclusive.
                              For example, (1, 3) allows the operands 1, 4, and 9.
                              If frac is True, this determines the range used for the numerator and
                              the denominator.
        :param frac: If False, the operand will be a non-negative integer.
                     If True, the operand will be a fraction of non-negative integer.
                     When True, operand_range cannot include 0.
        :param no_duplicate: If True, no duplicate operand will be generated.
                             There should be enough numbers in operand_range to generate all the necessary questions.
                             Be careful using this with frac since a number appearing in the numerator stops
                             it from appearing in the denominator too, so the numbers will run out
                             twice as fast as with integer operands.
        """
        super().__init__(num_quest, "0cm")

        # reject non-integer range
        if operand_range[0] % 1 != 0 or operand_range[0] % 1 != 0:
            raise ValueError("Operand range must be integers")
        # reject negative range
        if operand_range[0] < 0 or operand_range[1] < 0:
            raise ValueError("Operand range must be non-negative")
        # reject zero with fraction
        if frac and operand_range[0] == 0:
            raise ValueError("Operand range cannot include 0 for fractions")

        self.operand_candidates = [n**2 for n in range(operand_range[0], operand_range[1] + 1)]
        self.frac = frac
        self.no_duplicate = no_duplicate
        # make sure there are enough numbers
        if no_duplicate:
            if (not frac and len(self.operand_candidates) < num_quest) or (frac and len(self.operand_candidates) < 2 * num_quest):
                raise ValueError(f"Operand range is not big enough to generate {num_quest} questions")

    def get_problem(self) -> list[Math]:
        if not self.frac:
            chosen = choice(self.operand_candidates)
            if self.no_duplicate:
                self.operand_candidates.remove(chosen)
            result = Math(inline=True, data=[Command('sqrt', chosen)])
        else:
            num = choice(self.operand_candidates)
            if self.no_duplicate:
                self.operand_candidates.remove(num)
            denom = choice(self.operand_candidates)
            if self.no_duplicate:
                self.operand_candidates.remove(denom)
            result = Math(inline=True, data=[Command('sqrt', Fraction(num, denom).get_latex())])

        self.num_quest -= 1
        return [result]


class SquareRootDecimalProblem(SquareRootProblem):
    def __init__(self, num_quest: int, base_range: tuple[int, int], offset_range: tuple[int, int], no_duplicate=True):
        """
        Initializes a square root problem with a decimal operand.

        The operand is generated based on two parameters: the number base (b) and the offset exponent (o).
        The generation uses the following formula:

        b**2 / 100**o

        Here are some examples to illustrate how this works:
        * b = 3 and o = 1 -> 0.09
        * b = 11 and o = 1 -> 1.21
        * b = 5 and o = 2 -> 0.0025

        :param num_quest: The number of questions to be generated.
        :param base_range: The range for the number base, (begin, end) inclusive.
                           The bounds must be positive (non-zero) integers.
        :param offset_range: The range for the offset exponent, (begin, end) inclusive.
                             The bounds must be positive (non-zero) integers.
        :param no_duplicate: If True, no duplicate operand will be generated.
                             There should be enough numbers in operand_range to generate all the necessary questions.
        """
        # set operand_range so that nothing is generated
        # set no_duplicate to False temporarily to avoid num_quest check
        super().__init__(num_quest, (1, 0), False, False)
        self.no_duplicate = no_duplicate
        if base_range[0] % 1 != 0 or base_range[1] % 1 != 0 or offset_range[0] % 1 != 0 or offset_range[1] % 1 != 0:
            raise ValueError("The ranges must be integers")
        if base_range[0] <= 0 or offset_range[0] <= 0:
            raise ValueError("The ranges must be positive")

        # generate all candidates
        self.operand_candidates = []
        for b in range(base_range[0], base_range[1] + 1):
            for o in range(offset_range[0], offset_range[1] + 1):
                self.operand_candidates.append(str(Decimal(b**2) / Decimal(100**o)))

        if no_duplicate and len(self.operand_candidates) < num_quest:
            raise ValueError("The given ranges do not generate enough questions")


class LinearRelationProblem(ProblemBase):
    def __init__(self, num_quest: int,
                 h_range: tuple[int | str, int | str] | tuple[int | str, int | str, int | str],
                 k_range: tuple[int | str, int | str] | tuple[int | str, int | str, int | str],
                 x_range: tuple[int | str, int | str] | tuple[int | str, int | str, int | str]=(1, 5, 1),
                 no_constant=True):
        """
        Initializes a problem that gives a table of x and y values
        and asks the student to sketch the graph of the relation
        and find the equation of the relation.

        The table is generated by specifying the linear function.
        It is defined by the parameters in the standard form:

        y = hx + k

        It requires graphing_grid.png in the document_output folder.

        If you want decimal numbers for any of the ranges, the number must be given in a string, not a float.

        :param num_quest: The number of questions to be generated.
        :param h_range: The range for the slope, (begin, end, step) inclusive.
                        Step can be omitted, in which case it's assumed to be 1.
        :param k_range: The range for the y-intercept, (begin, end, step) inclusive.
                        Step can be omitted, in which case it's assumed to be 1.
        :param x_range: The range for the x-values in the table, (begin, end) inclusive.
                        Step can be omitted, in which case it's assumed to be 1.
                        The default is (1, 5, 1).
        :param no_constant: If True, the slope is never 0.
        """
        super().__init__(num_quest, "0cm")
        # add default step if necessary
        if len(h_range) == 2:
            h_range = (h_range[0], h_range[1], 1)
        if len(k_range) == 2:
            k_range = (k_range[0], k_range[1], 1)
        if len(x_range) == 2:
            x_range = (x_range[0], x_range[1], 1)

        # convert numbers to Decimal
        h_range = list(h_range)
        k_range = list(k_range)
        x_range = list(x_range)
        for t in [h_range, k_range, x_range]:
            for i in range(len(t)):
                t[i] = Decimal(t[i])

        self.h_range = tuple(h_range)
        self.k_range = tuple(k_range)
        self.no_constant = no_constant
        if no_constant and h_range[0] == 0 and h_range[1] - h_range[0] < h_range[2]:
            raise ValueError("The given h_range always produces a zero slope")

        # generate x-values
        self.x_values = []
        x = x_range[0]
        while x <= x_range[1]:
            self.x_values.append(x)
            x += x_range[2]

    def get_problem(self) -> list[DocInjector]:
        def decimal_randrange(start, end, step) -> Decimal:
            count = int((end - start) // step)   # 1 less than the total number of candidates
            return start + randint(0, count) * step

        # generate the parameters
        h = decimal_randrange(self.h_range[0], self.h_range[1], self.h_range[2])
        while self.no_constant and h == 0:
            h = decimal_randrange(self.h_range[0], self.h_range[1], self.h_range[2])
        k = decimal_randrange(self.k_range[0], self.k_range[1], self.k_range[2])

        def body(doc: Document):
            # put the figures into minipages for alignment
            with doc.create(MiniPage(width='3cm')):
                # assemble the table of values
                with doc.create(Tabular("c|c")) as xy_table:
                    xy_table.add_row(Math(inline=True, data=['x']), Math(inline=True, data=['y']))
                    xy_table.add_hline()
                    for x in self.x_values:
                        xy_table.add_row(Math(inline=True, data=[str(x)]), Math(inline=True, data=[str(h * x + k)]))
            with doc.create(MiniPage(width=r'0.3\textwidth')):
                doc.append(StandAloneGraphic("graphing_grid.png", "width=\\linewidth"))

        self.num_quest -= 1
        return [DocInjector(body)]


class EquationSingleOperation(ProblemBase):
    def __init__(self, num_quest: int, nrange: tuple[int, int], *operations: str):
        r"""
        Initializes an equation solving problem that requires one arithmetic operation to solve.
        The operation can be addition, subtraction, multiplication, and division.
        Note that division is always represented as a fraction.
        For example, if operations=('add', 'mul'), then the following equations (and others) can be generated:

        x + 5 = 13
        5x = 3

        However, the following equations cannot be generated:

        x - (-5) = 13
        \frac{x}{0.2} = 3

        The variable may be on the left or the right side of the equation,
        which is randomly determined by an equal chance.

        :param num_quest: The number of questions to be generated.
        :param nrange: The range used for the numbers in the equation, (begin, end) inclusive.
        :param operations: The operations that are allowed to be used.
                           The options are add, sub, mul, div.
                           By default, anything is allowed.
        """
        if any(o not in ['add', 'sub', 'mul', 'div'] for o in operations):
            raise ValueError(f"One of {operations} is an invalid operation")
        zero_only = nrange[1] - nrange[0] <= 0 and nrange[0] == 0
        if 'div' in operations and zero_only:
            raise ValueError("The given range can only generate 0 yet div is included")

        super().__init__(num_quest, '3cm')
        self.nrange = nrange
        self.operations = operations if len(operations) > 0 else (('add', 'sub', 'mul', 'div') if not zero_only else ('add', 'sub', 'mul'))

    def get_problem(self) -> list[Math]:
        operation = choice(self.operations)
        constant_side = [randint(self.nrange[0], self.nrange[1])]
        operand = Number(randint(self.nrange[0], self.nrange[1]))
        while operation == 'div' and operand == 0:
            operand = Number(randint(self.nrange[0], self.nrange[1]))
        variable_side = None
        match operation:
            case 'add':
                variable_side = ['x+'] + operand.get_latex()
            case 'sub':
                variable_side = ['x-'] + operand.get_latex()
            case 'mul':
                variable_side = operand.get_latex() + ['x']
            case 'div':
                operand_str = NoEscape(operand.dumps())
                variable_side = [Command('frac', ['x', operand_str])]

        self.num_quest -= 1
        if randint(0, 1) == 0:
            return [Math(inline=True, data=variable_side + ['='] + constant_side)]
        else:
            return [Math(inline=True, data=constant_side + ['='] + variable_side)]


class PolynomialSimplify(ProblemBase):
    def __init__(self,
                 num_quest: int,
                 crange: tuple[int, int],
                 drange: tuple[int, int],
                 *var: str,
                 max_like: int=2,
                 like_chance: float=0.75):
        """
        Initializes a problem where the student must simplify a polynomial.
        The polynomial only contains one variable.
        The produced polynomial is guaranteed to have at least 2 like terms unless max_like=1.

        :param num_quest: The number of questions to be generated.
        :param crange: The range used for the coefficients, (begin, end) inclusive.
                       The generated coefficient will never be 0.
        :param drange: The range used for the degree of the polynomial, (begin, end) inclusive.
                       This range cannot contain a negative number.
        :param var: The possible variables to be used. Only one of them will be used per question.
                    The default is just x.
        :param max_like: The maximum number of like terms in the polynomial for the same exponent.
                         Note that 1 like term means there will be no other like terms.
                         That is, if max_like=1, the polynomial 4x + 3 can be generated but not
                         4x + 3x. The latter requires at least max_like=2.
                         It must be at least 1. The default is 2.
        :param like_chance: The probability of having more than 1 like term.
                            This probability is applied for each exponent.
                            The probabilities of getting 0 and 1 like term are equal.
                            The probabilities of getting 2, 3, 4, etc. like terms are equal.
                            Has no effect if max_like=1.
        """
        super().__init__(num_quest, '4cm')
        if crange[0] > crange[1]:
            raise ValueError("The coefficient range contains no number")
        if crange[1] - crange[0] == 0 and crange[0] == 0:
            raise ValueError("The coefficient range only contains 0")
        if drange[0] > drange[1]:
            raise ValueError("The degree range contains no number")
        if drange[0] < 0:
            raise ValueError("The degree range includes negative numbers")
        if max_like < 1:
            raise ValueError("The max number of like terms must be at least 1")
        if like_chance < 0 or like_chance > 1:
            raise ValueError("Like chance must be 0 or 1 or anything in between")

        self.crange = crange
        self.drange = drange
        self.max_like = max_like
        self.like_chance = like_chance
        if not var:
            var = ('x',)
        self.var = var

    def random_coefficient(self) -> int:
        """
        Returns a random nonzero coefficient.
        """
        c = randint(self.crange[0], self.crange[1])
        while c == 0:
            c = randint(self.crange[0], self.crange[1])
        return c

    def generate_term(self, exp: int) -> dict[str, int]:
        """
        Generates a nonzero term with a random coefficient.

        :param exp: The exponent of the term.
        :return: The generated polynomial in a dictionary.
        """
        return {'coefficient': self.random_coefficient(), 'exponent': exp}

    def generate_polynomial(self, degree: int=None, num_terms: int=None, var: str=None) -> SingleVariablePolynomial:
        """
        Randomly generates a polynomial, following the rules outlined in __init()__.

        :param degree: The degree of the polynomial. If None (default), it is randomly generated using drange.
        :param num_terms: If set, the generated polynomial will have exactly this many terms.
                          To use this, max_like must be 1.
                          If set to None (default), there will be no set number.
        :param var: Use this variable for the polynomial. Must be a member of self.var.
                    If set to None (default), a random one will be drawn from self.var.
        """
        if degree is None:
            degree = randint(self.drange[0], self.drange[1])
        if degree < 0:
            raise ValueError("The degree must be non-negative")
        if num_terms is not None and num_terms < 1:
            raise ValueError("The number of terms must be positive")
        if num_terms and self.max_like > 1:
            raise ValueError("num_terms argument cannot be used unless max_like is 1")
        if var is None:
            var = choice(self.var)
        elif var not in self.var:
            raise ValueError(f"The chosen variable {var} is not in self.var == {self.var}")

        # determine the number of like terms for each possible exponent
        term_count = [0 for _ in range(degree + 1)]  # index is exponent
        to_do = list(range(degree + 1))
        if self.max_like == 1:
            if num_terms is None:
                for i in to_do:
                    term_count[i] = randint(0, 1)
            else:
                term_count[degree] = 1   # max degree term is enforced here so that nothing is added later
                if len(to_do) > 1:
                    to_do = sample(to_do[:-1], k=num_terms-1)
                    for i in to_do:
                        term_count[i] = 1
        else:
            # choose an exponent and make sure it has at least 2 like terms
            i = choice(to_do)
            to_do.pop(i)  # pop is fine since to_do is the same as the indices at this point
            term_count[i] = randint(2, self.max_like)
            # the rest is randomly selected
            for i in to_do:
                if random() < self.like_chance:
                    # two or more like terms
                    term_count[i] = randint(2, self.max_like)
                else:
                    # one or no like term
                    term_count[i] = randint(0, 1)
        if term_count[degree] == 0:
            # max exponent term must be present to have the right degree
            term_count[degree] = 1

        # generate the polynomial
        poly = []
        for i in range(len(term_count)):
            while term_count[i] > 0:
                poly.append(self.generate_term(i))
                term_count[i] -= 1

        # return result as polynomial
        shuffle(poly)
        return SingleVariablePolynomial(choice(var), poly)

    def get_problem(self) -> list[Math]:
        self.num_quest -= 1
        return [Math(inline=True, data=self.generate_polynomial().get_latex())]


class PolynomialAdd(PolynomialSimplify):
    def __init__(self, num_quest: int, crange: tuple[int, int], drange: tuple[int, int], *var: str, min_term_count: int = 1):
        """
        Generates a problem where two polynomials are added.
        The polynomials contain only one variable.

        :param num_quest: The number of questions to be generated.
        :param crange: The range used for the coefficients, (begin, end) inclusive.
                       The generated coefficient will never be 0.
        :param drange: The range used for the degree of the polynomial, (begin, end) inclusive.
                       This range cannot contain a negative number.
        :param min_term_count: The minimum number of terms to be used for each polynomial.
                               1 by default.
        :param var: The possible variables to be used. Only one of them will be used per question.
                    The default is just x.
        """
        super().__init__(num_quest, crange, drange, *var, max_like=1)
        if drange[1] + 1 < min_term_count:
            raise ValueError(f"drange not big enough to generate {min_term_count} terms")
        self.min_term_count = min_term_count

    def get_problem(self) -> list[Math]:
        polies = []
        var = choice(self.var)
        for _ in range(2):
            degree = randint(max(self.drange[0], self.min_term_count - 1), self.drange[1])
            term_count = randint(self.min_term_count, degree + 1)
            polies.append(self.generate_polynomial(degree, term_count, var))
        self.num_quest -= 1
        return [Math(inline=True,
                     data=polies[0].get_latex() + ['+', Command('left(')] + polies[1].get_latex() + [Command('right)')])]


class PolynomialSubtract(PolynomialSimplify):
    def __init__(self, num_quest: int, crange: tuple[int, int], drange: tuple[int, int], *var: str, min_term_count: int=1):
        """
        Generates a problem where two polynomials are subtracted.
        The polynomials contain only one variable.

        :param num_quest: The number of questions to be generated.
        :param crange: The range used for the coefficients, (begin, end) inclusive.
                       The generated coefficient will never be 0.
        :param drange: The range used for the degree of the polynomial, (begin, end) inclusive.
                       This range cannot contain a negative number.
        :param min_term_count: The minimum number of terms to be used for each polynomial.
                               1 by default.
        :param var: The possible variables to be used. Only one of them will be used per question.
                    The default is just x.
        """
        super().__init__(num_quest, crange, drange, *var, max_like=1)
        if drange[1] + 1 < min_term_count:
            raise ValueError(f"drange not big enough to generate {min_term_count} terms")
        self.min_term_count = min_term_count

    def get_problem(self) -> list[Math]:
        polies = []
        var = choice(self.var)
        for _ in range(2):
            degree = randint(max(self.drange[0], self.min_term_count - 1), self.drange[1])
            term_count = randint(self.min_term_count, degree + 1)
            polies.append(self.generate_polynomial(degree, term_count, var))
        self.num_quest -= 1
        return [Math(inline=True, data=polies[0].get_latex() + ['-', Command('left(')] + polies[1].get_latex() + [Command('right)')])]


class PolynomialMultiply(PolynomialSimplify):
    def __init__(self, num_quest: int, crange: tuple[int, int], drange: tuple[int, int], *var: str, min_term_count: int=1, max_term_count: int=2):
        """
        Generates a problem where two polynomials are multiplied.
        The polynomials contain only one variable.

        :param num_quest: The number of questions to be generated.
        :param crange: The range used for the coefficients, (begin, end) inclusive.
                       The generated coefficient will never be 0.
        :param drange: The range used for the degree of the polynomial, (begin, end) inclusive.
                       This range cannot contain a negative number.
                       The lower bound is capped by max_term_count-1 for the left multiplicand.
                       That is, if max_term_count=3, then the degree of the left multiplicand will be 2 or more,
                       even if the lower bound in drange is less than 2.
        :param var: The possible variables to be used. Only one of them will be used per question.
                    The default is just x.
        :param min_term_count: Minimum number of terms allowed for the left multiplicand. Must be at least 1.
                               This cannot be above drange[0] + 1. The default is 1 (monomial).
        :param max_term_count: Maximum number of terms allowed for the left multiplicand. The default is 2 (binomial).
                               This cannot be less than min_term_count.
        """
        if min_term_count > max_term_count:
            raise ValueError(f"min_term_count ({min_term_count}) cannot be greater than max_term_count({max_term_count})")
        if min_term_count < 1:
            raise ValueError(f"min_term_count must be at least 1")
        if min_term_count > drange[0] + 1:
            raise ValueError(f"the drange {drange} is not valid for the min_term_count {min_term_count}")
        super().__init__(num_quest, crange, drange, *var, max_like=1)
        self.min_term_count = min_term_count
        self.max_term_count = max_term_count

    def get_problem(self) -> list[Math]:
        var = choice(self.var)
        left_term_count = randint(self.min_term_count, min(self.max_term_count, self.drange[1] + 1))
        left = self.generate_polynomial(randint(max(self.drange[0], left_term_count-1), self.drange[1]), left_term_count, var)
        right = self.generate_polynomial(var=var)
        self.num_quest -= 1
        return [Math(inline=True, data=[Command('left(')] + left.get_latex() + [Command('right)'), Command('left(')] + right.get_latex() + [Command('right)')])]


class PolynomialDivide(PolynomialSimplify):
    def __init__(self, num_quest: int, crange: tuple[int, int], drange: tuple[int, int], *var: str, no_constant=False):
        """
        Generates a problem where a polynomial is divided by a monomial.
        The problem is always written as a fraction.
        The quotient always simplify to a polynomial with integer coefficients.

        :param num_quest: The number of questions to be generated.
        :param crange: The range used for the coefficients, (begin, end) inclusive.
                       The generated coefficient will never be 0.
        :param drange: The range used to generate the degree of the dividend and the divisor, (begin, end) inclusive.
                       This range cannot contain a negative number.
                       The numerator degree might exceed the range, but not the denominator.
        :param var: The possible variables to be used. Only one of them will be used per question.
                    The default is just x.
        :param no_constant: If True, the degree of the divisor will be at least 1.
                            False by default.
        """
        super().__init__(num_quest, crange, drange, *var, max_like=1)
        if no_constant and drange[1] < 1:
            raise ValueError("no_constant used yet drange doesn't contain 1 or higher")
        self.no_constant = no_constant

    def get_problem(self) -> list[Math]:
        var = choice(self.var)
        if self.no_constant:
            divisor = self.generate_polynomial(randint(max(1, self.drange[0]), self.drange[1]), 1, var)
        else:
            divisor = self.generate_polynomial(num_terms=1, var=var)
        quotient = self.generate_polynomial(var=var)
        dividend = divisor * quotient

        self.num_quest -= 1
        return [Math(inline=True, data=PolynomialFraction(dividend, divisor).get_latex())]


class EquationMultiOperation(ProblemBase):
    def __init__(self, num_quest: int, nrange: tuple[int, int], *types: str, var=('x',), inequality=False):
        r"""
        Initializes a problem where an equation must be solved.
        Every equation requires at least two operations to solve.
        The randomly generated numbers will never be 0.
        A denominator or an explicit coefficient (such as 'a' in ax + b) will never be 1.
        A fraction will always have distinct numerator and denominator.

        Possible equation types (the variable is always x, the rest are random. The order of terms may be randomized):

        simple: ax + b = c
        simple_div: \frac{x}{a} + b = c
        simple_dist: a(x + b) = c
        double: ax + b = cx + d
        double_dist: a(bx + c) = d(ex + f)
        double_frac: \frac{x}{a} + \frac{b}{c} = \frac{d}{e}
        double_frac_dist: \frac{a}{b}(x + c) = \frac{d}{e}(x + f)
        rational: \frac{a}{x} = b, cannot be used with inequality
        frac_const: \frac{x}{a} + b = \frac{c}{d}
        bino_frac: \frac{x + a}{b} + \frac{c}{d} = \frac{e}{f}
        double_bino_frac: \frac{x + a}{b} + \frac{x + c}{d} = \frac{e}{f}
        bino_frac_const: \frac{x + a}{b} + c = d
        double_bino_frac_large: \frac{x + a}{b} + cx + d = \frac{x + e}{f} + g
        insane_1: \frac{abx^(k+1) + acx^k}{ax^k} = d(ex + f + gx), where k in [1, 5]
        insane_2: \sqrt{(a^2/100)x(b^2x)} = \sqrt{\frac{c^2}{d^2}} + ex, x >= 0
        insane_3: \sqrt{ax(bx) + nx^2} + cx = \sqrt{(x + d)^2}, n is the smallest number that makes the coefficient a perfect square
        insane_4: (ax + b)^2 = (ox + c)^2 + d, where o == a or o == -a, x >= 0
        insane_5: ax + \sqrt{(b^2/100)x^2} = \sqrt{\frac{c^2}{d^2}}(ex + f), x >= 0

        :param num_quest: The number of questions to be generated.
        :param nrange: The range used for the numbers in the equation, (begin, end) inclusive.
        :param types: The types of equations to be used.
                      Refer to the docstring for the options and the description of each type.
                      If nothing is given, every type can appear.
        :param var: The potential variables to be used. The default is just x.
        :param inequality: If True, the problems will be inequality problems instead.
                           One of the four inequalities will be randomly chosen for each question.
        """
        super().__init__(num_quest, '6cm')
        possible_types = ('simple',
                          'simple_div',
                          'simple_dist',
                          'double',
                          'double_dist',
                          'double_frac',
                          'double_frac_dist',
                          'rational',
                          'frac_const',
                          'bino_frac',
                          'double_bino_frac',
                          'bino_frac_const',
                          'double_bino_frac_large',
                          'insane_1',
                          'insane_2',
                          'insane_3',
                          'insane_4',
                          'insane_5')

        if nrange[0] > nrange[1]:
            raise ValueError("The given range does not contain any integer")
        if nrange[0] == nrange[1]:
            if nrange[0] == 0:
                raise ValueError("The given range only contains 0")
            elif nrange[0] == 1:
                raise ValueError("The given range only contains 1")
        if nrange == (0, 1):
            raise ValueError("The given range only contains 0 and 1")
        for t in types:
            if t not in possible_types:
                raise ValueError(f"The problem type {t} is invalid")
        if inequality and len(types) == 1 and types[0] == 'rational':
            raise ValueError("Inequality problems require something other than rational type")

        if not types:
            types = possible_types
        if inequality and 'rational' in types:
            tmp = list(types)
            tmp.remove('rational')
            types = tuple(tmp)
        self.types = types
        self.nrange = nrange
        self.var = var
        self.inequality = inequality

    def draws(self, size: int, *blacklist: int) -> tuple[int, ...]:
        """
        Generates random numbers based on the nrange.
        Returns an empty tuple if size is 0.
        0 is never generated.

        :param size: The number of random numbers to generate.
        :param blacklist: These integers will not be generated.
        """
        if size == 0:
            return tuple()
        candidates = [n for n in range(self.nrange[0], self.nrange[1] + 1) if n != 0 and n not in blacklist]
        if not candidates:
            raise ValueError("no number can be drawn")
        return tuple(choices(candidates, k=size))

    def draw(self, *blacklist: int) -> int:
        """
        Generates a random number based on the nrange.
        0 is never generated.

        :param blacklist: These integers will not be generated.
        """
        return self.draws(1, *blacklist)[0]

    def get_problem(self) -> list[Math]:
        if self.inequality:
            middle = choice(['>', '<', Command('geq'), Command('leq')])
        else:
            middle = '='
        var = choice(self.var)
        prob_type = choice(self.types)
        num_params = 0

        match prob_type:
            case 'rational':
                num_params = 2
            case 'simple' | 'simple_div' | 'simple_dist':
                num_params = 3
            case 'double' | 'frac_const' | 'bino_frac_const' | 'insane_3' | 'insane_4':
                num_params = 4
            case 'double_frac' | 'insane_2':
                num_params = 5
            case 'double_dist' | 'double_frac_dist' | 'bino_frac' | 'double_bino_frac' | 'insane_5':
                num_params = 6
            case 'double_bino_frac_large' | 'insane_1':
                num_params = 7

        params = [0 for _ in range(num_params)]
        lhs = None
        rhs = None
        disclaimer = [NoEscape(rf", {var} \geq 0")] if prob_type in ["insane_2", "insane_3", "insane_5"] else []

        match prob_type:
            case 'simple':
                params[0] = self.draw(1, -1)   # params[0] shouldn't be 1 or -1
                params[1], params[2] = self.draws(2)

                lhs = SingleVariablePolynomial(var,
                                               [{'coefficient': params[0], 'exponent': 1},
                                                      {'coefficient': params[1], 'exponent': 0}], mix=True)
                rhs = TextWrapper([str(params[2])])
            case 'simple_div':
                params[0] = self.draw(1)   # params[0] shouldn't be 1
                params[1], params[2] = self.draws(2)

                lhs = UnsafePolynomial(Command('frac', [var, params[0]]).dumps(), str(params[1]), mix=True)
                rhs = TextWrapper([str(params[2])])
            case 'simple_dist':
                params[0] = self.draw(1)   # params[0] shouldn't be 1
                params[1], params[2] = self.draws(2)

                lhs = TextWrapper([str(params[0]) if params[0] != -1 else '-',
                                   Command('left(').dumps(),
                                   SingleVariablePolynomial(var,
                                                            [{'coefficient': 1, 'exponent': 1},
                                                             {'coefficient': params[1], 'exponent': 0}],
                                                            mix=True).dumps(),
                                   Command('right)').dumps()])
                rhs = TextWrapper([str(params[2])])
            case 'double':
                # ax + b = cx + d has a solution if a != c
                params[1], params[3] = self.draws(2)   # these don't matter
                params[0] = self.draw(1)   # a shouldn't be 1
                params[2] = self.draw(1, params[0])   # c shouldn't be 1 or a

                lhs = SingleVariablePolynomial(var,
                                               [{'coefficient': params[0], 'exponent': 1},
                                                {'coefficient': params[1], 'exponent': 0}], True)
                rhs = SingleVariablePolynomial(var,
                                               [{'coefficient': params[2], 'exponent': 1},
                                                {'coefficient': params[3], 'exponent': 0}], True)
            case 'double_dist':
                # generate parameters, making sure it has a solution
                # a(bx + c) = d(ex + f) has a solution if ab != de
                params[2], params[5] = self.draws(2)   # c, f don't matter
                params[0], params[1] = self.draws(2, 1)   # a, b shouldn't be 1
                for _ in range(100):
                    d, e = self.draws(2, 1)   # d, e shouldn't be 1
                    if d * e != params[0] * params[1]:
                        params[3] = d
                        params[4] = e
                        break
                if params[3] == 0:
                    raise ValueError("The given nrange cannot seem to generate a double_dist with a solution")

                lhs = TextWrapper([str(params[0]) if params[0] != -1 else '-',
                                   Command('left(').dumps(),
                                   SingleVariablePolynomial(var,
                                                            [{'coefficient': params[1], 'exponent': 1},
                                                             {'coefficient': params[2], 'exponent': 0}],
                                                            True).dumps(),
                                   Command('right)').dumps()])
                rhs = TextWrapper([str(params[2]) if params[2] != -1 else '-',
                                   Command('left(').dumps(),
                                   SingleVariablePolynomial(var,
                                                            [{'coefficient': params[3], 'exponent': 1},
                                                             {'coefficient': params[4], 'exponent': 0}],
                                                            True).dumps(),
                                   Command('right)').dumps()])
            case 'double_frac':
                params[0], params[2], params[4] = self.draws(3, 1, -1)   # these shouldn't be 1 or -1
                params[1] = self.draw(params[2], -params[2])   # b shouldn't be +- c
                params[3] = self.draw(params[4], -params[4])   # d shouldn't be +-e

                lhs = UnsafePolynomial(Command('frac', [var, params[0]]).dumps(),
                                       Command('frac', [params[1], params[2]]).dumps(), mix=True)
                rhs = Fraction(params[3], params[4], big=False)
            case 'double_frac_dist':
                # \frac{a}{b}(x + c) = \frac{d}{e}(x + f) has a solution if a/b != d/e
                params[2], params[5] = self.draws(2)
                params[1] = self.draw(1, -1)   # b shouldn't be +-1
                params[0] = self.draw(params[1], -params[1])   # a shouldn't be +-b
                for _ in range(100):
                    d = self.draw()
                    e = self.draw(1, -1, d, -d)   # e shouldn't be +-1 or +-d
                    if d/e != params[0]/params[1]:
                        params[3] = d
                        params[4] = e
                        break
                if params[3] == 0:
                    raise ValueError("The given nrange cannot seem to generate a double_frac_dist with a solution")

                lhs = TextWrapper([Fraction(params[0], params[1], big=False).dumps(),
                                   Command('left(').dumps(),
                                   SingleVariablePolynomial(var,
                                                            [{'coefficient': 1, 'exponent': 1},
                                                             {'coefficient': params[2], 'exponent': 0}],
                                                            True).dumps(),
                                   Command('right)').dumps()])
                rhs = TextWrapper([Fraction(params[3], params[4], big=False).dumps(),
                                   Command('left(').dumps(),
                                   SingleVariablePolynomial(var,
                                                            [{'coefficient': 1, 'exponent': 1},
                                                             {'coefficient': params[5], 'exponent': 0}],
                                                            True).dumps(),
                                   Command('right)').dumps()])
            case 'rational':
                params[0], params[1] = self.draws(2)

                lhs = TextWrapper([Command('frac', [params[0], var]).dumps()])
                rhs = TextWrapper([str(params[1])])
            case 'frac_const':
                params[1] = self.draw()
                params[0], params[3] = self.draws(2, 1, -1)   # a and d shouldn't be +-1
                params[2] = self.draw(params[3], -params[3])   # c shouldn't be +-d

                lhs = UnsafePolynomial(Command('frac', [var, params[0]]).dumps(),
                                       str(params[1]), mix=True)
                rhs = Fraction(params[0], params[1], big=False)
            case 'bino_frac':
                params[0] = self.draw()
                params[1], params[3], params[5] = self.draws(3, 1, -1)   # these shouldn't be 1 or -1
                params[2] = self.draw(params[3], -params[3])   # c shouldn't be +-d
                params[4] = self.draw(params[5], -params[5])   # e shouldn't be +-f

                lhs = UnsafePolynomial(Command('frac', [SingleVariablePolynomial(var,
                                                                                 [{'coefficient': 1, 'exponent': 1},
                                                                                  {'coefficient': params[0], 'exponent': 0}],
                                                                                 True).dumps(),
                                                        params[1]]).dumps(),
                                       Fraction(params[2], params[3], big=False), mix=True)
                rhs = Fraction(params[4], params[5], big=False)
            case 'double_bino_frac':
                # \frac{x + a}{b} + \frac{x + c}{d} = \frac{e}{f} has a solution as long as d != -b
                params[0], params[2] = self.draws(2)
                params[1], params[5] = self.draws(2, 1, -1)   # b, f shouldn't be 1 or -1
                params[3] = self.draw(1, -1, -params[1])   # d shouldn't be 1, -1, or -b
                params[4] = self.draw(params[5], -params[5])   # e shouldn't be +-f

                lhs = UnsafePolynomial(Command('frac', [SingleVariablePolynomial(var,
                                                                             [{'coefficient': 1, 'exponent': 1},
                                                                              {'coefficient': params[0], 'exponent': 0}],
                                                                             True).dumps(),
                                                    params[1]]).dumps(),
                                       Command('frac', [SingleVariablePolynomial(var,
                                                                                 [{'coefficient': 1, 'exponent': 1},
                                                                                  {'coefficient': params[2], 'exponent': 0}],
                                                                                 True).dumps(),
                                                        params[3]]).dumps())
                rhs = Fraction(params[4], params[5], big=False)
            case 'bino_frac_const':
                params[0], params[2], params[3] = self.draws(3)
                params[1] = self.draw(1, -1)   # b shouldn't be 1 or -1

                lhs = UnsafePolynomial(Command('frac', [SingleVariablePolynomial(var,
                                                                                 [{'coefficient': 1, 'exponent': 1},
                                                                                  {'coefficient': params[0], 'exponent': 0}],
                                                                                 True).dumps(),
                                                        params[1]]).dumps(),
                                       str(params[2]), mix=True)
                rhs = Number(params[3])
            case 'double_bino_frac_large':
                # frac{x + a}{b} + cx + d = \frac{x + e}{f} + g has a solution as long as f/b + fc != 1
                params[0], params[3], params[4], params[6] = self.draws(4)
                for _ in range(100):
                    b, f = self.draws(2, 1, -1)   # b and f shouldn't be 1 or -1
                    c = self.draw(1)   # c shouldn't be 1
                    if f/b + f*c != 1:
                        params[1] = b
                        params[2] = c
                        params[5] = f
                        break
                if params[1] == 0:
                    raise ValueError("The given nrange cannot seem to generate a double_bino_frac_large with a solution")

                lhs = UnsafePolynomial(Command('frac', [SingleVariablePolynomial(var,
                                                                                 [{'coefficient': 1, 'exponent': 1},
                                                                                  {'coefficient': params[0], 'exponent': 0}],
                                                                                 True).dumps(),
                                                        params[1]]).dumps(),
                                       SingleVariablePolynomial(var, [{'coefficient': params[2], 'exponent': 1}]).dumps(),
                                       str(params[3]),
                                       mix=True)
                rhs = UnsafePolynomial(Command('frac', [SingleVariablePolynomial(var,
                                                                                 [{'coefficient': 1, 'exponent': 1},
                                                                                  {'coefficient': params[4], 'exponent': 0}],
                                                                                 True).dumps(),
                                                        params[5]]).dumps(),
                                       str(params[6]), mix=True)
            case 'insane_1':
                # \frac{abx^(k+1) + acx^k}{ax^k} = d(ex + f + gx) has a solution as long as b != d(e + g) and c != df
                k = randint(1, 5)
                params[0], params[4], params[5], params[6] = self.draws(4)
                for _ in range(100):
                    b, c = self.draws(2)
                    d = self.draw(1)   # d shouldn't be 1
                    if b != d * (params[4] + params[6]) and c != d * params[5]:
                        params[1] = b
                        params[2] = c
                        params[3] = d
                        break
                if params[1] == 0:
                    raise ValueError("The given nrange cannot seem to generate an insane_1 with a solution")

                lhs = UnsafePolynomial(PolynomialFraction(SingleVariablePolynomial(var,
                                                                                   [{'coefficient': params[0] * params[1],
                                                                                     'exponent': k + 1},
                                                                                    {'coefficient': params[0] * params[2],
                                                                                     'exponent': k}], mix=True),
                                                          SingleVariablePolynomial(var,
                                                                                   [{'coefficient': params[0],
                                                                                     'exponent': k}])))
                rhs = TextWrapper(["{}({})".format(params[3] if params[3] != -1 else '-',
                                                   SingleVariablePolynomial(var,
                                                                            [{'coefficient': params[4],
                                                                              'exponent': 1},
                                                                             {'coefficient': params[5],
                                                                              'exponent': 0},
                                                                             {'coefficient': params[6],
                                                                              'exponent': 1}]).dumps())])
            case 'insane_2':
                # \sqrt{(a^2/100)x(b^2x)} = \sqrt{\frac{c^2}{d^2}} + ex has a solution as long as e != abd/10
                # x >= 0 if e <= |ab|/10
                params[2] = self.draw()
                for _ in range(100):
                    a = self.draw(1)   # a shouldn't be 1
                    b = self.draw()
                    d = self.draw(-1, 1, params[2], -params[2])   # d shouldn't be +-1 or +-c
                    if (a * b * d) % 10 == 0:   # since e is an integer, first condition is clear if this doesn't hold
                        e = self.draw(1, (params[0] * params[1] * params[3]) // 10)
                    else:
                        e = self.draw(1)
                    if e <= abs(a*b) / 10:
                        params[0] = a
                        params[1] = b
                        params[3] = d
                        params[4] = e
                        break
                if params[0] == 0:
                    raise ValueError("The given nrange cannot seem to generate an insane_1 with a solution")

                if random() < 0.5:
                    lhs = TextWrapper([Command('sqrt',
                                               "({})({})".format(Term(var, str(Decimal(params[0] ** 2) / 100), 1).dumps(),
                                                                 Term(var, (params[1] ** 2), 1).dumps())).dumps()])
                else:
                    lhs = TextWrapper([Command('sqrt',
                                               "({})({})".format(Term(var, (params[1] ** 2), 1).dumps(),
                                                                 Term(var, str(Decimal(params[0] ** 2) / 100), 1).dumps())).dumps()])
                rhs = UnsafePolynomial(Command('sqrt', Fraction(params[2] ** 2,
                                                                params[3] ** 2, big=False).dumps()).dumps(),
                                       Term(var, params[4], 1), mix=True)
            case 'insane_3':
                # \sqrt{ax(bx) + nx^2} + cx = \sqrt{(x + d)^2}, n is the smallest number such that ab + n is a perfect square
                # this has a solution as long as sqrt(ab + n) + c != 1
                # x + d >= 0 holds if x >= 0 and d >= 0
                # x >= 0 requires \sqrt{ab + n} + c - 1 >= 0 if d >= 0
                params[3] = self.draw(*tuple([n for n in range(self.nrange[0], self.nrange[1]) if n < 0]))
                for _ in range(100):
                    a, b, c = self.draws(3, 1)   # these shouldn't be 1
                    if a * b < 0:
                        b = -b   # a and b should have the same sign
                    k = math.sqrt(a * b)
                    if k % 1 < 0.0001:   # ab is a perfect square
                        n = (round(k) + 1) ** 2 - (a * b)
                    else:
                        n = math.ceil(k) ** 2 - (a * b)
                    if math.sqrt(a*b + n) + c - 1 >= 0.0001 :
                        params[0] = a
                        params[1] = b
                        params[2] = c
                        break
                if params[0] == 0:
                    raise ValueError("The given nrange cannot seem to generate an insane_3 with a solution")

                lhs = UnsafePolynomial(Command('sqrt',
                                               UnsafePolynomial("({})({})".format(Term(var, params[0], 1).dumps(),
                                                                                  Term(var, params[1], 1).dumps()),
                                                                Term(var, n, 2), mix=True).dumps()).dumps(),
                                       Term(var, params[2], 1), mix=True)
                rhs = TextWrapper([Command('sqrt',
                                           NoEscape("({})^2".format(SingleVariablePolynomial(var, [{'coefficient': 1, 'exponent': 1},
                                                                                                   {'coefficient': params[3], 'exponent': 0}], mix=True).dumps()))).dumps()])
            case 'insane_4':
                # (ax + b)^2 = (ax + c)^2 + d has a solution as long as b != c
                # (ax + b)^2 = (-ax + c)^2 + d has a solution as long as b != -c
                params[0], params[3] = self.draws(2)
                flipped = choice([True, False])
                o = -params[0] if flipped else params[0]
                for _ in range(100):
                    b, c = self.draws(2)
                    if (not flipped and b != c) or (flipped and b != -c):
                        params[1] = b
                        params[2] = c
                        break
                if params[0] == 0:
                    raise ValueError("The given nrange cannot seem to generate an insane_4 with a solution")

                lhs = TextWrapper(["({})^2".format(SingleVariablePolynomial(var, [{'coefficient': params[0], 'exponent': 1},
                                                                                  {'coefficient': params[1], 'exponent': 0}],
                                                                            mix=True).dumps())])
                rhs = UnsafePolynomial(TextWrapper(["({})^2".format(SingleVariablePolynomial(var, [{'coefficient': o, 'exponent': 1},
                                                                                                   {'coefficient': params[2], 'exponent': 0}],
                                                                                             mix=True).dumps())]),
                                       Number(params[3]))
            case 'insane_5':
                # ax + \sqrt{(b^2/100)x^2} = \sqrt{\frac{c^2}{d^2}}(ex + f) has a solution as long as d(10a + b) != 10ce
                # x >= 0 requires f / (10a|d| + |bd| - 10|c|e) >= 0
                for _ in range(100):
                    b, c, f = self.draws(3)
                    a, e = self.draws(2, 1)   # a and e shouldn't be 1
                    d = self.draw(1, -1, c, -c)   # d shouldn't be +-1 or +-c
                    if d * (10*a + b) != 10 * c * e and f / (10*a*abs(d) + abs(b*d) - 10*abs(c)*e) >= 0:
                        params[0] = a
                        params[1] = b
                        params[2] = c
                        params[3] = d
                        params[4] = e
                        params[5] = f
                        break
                if params[0] == 0:
                    raise ValueError("The given nrange cannot seem to generate an insane_4 with a solution")

                lhs = UnsafePolynomial(Term(var, params[0], 1),
                                       Command('sqrt',
                                               Term(var,
                                                    str(Decimal(params[1] ** 2) / 100),
                                                    2).dumps()).dumps())
                rhs = TextWrapper([Command('sqrt',
                                           Fraction(params[2] ** 2, params[3] ** 2, big=False).dumps()).dumps(),
                                   "({})".format(SingleVariablePolynomial(var, [{'coefficient': params[4], 'exponent': 1},
                                                                                {'coefficient': params[5], 'exponent': 0}]).dumps())])

        if random() < 0.5:
            result = lhs.get_latex() + [middle] + rhs.get_latex() + [NoEscape("\\quad")] + disclaimer
        else:
            result = rhs.get_latex() + [middle] + lhs.get_latex() + [NoEscape("\\quad")] + disclaimer

        self.num_quest -= 1
        return [Math(inline=True, data=result)]


class FactorPolynomial(EquationMultiOperation):
    def __init__(self, num_quest: int, nrange: tuple[int, int], *types: str, var=('x',)):
        """
        Initializes a polynomial factoring problem.
        The randomly generated numbers will never be 0.
        A denominator or an explicit coefficient (such as 'a' in ax + b) will never be 1.
        A denominator will never be 1.
        A fraction will always have distinct numerator and denominator.

        Possible equation types (the variable is always x, the rest are random. The order of terms may be randomized):

        number: single-variable polynomial with a common integer factor
        symbol: two-variable polynomial with a common variable factor
        twonum: two-variable polynomial with a common integer factor
        numsym: two-variable polynomial with common integer and variable factors
        mquad: quadratic polynomial that can be factored into two binomials, leading coefficient is 1
        nquad: quadratic polynomial that can be factored into two binomials, leading coefficient is +-1 (50/50 chance)
        quad: quadratic polynomial that can be factored into two binomials, leading coefficient isn't +-1
        quad_numsym: two-variable polynomial that can be factored to a monomial and two binomials in same variables
        quad_twosym: two-variable polynomial that can be factored into two binomials
        square: a perfect square of a single-variable binomial
        square_twosym: a perfect square of a two-variable binomial
        diffsq: the difference of a perfect square monomial and a perfect square constant
        quad_combine: quadratic polynomial that can be factored into two binomials, more than 3 terms

        :param num_quest: The number of questions to be generated.
        :param nrange: The range used for the numbers in the equation, (begin, end) inclusive.
        :param types: The types of equations to be used.
                      Refer to the docstring for the options and the description of each type.
                      If nothing is given, every type can appear.
        :param var: The potential variables to be used. The default is just x.
        """
        super().__init__(num_quest, nrange, var=var)
        self.vspace = "2.5cm"
        possible_types = ("number",
                          "symbol",
                          "twonum",
                          "numsym",
                          "mquad",
                          "nquad",
                          "quad",
                          "quad_numsym",
                          "quad_twosym",
                          "square",
                          "square_twosym",
                          "diffsq",
                          "quad_combine")
        for t in types:
            if t not in possible_types:
                raise ValueError(f"The problem type {t} is invalid")

        if not types:
            types = possible_types
        self.types = types

    def get_random_polynomial(self, prob_type: str) -> MultiVariablePolynomial:
        """
        Generates a polynomial with random non-zero parameters.
        A denominator or an explicit coefficient (such as 'a' in ax + b) will never be 1.
        A denominator will never be 1.
        A fraction will always have distinct numerator and denominator.

        Possible types (the variable is always x, the rest are random. The order of terms may be randomized):

        number: single-variable polynomial with a common integer factor
        symbol: two-variable polynomial with a common variable factor
        twonum: two-variable polynomial with a common integer factor
        numsym: two-variable polynomial with common integer and variable factors
        mquad: quadratic polynomial that can be factored into two binomials, leading coefficient is 1
        nquad: quadratic polynomial that can be factored into two binomials, leading coefficient is +-1 (50/50 chance)
        quad: quadratic polynomial that can be factored into two binomials, leading coefficient isn't +-1
        quad_numsym: two-variable polynomial that can be factored to a monomial and two binomials in same variables
        quad_twosym: two-variable polynomial that can be factored into two binomials
        square: a perfect square of a single-variable binomial
        square_twosym: a perfect square of a two-variable binomial
        diffsq: the difference of a perfect square monomial and a perfect square constant
        quad_combine: quadratic polynomial that can be factored into two binomials, more than 3 terms

        :param prob_type: The type of polynomial to be used.
                          Refer to the docstring for the options and the description of each type.
        """
        def two_vars() -> tuple[str, str]:
            if len(self.var) == 1:
                v1 = self.var[0]
                v2 = 'y' if v1 == 'x' else 'x'
            else:
                v1, v2 = tuple(sample(self.var, 2))
            return v1, v2

        poly = None

        match prob_type:
            case "number":
                var = choice(self.var)
                deg = randint(2, 4)
                exps = sample(list(range(deg)), randint(2, deg))
                exps.insert(randint(0, len(exps)), deg)

                terms = []
                for exp in exps:
                    terms.append(MultiVariableTerm(self.draw(), (var, exp)))
                poly = (MultiVariablePolynomial(terms) * self.draw(1, -1))
            case "symbol":
                var1, var2 = two_vars()
                exp_candidates = []
                for i in range(4):
                    for j in range(4):
                        exp_candidates.append((i, j))
                exps = sample(exp_candidates, randint(1, 4))
                # add single factors so that the polynomial cannot be factored at this point
                exps.insert(randint(0, len(exps)), (randint(1, 4), 0))
                exps.insert(randint(0, len(exps)), (0, randint(1, 4)))

                terms = []
                for exp in exps:
                    terms.append(MultiVariableTerm(self.draw(), (var1, exp[0]), (var2, exp[1])))

                e1 = randint(0, 3)
                e2 = randint(0, 3)
                if e1 == 0 and e2 == 0:
                    if random() < 0.5:
                        e1 = randint(1, 3)
                    else:
                        e2 = randint(1, 3)
                term = MultiVariableTerm(1, (var1, e1), (var2, e2))
                poly = MultiVariablePolynomial(terms) * term
            case "twonum":
                var1, var2 = two_vars()
                exp_candidates = []
                for i in range(4):
                    for j in range(4):
                        exp_candidates.append((i, j))
                exps = sample(exp_candidates, randint(1, 4))
                # add single factors so that the polynomial cannot be factored at this point
                exps.insert(randint(0, len(exps)), (randint(1, 4), 0))
                exps.insert(randint(0, len(exps)), (0, randint(1, 4)))

                terms = []
                for exp in exps:
                    terms.append(MultiVariableTerm(self.draw(1, -1), (var1, exp[0]), (var2, exp[1])))

                term = MultiVariableTerm(self.draw())
                poly = MultiVariablePolynomial(terms) * term
            case "numsym":
                var1, var2 = two_vars()
                exp_candidates = []
                for i in range(4):
                    for j in range(4):
                        exp_candidates.append((i, j))
                exps = sample(exp_candidates, randint(1, 4))
                # add single factors so that the polynomial cannot be factored at this point
                exps.insert(randint(0, len(exps)), (randint(1, 4), 0))
                exps.insert(randint(0, len(exps)), (0, randint(1, 4)))

                terms = []
                for exp in exps:
                    terms.append(MultiVariableTerm(self.draw(1, -1), (var1, exp[0]), (var2, exp[1])))

                e1 = randint(0, 3)
                e2 = randint(0, 3)
                if e1 == 0 and e2 == 0:
                    if random() < 0.5:
                        e1 = randint(1, 3)
                    else:
                        e2 = randint(1, 3)
                term = MultiVariableTerm(self.draw(), (var1, e1), (var2, e2))
                poly = MultiVariablePolynomial(terms) * term
            case "mquad":
                var = choice(self.var)
                n, m = self.draws(2)
                poly = SingleVariablePolynomial(var, [
                    {"coefficient": 1, "exponent": 2},
                    {"coefficient": n + m, "exponent": 1},
                    {"coefficient": n * m, "exponent": 0}
                ])
            case "nquad":
                var = choice(self.var)
                n, m = self.draws(2)
                s = 1 if random() < 0.5 else -1
                poly = SingleVariablePolynomial(var, [
                    {"coefficient": s, "exponent": 2},
                    {"coefficient": n + s*m, "exponent": 1},
                    {"coefficient": n * m, "exponent": 0}
                ])
            case "quad":
                var = choice(self.var)
                a, b, n, m = self.draws(4)
                if (a == 1 and b == 1) or (a == -1 and b == -1):
                    a = self.draw(1, -1)
                poly = SingleVariablePolynomial(var, [
                    {"coefficient": a * b, "exponent": 2},
                    {"coefficient": a*m + b*n, "exponent": 1},
                    {"coefficient": n * m, "exponent": 0}
                ])
            case "quad_numsym":
                var1, var2 = two_vars()
                a, b, n, m = self.draws(4)
                exp1 = randint(0, 4)
                exp2 = randint(1, 4)
                poly = MultiVariablePolynomial([
                    MultiVariableTerm(a * b, (var1, 2)),
                    MultiVariableTerm(a*m + b*n, (var1, 1)),
                    MultiVariableTerm(n * m, (var1, 0))
                ])
                poly *= MultiVariableTerm(self.draw(), (var1, exp1), (var2, exp2))
            case "quad_twosym":
                var1, var2 = two_vars()
                a, b, n, m = self.draws(4)
                poly = MultiVariablePolynomial([
                    MultiVariableTerm(a * b, (var1, 2)),
                    MultiVariableTerm(a*m + b*n, (var1, 1), (var2, 1)),
                    MultiVariableTerm(n * m, (var2, 2))
                ])
            case "square":
                var = choice(self.var)
                a, n = self.draws(2)
                poly = SingleVariablePolynomial(var, [
                    {"coefficient": a**2, "exponent": 2},
                    {"coefficient": 2*a*n, "exponent": 1},
                    {"coefficient": n**2, "exponent": 0}
                ])
            case "square_twosym":
                var1, var2 = two_vars()
                a, n = self.draws(2)
                poly = MultiVariablePolynomial([
                    MultiVariableTerm(a**2, (var1, 2)),
                    MultiVariableTerm(2*a*n, (var1, 1), (var2, 1)),
                    MultiVariableTerm(n**2, (var2, 2))
                ])
            case "diffsq":
                var = choice(self.var)
                a, n = self.draws(2)
                poly = SingleVariablePolynomial(var, [
                    {"coefficient": a**2, "exponent": 2},
                    {"coefficient": -n**2, "exponent": 0}
                ])
            case "quad_combine":
                var = choice(self.var)
                a, b, n, m = self.draws(4)   # (ax + n)(bx + m)
                # c[0]x^2 + c[1]x + c[2]
                c = [a * b, a*m + b*n, n * m]
                s = [0, 0, 0]   # amount to subtract from coefficients
                while s == [0, 0, 0]:
                    for i in range(3):
                        if random() < 0.8:   # 20% chance to be left at 0
                            s[i] = self.draw(c[i])
                poly = SingleVariablePolynomial(var, [
                    Term(var, c[0] - s[0], 2),
                    Term(var, c[1] - s[1], 1),
                    Term(var, c[2] - s[2], 0)
                ])
                for i in range(3):
                    if s[i] != 0:
                        poly.append(Term(var, s[i], 2-i))
                poly.mix()

        return poly

    def get_problem(self):
        prob_type = choice(self.types)
        poly = self.get_random_polynomial(prob_type)
        self.num_quest -= 1
        return [Math(inline=True, data=[poly.dumps()])]


class QuadraticEquation(EquationMultiOperation):
    def __init__(self, num_quest: int, nrange: tuple[int, int], *types: str, var=('x',), inequality=False):
        r"""
        Initializes a problem where a quadratic equation must be solved.
        The randomly generated numbers will never be 0.
        A denominator or an explicit coefficient (such as 'a' in ax + b) will never be 1.
        A fraction will always have distinct numerator and denominator.

        Possible equation types (the variable is always x, the rest are random. The order of terms may be randomized):

        fact_standard: ax^2 + bx + c = 0, can be solved by factoring
        fact_separated: ax^2 + bx = c, can be solved by factoring
        fact_double: ax^2 + bx + c = dx^2 + ex + f, can be solved by factoring


        :param num_quest: The number of questions to be generated.
        :param nrange: The range used for generating the numbers in the equation, (begin, end) inclusive.
                       The numbers in the actual problem won't necessarily fall into the given interval.
        :param types: The types of equations to be used.
                      Refer to the docstring for the options and the description of each type.
                      If nothing is given, every type can appear.
        :param var: The potential variables to be used. The default is just x.
        :param inequality: If True, the problems will be inequality problems instead.
                           One of the four inequalities will be randomly chosen for each question.
        """
        super().__init__(num_quest, nrange, var=var, inequality=inequality)
        possible_types = ('fact_standard',
                          'fact_separated',
                          'fact_double')
        for t in types:
            if t not in possible_types:
                raise ValueError(f"The problem type {t} is invalid")

        if not types:
            self.types = possible_types
        else:
            self.types = types

    def get_problem(self):
        if self.inequality:
            middle = choice(['>', '<', Command('geq'), Command('leq')])
        else:
            middle = '='
        var = choice(self.var)
        prob_type = choice(self.types)
        lhs = None
        rhs = None

        match prob_type:
            case 'fact_standard':
                # generate by expanding (ax + b)(cx + d) = 0
                a, b, c, d = self.draws(4)
                lhs = SingleVariablePolynomial(var, [
                    {'coefficient': a*c, 'exponent': 2},
                    {'coefficient': a*d + b*c, 'exponent': 1},
                    {'coefficient': b*d, 'exponent': 0}
                ], mix=True).remove_zeros()
                rhs = TextWrapper(["0"])
            case 'fact_separated':
                # generate by expanding (ax + b)(cx + d) = 0 and isolating constant
                a, b, c, d = self.draws(4)
                lhs = SingleVariablePolynomial(var, [
                    {'coefficient': a*c, 'exponent': 2},
                    {'coefficient': a*d + b*c, 'exponent': 1}
                ], mix=True).remove_zeros()
                rhs = TextWrapper([str(-b * d)])
            case 'fact_double':
                # generate lhs by expanding (ax + b)(cx + d)
                a, b, c, d = self.draws(4)
                lhs = SingleVariablePolynomial(var, [
                    {'coefficient': a*c, 'exponent': 2},
                    {'coefficient': a*d + b*c, 'exponent': 1},
                    {'coefficient': b*d, 'exponent': 0}
                ])
                rhs = SingleVariablePolynomial(var, [])
                # separate by generating differences
                diff = [0, 0, 0]
                for i in range(3):
                    if random() < 0.8:   # 80% chance for each term to be separated
                        diff[i] = self.draw()
                if diff == [0, 0, 0]:
                    # at least one term should be separated
                    diff[randint(0, 2)] = self.draw()
                for i in range(3):
                    coef = lhs[i].get_signed_coefficient() + diff[i]
                    lhs[i].coefficient = Number(coef.mag)
                    lhs[i].sign = coef.sign
                    rhs.append(Term(var, diff[i], 2-i))
                lhs.remove_zeros().mix()
                rhs.remove_zeros().mix()

        self.num_quest -= 1
        if random() < 0.5:
            return [Math(inline=True, data=lhs.get_latex() + [middle] + rhs.get_latex())]
        else:
            return [Math(inline=True, data=rhs.get_latex() + [middle] + lhs.get_latex())]


class LinearSystem(ProblemBase):
    def __init__(self, num_quest: int, crange: tuple[int, int], size: int, *types: str, var: tuple[str, ...]=tuple(), solvability=0):
        """
        Initializes a problem where a system of linear equations must be solved.
        In standard form, the rank of the coefficient matrix will be at least 2 unless size is 1.

        The possible problem types are as follows:
        - standard: size * size system in standard form (Ax = b)

        :param num_quest: The number of questions to be generated.
        :param crange: The range of generated coefficients, (begin, end) inclusive.
        :param size: The number of variables in the problem. Must be at least 2.
        :param types: The problem types to be used (one will be randomly chosen for each problem).
                      Refer to the docstring for the possible types. If omitted, the problems will
                      be drawn from all possible types.
        :param var: The variable names to be used. If fewer than size, more will be added
                    in the priority x, y, z, w, t, u, v, p, q, r to match size. If this runs out,
                    the code results in an error. If more variables than size are provided, then
                    they will be randomly drawn from the tuple.
        :param solvability: When set to 0 (default), the system has a unique solution.
                            When set to 1, the system has no solution.
                            When set to 2, the system has infinitely many solutions.
                            When set to 3, the system has 0, 1, or infinitely many solutions by uniform probability.
        """
        super().__init__(num_quest, "6cm")
        if size < 2:
            raise ValueError("The problem size must be 2")
        possible_types = (
            "standard",
        )
        for t in types:
            if t not in possible_types:
                raise ValueError(f"Problem type {t} is invalid; refer to docstring for the list of possible types")
        if not types:
            types = possible_types
        if solvability not in (0, 1, 2, 3):
            raise ValueError("Invalid argument value for solvability")
        if len(var) < size:
            for v in ('x', 'y', 'z', 'w', 't', 'u', 'v', 'p', 'q', 'r'):
                if v not in var:
                    var = (*var, v)
                if len(var) == size:
                    break
            if len(var) < size:
                raise ValueError("Not enough variables for the given size; you must manually assign additional variables")

        self.crange = crange
        self.size = size
        self.var = var
        self.types = types
        self.solvability = solvability

    def generate_standard_system(self) -> np.ndarray:
        """
        Returns a size * (size + 1) integer array that represents the augmented coeffiicient matrix
        Ax = b as (A|b). The system may or may not be consistent based on the solvability value.
        """
        if self.solvability == 0 or (self.solvability == 3 and random() < 1/3):
            system = np.random.randint(self.crange[0], self.crange[1] + 1, (self.size, self.size + 1))
            for _ in range(100):
                if np.linalg.det(system[:, :-1]) == 0:
                    system = np.random.randint(self.crange[0], self.crange[1] + 1, (self.size, self.size + 1))
                else:
                    break
            if np.linalg.det(system[:, :-1]) == 0:
                raise ValueError(f"The given crange of {self.crange} cannot seem to generate a consistent system")
            return system
        else:
            # choose the LHS matrix rank
            rank = randint(min(self.size - 1, 2), self.size - 1)

            def generate_independent_rows():
                result = np.random.randint(self.crange[0], self.crange[1] + 1, (rank, self.size))
                if rank > 1:
                    # check linear independence
                    for _ in range(100):
                        sol = sp.solvers.linsolve(sp.Matrix(np.hstack((result.T, np.zeros((self.size, 1), dtype=np.int64)))))
                        if len(sol) == 1:
                            break
                    if len(sol) != 1:
                        raise ValueError(f"The given crange of {self.crange} cannot seem to generate a system with rank {rank}")
                return result

            # generate dependent rows using linear combinations
            done = False
            for _ in range(100):
                system = generate_independent_rows()
                valid_system = True
                while system.shape[0] < self.size and valid_system:
                    basis = system[:rank, :]
                    valid_row = False
                    for _ in range(1000):
                        # try a random combination and see if it falls inside the range
                        combined = np.zeros(self.size, np.int64)
                        factors = np.random.randint(-5, 6, rank)
                        while not np.any(factors):
                            factors = np.random.randint(-5, 6, rank)
                        for i in range(rank):
                            combined += factors[i] * basis[i]
                        if np.all(self.crange[0] <= combined) and np.all(combined <= self.crange[1]):
                            valid_row = True
                            break
                    if valid_row:
                        system = np.vstack((system, combined))
                    else:
                        valid_system = False
                if valid_system:
                    done = True
                    break
            if not done:
                raise ValueError(f"The given crange cannot seem to generate a system with rank {rank}")
            # append the appropriate RHS
            if self.solvability == 1 or (self.solvability == 3 and random() < 0.5):
                # choose RHS to have no solution
                # this is more likely, so just use a random RHS and check
                done = False
                for _ in range(100):
                    rhs = np.random.randint(self.crange[0], self.crange[1] + 1, (self.size, 1))
                    if sp.solvers.linsolve(sp.Matrix(np.hstack((system, rhs)))).is_empty:
                        done = True
                        break
                assert done, f"No RHS seems to result in having no solution, the LHS is:\n{system}"
                return np.hstack((system, rhs))
            else:
                # choose RHS to have infinitely many solutions
                # try different vectors from column space that fit inside the range
                column_space = [np.array(a).astype(np.int64) for a in sp.Matrix(system).columnspace()]
                for _ in range(1000):
                    combined = np.zeros((self.size, 1), np.int64)
                    factors = np.random.randint(-5, 6, len(column_space))
                    while not np.any(factors):
                        factors = np.random.randint(-5, 6, len(column_space))
                    for i in range(len(column_space)):
                        combined += factors[i] * column_space[i]
                    if np.all(self.crange[0] <= combined) and np.all(combined <= self.crange[1]):
                        return np.hstack((system, combined))
                # end of loop reached without finding a valid RHS; simply try again
                print("WARNING: failed attempt at generating a system with infinitely many solutions; reattempting with a different LHS")
                return self.generate_standard_system()

    def get_problem(self) -> list[Math]:
        problem = choice(self.types)
        variables = sample(self.var, self.size)
        result = [r"\begin{cases}"]

        match problem:
            case "standard":
                coefficients = self.generate_standard_system()
                for i in range(self.size):
                    lhs = list(coefficients[i, :-1])
                    rhs = coefficients[i, -1]
                    lhs_terms = []
                    for j, n in enumerate(lhs):
                        lhs_terms.append(Term(variables[j], int(n), 1))
                    lhs = MultiVariablePolynomial(lhs_terms).remove_zeros()
                    result.append(lhs.dumps() + "&=" + str(rhs))
                    result.append(r"\\" + "\n")
        result.pop(-1)   # remove the last newline
        result.append(r"\end{cases}")
        result = [NoEscape(s) for s in result]

        self.num_quest -= 1
        return [Math(inline=True, data=result)]


class RadicalSimplify(ProblemBase):
    def __init__(self, num_quest: int, nrange: tuple[int, int], erange=(2, 2)):
        """
        Generates problems where a radical expression must be simplified.
        The base of the generated radical is a * b^e,
        where a, b are drawn from nrange and e is drawn from erange.
        a, b are never perfect e-power.

        :param num_quest: The number of questions to be generated.
        :param nrange: The range of values used to generate the radical expressions, (begin, end) inclusive.
                       The lower bound must be at least 2 and the range must contain
                       at least one integer that is not a perfect e-power for all possible values in erange.
        :param erange: The range of possible exponents of the radical expression, (begin, end) inclusive.
                       The lower bound must be at least 2 and must contain at least one integer.
                       By default, all exponents will be 2.
        """
        if nrange[0] < 2 or erange[0] < 2:
            raise ValueError("The lower bound of the ranges must be at least 2")
        if nrange[0] > nrange[1] or erange[0] > erange[1]:
            raise ValueError("At least one of the ranges do not contain any integer")
        for e in range(erange[0], erange[1] + 1):
            valid = False
            n = nrange[0]
            while not valid:
                valid = RadicalSimplify.not_perfect_power(n, e)
                if n > nrange[1]:
                    raise ValueError(f"The given nrange cannot generate a non-perfect-{e}-power")
                else:
                    n += 1

        super().__init__(num_quest, "0cm")
        self.nrange = nrange
        self.erange = erange

    @staticmethod
    def not_perfect_power(n: int, e: int, tol=0.0001) -> bool:
        """
        Returns True if \\sqrt[e]{n} doesn't evaluate to an integer, False otherwise.
        """
        return (n**(1/e)) % 1 > tol

    def get_problem(self) -> Math:
        e = randint(self.erange[0], self.erange[1])
        a = randint(self.nrange[0], self.nrange[1])
        while not RadicalSimplify.not_perfect_power(a, e):
            # this loop should theoretically end thanks to the check in __init__
            a = randint(self.nrange[0], self.nrange[1])
        b = randint(self.nrange[0], self.nrange[1])
        while not RadicalSimplify.not_perfect_power(b, e):
            b = randint(self.nrange[0], self.nrange[1])

        self.num_quest -= 1
        if e == 2:
            return Math(inline=True, data=[Command("sqrt", a * b**e), "="])
        else:
            return Math(inline=True, data=[Command("sqrt", a * b**e, e), "="])


class QuadraticGraphingFactorable(GraphingProblem, FactorPolynomial):
    def __init__(self, num_quest: int, nrange: tuple[int, int], *types: str):
        """
        Initializes a problem where a quadratic function must be graphed,
        where the quadratic function can be factored.

        :param num_quest: The number of questions to be generated.
        :param nrange: The range used for the numbers in the polynomial, (begin, end) inclusive.
        :param types: The types of polynomials to be used.
                      Refer to FactorPolynomial's docstring for the options.
                      You can only use quadratic types that use one variable, others will be ignored.
                      If nothing is given, every type can appear.
        """
        GraphingProblem.__init__(self, num_quest)
        FactorPolynomial.__init__(self, num_quest, nrange, *types, var=('x',))
        # remove non-quadratic problems or those with more than one variable
        prohibited_types = ("number",
                            "symbol",
                            "twonum",
                            "numsym",
                            "quad_numsym",
                            "quad_twosym",
                            "square_twosym")
        self.types = tuple(t for t in self.types if t not in prohibited_types)
        if not self.types:
            raise ValueError("All the provided problem types are invalid for quadratic graphing")

    def get_random_function(self) -> NoEscape:
        return self.get_random_polynomial(choice(self.types)).dumps()


class IdentifyGraph(ProblemBase, ABC):
    def __init__(self, num_quest: int):
        """
        Initializes a problem where the equation of the graph must be identified.

        :param num_quest: The number of questions to be generated.
        """
        super().__init__(num_quest, "0cm")

    @abstractmethod
    def get_random_function(self) -> tuple[NoEscape, tuple[int, int], tuple[int, int]]:
        """
        Returns a function with randomized parameters,
        along with the axes limits to be used for graphing.

        :return: function expression, xlim, ylim
        """
        pass   # child class should implement this

    def get_problem(self) -> list[DocInjector]:
        def body(doc: Document):
            func, xlim, ylim = self.get_random_function()
            xtick_distance = max(1, (xlim[1] - xlim[0]) // 3)
            ytick_distance = max(1, (ylim[1] - ylim[0]) // 3)
            with doc.create(TikZ()):
                axis_options = rf"""
                height=6cm, width=6cm,
                grid=both,
                grid style={{line width=.1pt, draw=gray!10}},
                major grid style={{line width=.2pt, draw=gray!50}},
                axis lines=center,
                xmin={xlim[0]}, xmax={xlim[1]}, ymin={ylim[0]}, ymax={ylim[1]},
                xtick distance={xtick_distance}, minor x tick num={xtick_distance - 1},
                ytick distance={ytick_distance}, minor y tick num={ytick_distance - 1},
                xlabel=\(x\), ylabel=\(y\)
                """
                with doc.create(Axis(options=NoEscape(axis_options))) as plot:
                    plot_options = rf"""
                    samples=100,
                    domain={xlim[0]}:{xlim[1]},
                    line width=1pt,
                    mark=none
                    """
                    plot.append(Plot(func=func, options=NoEscape(plot_options)))

        self.num_quest -= 1
        return [DocInjector(body)]


class IdentifyQuadraticGraph(IdentifyGraph):
    def __init__(self, num_quest: int, arange: tuple[int, int], vrange: tuple[int, int], show_intercepts=False):
        """
        Initializes problems where a graph of a quadratic relation is drawn
        and the student needs to find an equation for it.
        The graph always shows the vertex.

        :param num_quest: The number of questions to be generated.
        :param arange: The range used for the coefficient of the x^2 term, (begin, end) inclusive.
                       There's a 50:50 chance that the reciprocal of the generated number is used instead.
                       a will never be 0 even if arange includes it.
        :param vrange: The range used for the x and y-coordinate of the vertex, (begin, end) inclusive.
        :param show_intercepts: If True, the x-intercept(s) are guaranteed to be shown (if they exist).
        """
        super().__init__(num_quest)
        if 0 in arange:
            raise ValueError("arange bounds cannot be 0")
        self.arange = arange
        self.vrange = vrange
        self.show_intercepts = show_intercepts

    def get_random_function(self):
        a = 0
        while a == 0:
            a = randint(self.arange[0], self.arange[1])
        if a != 0 and random() < 0.5:
            a = 1 / a
        vx = randint(self.vrange[0], self.vrange[1])
        vy = randint(self.vrange[0], self.vrange[1])

        # determine view region
        r = int(max(abs(a), abs(1 / a)) * 3)
        jitter = max(r // 6, 1)
        xmin = vx - r + randint(1, jitter)
        xmax = vx + r + randint(1, jitter)
        ymin = vy - r + randint(1, jitter)
        ymax = vy + r + randint(1, jitter)
        if self.show_intercepts and a * vy < 0:
            # there will be two x-intercepts, make sure they show up
            b = -2 * a * vx
            c = a * vx**2 + vy
            x1 = math.floor((-b + math.sqrt(b**2 - 4*a*c)) / (2*a))   # rounded down to nearest integer
            x2 = math.ceil((-b - math.sqrt(b**2 - 4*a*c)) / (2*a))   # rounded up to nearest integer
            xmin = min(xmin, x1-1, x2-1)
            xmax = max(xmax, x1+1, x2+1)
            if a > 0:
                ymax = max(ymax, 1)
            else:
                ymin = min(ymin, -1)

        return NoEscape(f"({a}) * (x - ({vx})) * (x - ({vx})) + ({vy})"), (xmin, xmax), (ymin, ymax)


class ExponentRulePractice(ProblemBase):
    def __init__(self, num_quest: int, brange: tuple[int, int], erange: tuple[int, int], *types: str, sqrt_only=False):
        r"""
        Initializes problems where an expression needs to be simplified using exponent rules.
        The bases are drawn randomly from brange and the exponents are drawn randomly from erange.
        The drawn base will never be 0 or +-1.
        The drawn exponent will never be 1.

        Here are the possible problem types:
        - simple_mult: b^(e1) * b^(e2)
        - simple_div: \frac{b^(e1)}{b^(e2)}, e1 >= e2
        - simple_exp: (b^(e1))^(e2)
        - simple_dist: ((b1)^(e1) * (b2)^(e2))^(e3)
        - simple_root: \sqrt[e2]{b^(e1*e2)}, e2 > 1 and e2 != 0
            - If erange doesn't include anything greater than 1, e2 will always be 2 even if it's outside the range.
            - This problem type requires erange to also contain something other than 0.
        - multdiv: product/quotient of b^(en), n = 3-5
        - twobase_multdiv: product/quotient of (b1)^(en) and (b2)^(en), n = 4-6

        :param num_quest: The number of questions to be generated.
        :param brange: The range used for bases, (begin, end) inclusive. Must contain something other than 0 and +-1.
        :param erange: The range used for exponents (including radical index), (begin, end) inclusive.
                       Must contain something other than 1.
        :param types: The problem types to be chosen from. If omitted, all types will be allowed.
        :param sqrt_only: If True, radical expressions will only use square roots regardless of erange.
        """
        if (brange[0] > brange[1]) or not set(range(brange[0], brange[1] + 1)) - {-1, 0, 1}:
            raise ValueError("brange must contain a number other than 0 and 1. brange given: " + str(brange))
        if (erange[0] > erange[1]) or (erange[0] == 1 and erange[1] == 1):
            raise ValueError("erange must contain a number other than 1. erange given: " + str(erange))
        possible_types = (
            "simple_mult",
            "simple_div",
            "simple_exp",
            "simple_dist",
            "simple_root",
            "multdiv",
            "twobase_multdiv"
        )
        root_types = {
            "simple_root"
        }
        for t in types:
            if t not in possible_types:
                raise ValueError(f"the problem type {t} is not valid")
        if not types:
            types = possible_types
        if not sqrt_only and not set(range(erange[0], erange[1] + 1)) - {0, 1}:
            # remove root types
            types = tuple(set(types) - root_types)
        if not types:
            raise ValueError("Only root problems chosen but erange contains nothing but 0 and/or 1")
        super().__init__(num_quest, "0cm")
        self.brange = brange
        self.erange = erange
        self.types = types
        self.sqrt_only = sqrt_only


    def get_problem(self) -> list[Math]:
        def draw_b(wrap) -> Number:
            b = randint(self.brange[0], self.brange[1])
            while b in [0, 1, -1]:
                b = randint(self.brange[0], self.brange[1])
            return Number(b, wrap=wrap)
        def draw_e(lower_bound=None, upper_bound=None, no_zero=False) -> int:
            if lower_bound is None:
                lower_bound = self.erange[0]
            if upper_bound is None:
                upper_bound = self.erange[1]
            if lower_bound < self.erange[0] or upper_bound > self.erange[1]:
                raise ValueError(f"draw_e bounds set outside erange: lower_bound={lower_bound}, upper_bound={upper_bound}, erange={self.erange}")
            e = randint(lower_bound, upper_bound)
            while e == 1 or (no_zero and e == 0):
                e = randint(lower_bound, upper_bound)
            return e

        prob_type = choice(self.types)
        result = []

        match prob_type:
            case "simple_mult":
                b = draw_b(True)
                e1 = draw_e()
                e2 = draw_e()
                result.extend([NoEscape(f"{b.dumps()}^{{{e1}}}"), Command("times"), NoEscape(f"{b.dumps()}^{{{e2}}}")])
            case "simple_div":
                b = draw_b(True)
                e1 = draw_e()
                e2 = draw_e(upper_bound=e1)   # make sure e1 >= e2
                result.extend(UnsafeFraction(f"{b.dumps()}^{{{e1}}}", f"{b.dumps()}^{{{e2}}}").get_latex())
            case "simple_exp":
                b = draw_b(True)
                e1 = draw_e()
                e2 = draw_e()
                result.extend([Command("left("), NoEscape(f"{b.dumps()}^{{{e1}}}"), Command("right)"), NoEscape(f"^{{{e2}}}")])
            case "simple_dist":
                b1 = draw_b(True)
                b2 = draw_b(True)
                e1 = draw_e()
                e2 = draw_e()
                e3 = draw_e()
                result.extend([Command("left("), *b1.get_latex(), NoEscape(f"^{{{e1}}}"),
                               Command("times"), *b2.get_latex(), NoEscape(f"^{{{e2}}}"),
                               Command("right)"), NoEscape(f"^{{{e3}}}")])
            case "simple_root":
                b = draw_b(True)
                e1 = draw_e()
                e2 = 2 if (self.sqrt_only or self.erange[1] < 2) else draw_e(lower_bound=max(2, self.erange[0]), no_zero=True)
                result.append(Command("sqrt", NoEscape(f"{b.dumps()}^{{{e1 * e2}}}"), e2 if e2 != 2 else None))
            case "multdiv":
                b = draw_b(True)
                enum = [draw_e()]   # put at least one power in numerator
                edenom = []
                for _ in range(randint(2, 4)):
                    if random() < 0.5:
                        enum.append(draw_e())
                    else:
                        edenom.append(draw_e())
                num = NoEscape(f"{b.dumps()}^{{{enum[0]}}}")
                for e in enum[1:]:
                    num += NoEscape(f" \\times {b.dumps()}^{{{e}}}")
                if not edenom:
                    result.append(num)
                else:
                    denom = NoEscape(f"{b.dumps()}^{{{edenom[0]}}}")
                    for e in edenom[1:]:
                        denom += NoEscape(f" \\times {b.dumps()}^{{{e}}}")
                    result.extend(UnsafeFraction(num, denom).get_latex())
            case "twobase_multdiv":
                b1 = draw_b(True)
                b2 = draw_b(True)
                while self.brange[0] < self.brange[1] and b1 == b2:
                    b2 = draw_b(True)
                # choose exponents for each base
                b1_e = [draw_e()]
                b2_e = [draw_e()]
                for _ in range(randint(2, 4)):
                    if random() < 0.5:
                        b1_e.append(draw_e())
                    else:
                        b2_e.append(draw_e())
                # randomly assign exponents to numerator and denominator
                indices = [(i, b1_e, b1) for i in range(len(b1_e))] + [(i, b2_e, b2) for i in range(len(b2_e))]
                inum = []
                idenom = []
                while indices:
                    j = randint(0, len(indices) - 1)
                    if not inum or random() < 0.5:   # make sure numerator has at least one power
                        inum.append(indices.pop(j))
                    else:
                        idenom.append(indices.pop(j))
                # assemble the numerator text
                num = NoEscape(f"{inum[0][2].dumps()}^{{{inum[0][1][inum[0][0]]}}}")
                for i, be, b in inum[1:]:
                    num += NoEscape(f" \\times {b.dumps()}^{{{be[i]}}}")
                # assemble the denominator text
                if not idenom:
                    result.append(num)
                else:
                    denom = NoEscape(f"{idenom[0][2].dumps()}^{{{idenom[0][1][idenom[0][0]]}}}")
                    for i, be, b in idenom[1:]:
                        denom += NoEscape(f" \\times {b.dumps()}^{{{be[i]}}}")
                    result.extend(UnsafeFraction(num, denom).get_latex())

        result.append(NoEscape("="))

        self.num_quest -= 1
        return [Math(data=result, inline=True)]
