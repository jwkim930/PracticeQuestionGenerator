from random import randint, choice, shuffle, random, sample
from decimal import Decimal
from abc import ABC, abstractmethod

from pylatex import Math, Alignat, Command, Tabular, MiniPage
from pylatex.base_classes import LatexObject
from pylatex.utils import NoEscape

from classes.math_objects import *


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
    def get_problem(self) -> Math | Alignat | NoEscape:
        """
        Returns a practice problem.
        Decrements num_quest in the object.
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

    def get_problem(self) -> Math:
        o1, o2 = self.generate_random_operands()
        self.num_quest -= 1
        return Math(inline=True, data=o1.get_latex() + [self.operand] + o2.get_latex() + ["="])


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

    def get_problem(self) -> NoEscape:
        result = self.problem

        for prange in self.ranges:
            result = result.replace('@', str(randint(prange[0], prange[1])), 1)

        self.num_quest -= 1
        return NoEscape(result)


class GraphingProblem(ProblemBase, ABC):
    def __init__(self, num_quest: int):
        """
        Initializes a graphing problem.
        It requires graphing_grid.png in the document_output folder.
        """
        super().__init__(num_quest, '0cm')

    @abstractmethod
    def get_random_function(self) -> str:
        """
        Returns a function with randomized parameters.
        """
        pass   # child class should implement this

    def get_problem(self) -> NoEscape:
        result = NoEscape('$f(x)=' + self.get_random_function() + '$\n\n' \
               + r'\vspace{2em}' + '\n\n' \
               + r'\includegraphics[width=0.3\linewidth]{graphing_grid.png}')

        self.num_quest -= 1
        return result


class LinearGraphingProblem(GraphingProblem):
    def __init__(self, num_quest: int, a_range: tuple[int, int], b_range: tuple[int, int]):
        """
        Initializes a graphing problem for a linear function.
        The produced function will be in the form ax + b.
        It requires graphing_grid.png in the document_output folder.

        :param num_quest: The number of questions to be generated.
        :param a_range: The range for the coefficient, (begin, end) inclusive. 0 is automatically excluded.
        :param b_range: The range for the constant, (begin, end) inclusive.
        """
        super().__init__(num_quest)
        self.a_range = a_range
        self.b_range = b_range

    def get_random_function(self):
        a = randint(self.a_range[0], self.a_range[1])
        while a == 0:
            a = randint(self.a_range[0], self.a_range[1])
        b = randint(self.b_range[0], self.b_range[1])
        a_str = ''
        if a == -1:
            a_str = '-'
        elif a != 1:
            a_str = str(a)
        b_str = ''
        if b < 0:
            b_str = '-' + str(-b)
        elif b != 0:
            b_str = '+' + str(b)

        return a_str + 'x' + b_str


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

    def get_problem(self) -> Math:
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
        return result


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

    def get_problem(self) -> NoEscape:
        def decimal_randrange(start, end, step) -> Decimal:
            count = int((end - start) // step)   # 1 less than the total number of candidates
            return start + randint(0, count) * step

        # generate the parameters
        h = decimal_randrange(self.h_range[0], self.h_range[1], self.h_range[2])
        while self.no_constant and h == 0:
            h = decimal_randrange(self.h_range[0], self.h_range[1], self.h_range[2])
        k = decimal_randrange(self.k_range[0], self.k_range[1], self.k_range[2])

        # assemble the table of values
        xy_table = Tabular("c|c")
        xy_table.add_row(Math(inline=True, data=['x']), Math(inline=True, data=['y']))
        xy_table.add_hline()
        for x in self.x_values:
            xy_table.add_row(Math(inline=True, data=[str(x)]), Math(inline=True, data=[str(h*x + k)]))

        # put the figures into minipages for alignment
        xy_page = MiniPage(width='3cm', data=xy_table)
        graph_page = MiniPage(width=r'0.3\textwidth', data=NoEscape(r'\includegraphics[width=\linewidth]{graphing_grid.png}'))

        self.num_quest -= 1
        return NoEscape(xy_page.dumps() + graph_page.dumps())


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

    def get_problem(self) -> Math:
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
                operand_str = ''
                for c in operand.get_latex():
                    operand_str += c.dumps() if isinstance(c, LatexObject) else c
                variable_side = [Command('frac', ['x', NoEscape(operand_str)])]

        self.num_quest -= 1
        if randint(0, 1) == 0:
            return Math(inline=True, data=variable_side + ['='] + constant_side)
        else:
            return Math(inline=True, data=constant_side + ['='] + variable_side)


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

    def generate_polynomial(self, degree: int=None, num_terms: int=None) -> SingleVariablePolynomial:
        """
        Randomly generates a polynomial, following the rules outlined in __init()__.

        :param degree: The degree of the polynomial. If None (default), it is randomly generated using drange.
        :param num_terms: If set, the generated polynomial will have exactly this many terms.
                          To use this, max_like must be 1.
                          If set to None (default), there will be no set number.
        """
        if degree is None:
            degree = randint(self.drange[0], self.drange[1])
        if degree < 0:
            raise ValueError("The degree must be non-negative")
        if num_terms is not None and num_terms < 1:
            raise ValueError("The number of terms must be positive")
        if num_terms and self.max_like > 1:
            raise ValueError("num_terms argument cannot be used unless max_like is 1")

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
        return SingleVariablePolynomial(choice(self.var), poly)

    def get_problem(self) -> Math:
        self.num_quest -= 1
        return Math(inline=True, data=self.generate_polynomial().get_latex())


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

    def get_problem(self) -> Math:
        polies = []
        for _ in range(2):
            degree = randint(max(self.drange[0], self.min_term_count - 1), self.drange[1])
            term_count = randint(self.min_term_count, degree + 1)
            polies.append(self.generate_polynomial(degree, term_count))
        self.num_quest -= 1
        return Math(inline=True, data=polies[0].get_latex() + ['-', Command('left(')] + polies[1].get_latex() + [Command('right)')])


class PolynomialMultiply(PolynomialSimplify):
    def __init__(self, num_quest: int, crange: tuple[int, int], drange: tuple[int, int], max_term_count: int=2, *var: str):
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
        :param max_term_count: Maximum number of terms allowed for the left multiplicand. The default is 2 (binomial).
        :param var: The possible variables to be used. Only one of them will be used per question.
                    The default is just x.
        """
        super().__init__(num_quest, crange, drange, *var, max_like=1)
        self.max_term_count = max_term_count

    def get_problem(self) -> Math:
        left_term_count = randint(1, self.max_term_count)
        left = self.generate_polynomial(randint(max(self.drange[0], left_term_count-1), self.drange[1]), left_term_count)
        right = self.generate_polynomial()
        self.num_quest -= 1
        return Math(inline=True, data=[Command('left(')] + left.get_latex() + [Command('right)'), Command('left(')] + right.get_latex() + [Command('right)')])


class PolynomialDivide(PolynomialSimplify):
    def __init__(self, num_quest: int, crange: tuple[int, int], drange: tuple[int, int], *var: str, no_constant=False):
        """
        Generates a problem where a polynomial is divided by a monomial.
        The problem is always written as a fraction.
        The quotient always simplify to a polynomial with integer coefficients.

        :param num_quest: The number of questions to be generated.
        :param crange: The range used for the coefficients, (begin, end) inclusive.
                       The generated coefficient will never be 0.
        :param drange: The range used for the degree of the dividend and the divisor, (begin, end) inclusive.
                       This range cannot contain a negative number.
        :param var: The possible variables to be used. Only one of them will be used per question.
                    The default is just x.
        :param no_constant: If True, the degree of the divisor will be at least 1.
                            False by default.
        """
        super().__init__(num_quest, crange, drange, *var, max_like=1)
        if no_constant and drange[1] < 1:
            raise ValueError("no_constant used yet drange doesn't contain 1 or higher")
        self.no_constant = no_constant

    def get_problem(self) -> Math:
        if self.no_constant:
            divisor = self.generate_polynomial(randint(max(1, self.drange[0]), self.drange[1]), 1)
        else:
            divisor = self.generate_polynomial(num_terms=1)
        quotient = self.generate_polynomial(randint(0, self.drange[1] - divisor.degree))
        dividend = divisor * quotient

        self.num_quest -= 1
        return Math(inline=True, data=PolynomialFraction(dividend, divisor).get_latex())
