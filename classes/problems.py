from random import randint, choice

from pylatex import Math, Alignat, Command
from pylatex.utils import NoEscape

from classes.math_objects import Fraction


class ProblemBase:
    def __init__(self, num_quest: int, vspace: str):
        """
        Initializes a problem.
        :param num_quest: The number of questions to be generated.
        :param vspace: The minimum vertical space to be added below each problem.
        """
        self.num_quest = num_quest
        self.vspace = vspace

    def get_problem(self) -> Math | Alignat | str | NoEscape:
        """
        Returns a practice problem for LaTeX math/align environment.
        Decrements num_quest in the object.
        """
        pass   # child class should implement this


class FractionBinaryOperation(ProblemBase):
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
        super().__init__(num_quest,
                         "0cm")
        self.operand = operand
        self.nrange = nrange
        self.drange = drange
        self.no1 = no1
        self.neg = neg

    def get_problem(self) -> Math:
        f1 = self.generate_random_fraction(self.nrange, self.drange, self.no1)
        f2 = self.generate_random_fraction(self.nrange, self.drange, self.no1)
        if self.neg:
            fracs = [f1, f2]
            neg_i = randint(0, 1)
            other_i = 1 - neg_i
            fracs[neg_i].sign = -1
            if randint(0, 1) == 1:
                fracs[other_i].sign = -1

        self.num_quest -= 1
        return Math(inline=True, data=f1.get_latex() + [self.operand] + f2.get_latex() + ["="])

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


class GraphingProblem(ProblemBase):
    def __init__(self, num_quest: int):
        """
        Initializes a graphing problem.
        It requires graphing_grid.png in the document_output folder.
        """
        super().__init__(num_quest, '0cm')

    def get_random_function(self) -> str:
        """
        Returns a function with randomized parameters.
        """
        pass   # child class should implement this

    def get_problem(self):
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
                self.operand_candidates.append(b**2 / 100**o)

        if no_duplicate and len(self.operand_candidates) < num_quest:
            raise ValueError("The given ranges do not generate enough questions")
