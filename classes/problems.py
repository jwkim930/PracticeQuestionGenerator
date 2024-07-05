from random import randint

from pylatex import Math, Alignat, Command

from classes.math_objects import Fraction


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


class ProblemBase:
    def __init__(self, num_quest: int, vspace: str):
        """
        Initializes a problem.
        :param num_quest: The number of questions to be generated.
        :param vspace: The minimum vertical space to be added between problems.
        """
        self.num_quest = num_quest
        self.vspace = vspace

    def get_problem(self) -> Math | Alignat:
        """
        Returns a practice problem for LaTeX math/align environment.
        Decrements num_quest in the object.
        """
        pass


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
        f1 = generate_random_fraction(self.nrange, self.drange, self.no1)
        f2 = generate_random_fraction(self.nrange, self.drange, self.no1)
        if self.neg:
            fracs = [f1, f2]
            neg_i = randint(0, 1)
            other_i = 1 - neg_i
            fracs[neg_i].sign = -1
            if randint(0, 1) == 1:
                fracs[other_i].sign = -1

        self.num_quest -= 1
        return Math(inline=True, data=f1.get_latex() + [self.operand] + f2.get_latex() + ["="])


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
