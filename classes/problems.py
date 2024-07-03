from random import randint

from pylatex import Math, Alignat

from classes.math_objects import Fraction


class ProblemBase:
    def __init__(self, name: str, instruction: str, num_quest: int, num_col: int, vspace: str):
        """
        Initializes a problem.
        :param name: The name of the problem to be used for the document title.
        :param instruction: The instruction for the student. Displayed at the top of the document.
        :param num_quest: The number of questions to be generated.
        :param num_col: The number of columns to be used.
        :param vspace: The minimum vertical space to be added between problems.
        """
        self.name = name
        self.instruction = instruction
        self.num_quest = num_quest
        self.num_col = num_col
        self.vspace = vspace

    def get_problem(self) -> Math | Alignat:
        """
        Returns a practice problem for LaTeX math/align environment.
        """
        pass


class FractionAddition(ProblemBase):
    def __init__(self, num_quest: int, nrange: tuple[int, int], drange: tuple[int, int]):
        """
        Initializes a fraction addition problem.
        :param num_quest: The number of questions to be generated.
        :param nrange: The range for numerator, (begin, end) inclusive.
        :param drange: The range for denominator, (begin, end) inclusive.
        """
        super().__init__("Fraction Addition",
                         "",
                         num_quest,
                         3,
                         "0cm")
        self.nrange = nrange
        self.drange = drange

    def get_problem(self) -> Math:
        f1 = Fraction(randint(self.nrange[0], self.nrange[1]), randint(self.drange[0], self.drange[1]))
        f2 = Fraction(randint(self.nrange[0], self.nrange[1]), randint(self.drange[0], self.drange[1]))

        return Math(inline=True, data=[f1.get_command(), "+", f2.get_command(), "="])
