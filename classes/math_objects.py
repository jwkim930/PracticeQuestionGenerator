from numpy.f2py.symbolic import as_numer_denom
from pylatex import Command
from pylatex.base_classes import LatexObject


class BaseMathClass:
    def get_latex(self) -> list[LatexObject, str, int, float]:
        """
        Returns the LaTeX representation of the object in a list.
        """
        pass


class Fraction(BaseMathClass):
    def __init__(self, num: int, denom: int, sign=1, big=True):
        """
        A fraction with integer numerator/denominator.
        :param num: Numerator of the fraction.
        :param denom: Denominator of the fraction.
        :param sign: The sign of the fraction. 1 if positive, -1 if negative.
        :param big: If True, get_command() will return a fraction in display mode (bigger text).
        """
        if sign != 1 and sign != -1:
            raise ValueError("sign must be 1 or -1")
        self.num = num
        self.denom = denom
        self.sign = sign
        self.big = big

    def get_latex(self) -> list:
        if self.sign == -1:
            return [Command("left("),
                    '-',
                    Command("dfrac" if self.big else "frac", [self.num, self.denom]),
                    Command("right)")]
        else:
            return [Command("dfrac" if self.big else "frac", [self.num, self.denom])]

    def __eq__(self, other) -> bool:
        """
        Checks if this fraction is equal to another.
        For example, 4/5 is equal to 4/5, but 4/5 is not equal to 8/9 or 8/10.
        This ignores the attribute big.

        :return: True if the two are the same fraction, false otherwise.
        """
        if not isinstance(other, Fraction):
            return False
        return self.num == other.num and self.denom == other.denom and self.sign == other.sign