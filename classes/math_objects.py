from decimal import Decimal
from abc import ABC, abstractmethod

from pylatex import Command
from pylatex.base_classes import LatexObject


class BaseMathClass(ABC):
    @abstractmethod
    def get_latex(self) -> list[LatexObject, str, int, float]:
        """
        Returns the LaTeX representation of the object in a list.
        """
        pass   # child class should implement this

    @abstractmethod
    def __eq__(self, other):
        """
        Check if this object is equal to another.
        """
        pass   # child class should implement this

    @abstractmethod
    def __neg__(self):
        """
        Returns the negative of this object.
        """
        pass   # child class should implement this


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

    def get_latex(self) -> list[Command | str]:
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

    def __neg__(self):
        return Fraction(self.num, self.denom, -self.sign, self.big)


class Number(BaseMathClass):
    def __init__(self, num: int | str):
        """
        An integer or a decimal number.

        :param num: The number to be represented. For a decimal number, enter it as a string.
        """
        if type(num) == str:
            num = Decimal(num)

        neg = False
        if num < 0:
            neg = True
            num = -num

        self.sign = -1 if neg else 1
        self.mag = num

    def get_latex(self) -> list[Command | str]:
        if self.sign == -1:
            return [Command("left("),
                    '-',
                    str(self.mag),
                    Command("right)")]
        else:
            return [str(self.mag)]

    def __eq__(self, other):
        if not isinstance(other, Number):
            return self.sign * self.mag == other
        else:
            return self.sign == other.sign and self.mag == other.mag

    def __neg__(self):
        if type(self.mag) == int:
            return Number(-self.mag)
        else:
            return Number(str(self.sign)[:-1] + str(self.mag))