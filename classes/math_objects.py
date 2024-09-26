from decimal import Decimal
from abc import ABC, abstractmethod

from pylatex import Command, NoEscape
from pylatex.base_classes import LatexObject


class BaseMathClass(ABC):
    @abstractmethod
    def get_latex(self) -> list[LatexObject, str, int, float]:
        """
        Returns the LaTeX representation of the object in a list.
        """
        pass   # child class should implement this


class BaseMathEntity(BaseMathClass, ABC):
    @abstractmethod
    def __eq__(self, other):
        """
        Check if this object is equal to another.
        """
        pass  # child class should implement this

    @abstractmethod
    def __neg__(self):
        """
        Returns the negative of this object.
        """
        pass  # child class should implement this


class Fraction(BaseMathEntity):
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


class Number(BaseMathEntity):
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


class SingleVariablePolynomial(BaseMathClass):
    def __init__(self, variable: str, data: list[dict[str, int]]):
        """
        A polynomial with integer coefficients and one variable.
        The input data should be a list of dictionaries,
        where each dictionary represents a term in the polynomial.

        Each dictionary must be in the following structure:
        {'coefficient': coefficient value, int,
         'exponent': exponent of the variable, int}

        For example, the term -4x^3 is represented by the dictionary:
        {'coefficient': -4
         'exponent': 3
        }

        For a constant term, the exponent must be 0 and the coefficient must record the value of the constant.
        The coefficient should be 0 if and only if the number is the constant 0.
        In such a case, the dictionary must be:

        {'coefficient': 0,
         'exponent': 0,}

        Each term is assumed to be added. That is, the polynomial 4x - 2 is
        uniquely identified by the data:

        [{'coefficient': 4,
          'exponent': 1},
         {'coefficient': -2,
          'exponent': 0}
        ]

        This object also records the degree of the polynomial.

        :param variable: The variable to be used.
        :param data: The list of data as explained above.
        """
        self.variable = variable
        self.data = data
        # check data integrity
        for term in data:
            c = term['coefficient']
            e = term['exponent']

            if e < 0:
                raise ValueError(f"The term {term} has a negative exponent")
            if c == 0 and e != 0:
                raise ValueError(f"The term {term} has zero coefficient but nonzero exponent")

        self.degree = max([d['exponent'] for d in data])

    def __len__(self):
        """
        Returns the number of terms in the polynomial.
        """
        return len(self.data)

    def get_latex(self) -> list[NoEscape]:
        def term_str(v: str, c: int, e: int) -> str:
            if e == 0:
                return str(c)
            elif e == 1:
                if c == 1:
                    return v
                elif c == -1:
                    return f"-{v}"
                else:
                    return f"{c}{v}"
            elif c == 1:
                return f"{v}^{{{e}}}"
            elif c == -1:
                return f"-{v}^{{{e}}}"
            else:
                return f"{c}{v}^{{{e}}}"

        result = []

        # first iteration
        term = self.data[0]
        coe = term['coefficient']
        exp = term['exponent']
        result.append(NoEscape(term_str(self.variable, coe, exp)))

        # later iterations
        for i in range(1, len(self.data)):
            term = self.data[i]
            coe = term['coefficient']
            exp = term['exponent']
            opr = '' if coe < 0 else '+'
            result.append(NoEscape(opr + term_str(self.variable, coe, exp)))

        return result
