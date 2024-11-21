from decimal import Decimal
from abc import ABC, abstractmethod
from random import shuffle

from pylatex import Command, NoEscape
from pylatex.base_classes import LatexObject


class BaseMathClass(ABC):
    @abstractmethod
    def get_latex(self) -> list[LatexObject, NoEscape, int]:
        """
        Returns the LaTeX representation of the object in a list.
        """
        pass   # child class should implement this

    def dumps(self) -> NoEscape:
        """
        Returns the string representation of the object in NoEscape.
        """
        result = NoEscape()
        for l in self.get_latex():
            if isinstance(l, LatexObject):
                result += NoEscape(l.dumps())
            else:
                result += NoEscape(l)
        return result


class TextWrapper(BaseMathClass):
    def __init__(self, texts: list[str | NoEscape]=None):
        """
        A class to wrap raw texts into BaseMathClass.
        get_latex() returns the texts in NoEscape.
        """
        if texts is None:
            self.texts = []
        else:
            self.texts = [NoEscape(t) for t in texts]

    def get_latex(self) -> list[NoEscape]:
        return self.texts

    def append(self, text: str | NoEscape):
        """
        Adds a text.
        """
        self.texts.append(NoEscape(text))


class BaseMathEntity(BaseMathClass, ABC):
    """
    This class must include the attribute sign and wrap,
    as well as implementing __eq__() and __neg__.
    """
    def __init__(self, sign: int, wrap: bool):
        if sign != 1 and sign != -1:
            raise ValueError("sign must be 1 or -1")
        self.sign = sign
        self.wrap = wrap

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
    def __init__(self, num: int, denom: int, sign=1, big=True, wrap=False):
        """
        A fraction with integer numerator/denominator.
        Multiplying it by other entity leaves the fraction unsimplified.
        Multiplying a big fraction and small fraction results in a big fraction.

        :param num: Numerator of the fraction.
        :param denom: Denominator of the fraction. Cannot be zero.
        :param sign: The sign of the fraction. 1 if positive, -1 if negative.
        :param big: If True, get_command() will return a fraction in display mode (bigger text).
        :param wrap: If True, the fraction will be surrounded by parentheses when it's negative.
                     Has no effect if the sign is positive.
        """
        super().__init__(sign, wrap)
        if denom == 0:
            raise ValueError("The denominator cannot be 0")
        self.num = num
        self.denom = denom
        self.big = big

    def get_latex(self) -> list[Command | NoEscape]:
        if self.sign == -1:
            if self.wrap:
                return [Command("left("),
                        NoEscape('-'),
                        Command("dfrac" if self.big else "frac", [self.num, self.denom]),
                        Command("right)")]
            else:
                return [NoEscape('-') if self.sign == -1 else NoEscape(),
                        Command('dfrac' if self.big else 'frac', [self.num, self.denom])]
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

    def __mul__(self, other):
        if isinstance(other, int):
            return Fraction(self.num * other, self.denom, self.sign, self.big)
        elif isinstance(other, Number) and other.is_int():
            return Fraction(int(self.num * other.mag), self.denom, self.sign * other.sign, self.big)
        elif isinstance(other, Fraction):
            return Fraction(self.num * other.num, self.denom * other.denom, self.sign * other.sign, self.big or other.big)
        else:
            raise ValueError(f"Multiplication between {type(self).__name__} and {type(other).__name__} is undefined")

class Number(BaseMathEntity):
    def __init__(self, num: int | str, wrap=False):
        """
        An integer or a decimal number.

        :param num: The number to be represented. For a decimal number, enter it as a string.
        :param wrap: If True, the number will be surrounded by parentheses when it's negative.
                     Has no effect if the sign is positive.
        """
        if type(num) == str:
            num = Decimal(num)

        neg = False
        if num < 0:
            neg = True
            num = -num

        super().__init__(-1 if neg else 1, wrap)
        self.mag = num

    def get_latex(self) -> list[Command | NoEscape]:
        if self.sign == -1:
            if self.wrap:
                return [Command("left("),
                        NoEscape('-'),
                        NoEscape(self.mag),
                        Command("right)")]
            else:
                return [NoEscape('-'), NoEscape(self.mag)]
        else:
            return [NoEscape(self.mag)]

    def is_int(self):
        """
        Returns True if this number can be converted to integer with no loss.
        """
        return int(self.mag) == self.mag

    def __eq__(self, other):
        if not isinstance(other, Number):
            return self.sign * self.mag == other
        else:
            return self.sign == other.sign and self.mag == other.mag

    def __neg__(self):
        if type(self.mag) == int:
            return Number(self.mag * -self.sign)
        else:
            return Number(str(-self.sign)[:-1] + str(self.mag))

    def __mul__(self, other):
        if isinstance(other, int):
            if type(self.mag) is int:
                return Number(self.mag * other * self.sign)
            else:
                return Number(str(self.mag * other * self.sign))
        elif isinstance(other, Decimal):
            return Number(str(other * self.mag * self.sign))
        elif isinstance(other, Number):
            if type(self.mag) is int and type(other.mag) is int:
                return Number(self.mag * self.sign * other.mag * other.sign)
            else:
                return Number(str(self.mag * self.sign * other.mag * other.sign))
        elif isinstance(other, Fraction):
            return other * self
        else:
            raise ValueError(f"Multiplication between {type(self).__name__} and {type(other).__name__} is undefined")

    def __int__(self):
        return self.sign * int(self.mag)


class Term(BaseMathEntity):
    def __init__(self, variable: str, coefficient: str | int, exponent: int):
        """
        A term of a polynomial with a numerical coefficient.
        The stored coefficient will become positive if the input coefficient was negative.

        :param variable: The variable to be used.
        :param coefficient: The coefficient of the polynomial. Use string for any non-integer values.
        :param exponent: The exponent of this term.
        """
        coefficient = Number(coefficient)
        super().__init__(coefficient.sign, False)
        self.coefficient = coefficient if coefficient.sign == 1 else -coefficient
        self.exponent = exponent
        self.variable = variable

    def get_latex(self) -> list[NoEscape]:
        coe = ''
        if self.coefficient != 1 or self.exponent == 0:
            coe = (self.coefficient * self.sign).dumps()
        elif self.sign == -1:
            coe = '-'

        return [NoEscape(coe),
                NoEscape(self.variable if self.exponent != 0 else ''),
                NoEscape('^' if self.exponent not in [0, 1] else ''),
                NoEscape(f"{{{self.exponent}}}" if self.exponent not in [0, 1] else '')]

    def __eq__(self, other):
        if isinstance(other, Term):
            return self.variable == other.variable and self.coefficient == other.coefficient and self.sign == other.sign
        elif self.exponent == 0:
            return self.coefficient * self.sign == other

    def __neg__(self):
        if type(self.coefficient.mag) is int:
            return Term(self.variable, self.coefficient.mag * -self.sign, self.exponent)


class SingleVariablePolynomial(BaseMathClass):
    def __init__(self, variable: str, data: list[dict[str, int] | Term], mix=False):
        """
        A polynomial with integer coefficients and one variable.
        The input data can be a list of dictionaries,
        where each dictionary represents a term in the polynomial.

        Each dictionary must be in the following structure:
        {'coefficient': coefficient value, int,
         'exponent': exponent of the variable, int}

        For example, the term -4x^3 is represented by the dictionary:
        {'coefficient': -4
         'exponent': 3
        }

        For a constant term, the exponent must be 0 and the coefficient must record the value of the constant.

        Each term is assumed to be added. That is, the polynomial 4x - 2 is
        uniquely identified by the data:

        [{'coefficient': 4,
          'exponent': 1},
         {'coefficient': -2,
          'exponent': 0}
        ]

        This object also records the degree of the polynomial.

        :param variable: The variable to be used.
        :param data: The list of data as explained above, or as a Term object.
        :param mix: If True, the terms will be shuffled.
        """
        self.variable = variable
        self.data = []
        self.degree = float('-inf')
        for t in data:
            if isinstance(t, Term):
                self.data.append(t)
            else:
                self.data.append(Term(variable, t['coefficient'], t['exponent']))
            self.degree = max(self.degree, self.data[-1].exponent)
        if mix:
            shuffle(self.data)

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
        coe = int(term.coefficient) * term.sign
        exp = term.exponent
        result.append(NoEscape(term_str(self.variable, coe, exp)))

        # later iterations
        for i in range(1, len(self.data)):
            term = self.data[i]
            coe = int(term.coefficient) * term.sign
            exp = term.exponent
            opr = '' if coe < 0 else '+'
            result.append(NoEscape(opr + term_str(self.variable, coe, exp)))

        return result

    def __mul__(self, other):
        if not isinstance(other, SingleVariablePolynomial):
            raise ValueError(f"multiplication between {type(self).__name__} and {type(other).__name__} is undefined")
        if self.variable != other.variable:
            raise ValueError(f"The two polynomials use different variables: {self.variable} and {other.variable}")
        result = []
        for sterm in self.data:
            for oterm in other.data:
                result.append(Term(self.variable, int(sterm.coefficient) * int(oterm.coefficient), sterm.exponent * oterm.exponent))
        return SingleVariablePolynomial(self.variable, result)


class PolynomialFraction(BaseMathClass):
    def __init__(self, num: SingleVariablePolynomial, denom: SingleVariablePolynomial, sign=1, wrap=False):
        """
        A fraction of single-variable polynomials.
        The numerator and the denominator must use the same variable.

        :param num: The numerator polynomial.
        :param denom: The denominator polynomial.
        :param sign: The sign of the fraction. 1 means positive, -1 means negative.
        :param wrap: If True, the fraction will be enclosed in parentheses.
        """
        if num.variable != denom.variable:
            raise ValueError(f"The numerator uses variable {num.variable} whereas the denominator uses {denom.variable}")
        if sign not in (-1, 1):
            raise ValueError("The sign must be either 1 or -1")
        self.num = num
        self.denom = denom
        self.variable = num.variable
        self.sign = sign
        self.wrap = wrap

    def get_latex(self) -> list[Command | NoEscape]:
        num_text = self.num.dumps()
        denom_text = self.denom.dumps()
        result = [Command('dfrac', [num_text, denom_text])]
        if self.sign == -1:
            result.insert(0, NoEscape('-'))
        if self.wrap:
            result.insert(0, Command('left('))
            result.append(Command('right)'))
        return result


class UnsafePolynomial(BaseMathClass):
    def __init__(self, *terms: str | BaseMathClass, mix=False):
        """
        A polynomial with arbitrary terms.

        Internally, each term will be assigned a sign for concatenation.
        If the term is given as a string, the term is negative if it has '-' in the beginning and positive otherwise.
        If the term is given as a BaseMathEntity, its sign will be used.
        If the term is other BaseMathClass, it is always positive.

        Each term will be represented as a dictionary with keys 'sign' and 'value'.
        'sign' will be either 1 or -1. 'value' will be negative if the original term was negative.

        :param terms: The terms for the polynomial.
        :param mix: If True, the order of the terms will be shuffled.
        """
        self.terms = []
        for term in terms:
            t = {}
            if isinstance(term, str):
                if term[0] == '-':
                    t['sign'] = -1
                else:
                    t['sign'] = 1
                t['value'] = NoEscape(term)
            elif isinstance(term, BaseMathEntity):
                t['sign'] = term.sign
                t['value'] = term.dumps()
            elif isinstance(term, BaseMathClass):
                t['sign'] = 1
                t['value'] = term.dumps()
            else:
                raise ValueError(f"type {type(term).__name__} is not allowed for a term")
            self.terms.append(t)
        if mix:
            shuffle(self.terms)

    def get_latex(self) -> list[NoEscape]:
        result = [self.terms[0]['value']]
        for t in self.terms[1:]:
            if t['sign'] == 1:
                result.append(NoEscape('+'))
            result.append(t['value'])

        return result
