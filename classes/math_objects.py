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


type NumberArgument = int | float | Decimal | str | Number


class Number(BaseMathEntity):
    def __init__(self, num: NumberArgument, wrap=False):
        """
        An integer or a decimal number.

        :param num: The number to be represented.
                    For a decimal number, you can enter it as a string to avoid float rounding error.
        :param wrap: If True, the number will be surrounded by parentheses when it's negative.
                     Has no effect if the sign is positive.
        """
        if type(num) == Number:
            super().__init__(num.sign, wrap)
            self.mag: Decimal = num.mag
        else:
            num = Decimal(num)
            sign = 1
            if num < 0:
                sign = -1
                num = -num

            super().__init__(sign, wrap)
            self.mag: Decimal = num

    def get_signed(self) -> Decimal:
        """
        Returns the value of the number with the sign attached.
        """
        return self.sign * self.mag

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

    def is_int(self) -> bool:
        """
        Returns True if this number can be converted to integer with no loss.
        """
        return self.mag == int(self.mag)

    def __str__(self):
        return str(self.get_signed())

    def __eq__(self, other):
        if not isinstance(other, Number):
            return self.get_signed() == other
        else:
            return self.sign == other.sign and self.mag == other.mag

    def __neg__(self):
        return Number(-self.get_signed())

    def __mul__(self, other):
        if type(other) in (int, float, Decimal):
            return Number(self.get_signed() * other)
        elif type(other) is Number:
            return Number(self.get_signed() * other.get_signed())
        elif isinstance(other, Fraction):
            return other * self
        else:
            raise ValueError(f"Multiplication between {type(self).__name__} and {type(other).__name__} is undefined")

    def __int__(self):
        return int(self.get_signed())

    def __add__(self, other):
        if isinstance(other, Number):
            if self.sign == other.sign:
                return Number(self.sign * (self.mag + other.mag))
            else:
                if self.mag >= other.mag:
                    return Number(self.sign * (self.mag - other.mag))
                else:
                    return Number(other.sign * (other.mag - self.mag))
        else:
            return self + Number(other)


class Term(BaseMathEntity):
    def __init__(self, variable: str, coefficient: NumberArgument, exponent: int):
        """
        A term of a polynomial with a numerical coefficient with one variable.
        The stored coefficient will always be non-negative.

        :param variable: The variable to be used.
        :param coefficient: The coefficient of the polynomial. Use string for any non-integer values.
        :param exponent: The exponent of this term.
        """
        self.coefficient = Number(coefficient)
        super().__init__(self.coefficient.sign, False)
        self.coefficient.sign = 1
        self.exponent = exponent
        self.variable = variable

    def get_signed_coefficient(self) -> Number:
        return self.coefficient * self.sign

    def get_latex(self) -> list[NoEscape]:
        coe = ''
        if self.coefficient != 1 or self.exponent == 0:
            coe = self.get_signed_coefficient().dumps()
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
            return self.get_signed_coefficient() == other

    def __neg__(self):
        if type(self.coefficient.mag) is int:
            return Term(self.variable, -self.get_signed_coefficient(), self.exponent)


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
        :param mix: If True, the terms will be shuffled upon initialization.
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


class MultiVariableTerm(BaseMathEntity):
    def __init__(self, coefficient: NumberArgument, *variables: tuple[str, NumberArgument]):
        """
        A term of a polynomial with a numerical coefficient and variables raised to a numerical power.
        The stored coefficient will become positive if the input coefficient was negative.

        :param coefficient: The coefficient of the polynomial. Use string for any non-integer values.
        :param variables: The variables raised to a numerical power. The first element should be the variable
                          and the second should be the exponent. Use string for any non-integer exponent.
        """
        self.coefficient = Number(coefficient)
        super().__init__(self.coefficient.sign, False)
        self.coefficient.sign = 1
        self.variables = [(var[0], Number(var[1])) for var in variables]

    def __eq__(self, other):
        if not isinstance(other, MultiVariableTerm):
            return False
        else:
            if self.sign != other.sign:
                return False
            if self.coefficient != other.coefficient:
                return False
            for var in self.variables:
                if var not in other.variables:
                    return False
            for var in other.variables:
                if var not in self.variables:
                    return False

            return True

    def __neg__(self):
        return MultiVariableTerm(self.coefficient.mag * -self.sign, *self.variables)

    def __mul__(self, other):
        if isinstance(other, MultiVariableTerm):
            variables = self.variables.copy()
            for var in other.variables:
                vs = [v[0] for v in variables]
                if var[0] not in vs:
                    variables.append(var)
                else:
                    i = vs.index(var[0])
                    variables[i] = (variables[i][0], variables[i][1] + var[1])
            return MultiVariableTerm(self.coefficient * other.coefficient * self.sign * other.sign,
                                     *variables)
        elif isinstance(other, MultiVariablePolynomial):
            return other * self
        else:
            raise ValueError(f"Multiplication between MultiVariableTerm and {type(other)} is undefined")

    def get_latex(self) -> list[NoEscape]:
        result = []
        if self.coefficient != 1 or len(self.variables) == 0:
            result.append((self.coefficient * self.sign).dumps())
        elif self.sign == -1:
            result.append(NoEscape("-"))
        for var in self.variables:
            if var[1] != 0:
                exp = ""
                if var[1] != 1:
                    exp = var[1].dumps()
                result.append(NoEscape(f"{var[0]}^{{{exp}}}"))
            elif self.coefficient in (-1, 1):
                result.append(NoEscape(1))

        return result


class MultiVariablePolynomial(BaseMathClass):
    def __init__(self, terms: list[MultiVariableTerm], mix=False):
        """
        A polynomial with integer coefficients and one variable.
        The input data can be a list of dictionaries,
        where each dictionary represents a term in the polynomial.

        :param terms: The terms of the polynomial.
        :param mix: If True, the terms will be shuffled upon initialization.
        """
        self.terms = terms
        if mix:
            shuffle(self.terms)

    def __mul__(self, other):
        if isinstance(other, MultiVariableTerm):
            terms = self.terms.copy()
            for i in range(len(terms)):
                terms[i] = terms[i] * other
            return MultiVariablePolynomial(terms)
        elif isinstance(other, MultiVariablePolynomial):
            terms = []
            for mine in self.terms:
                for others in other.terms:
                    terms.append(mine * others)
            return MultiVariablePolynomial(terms)
        else:
            raise ValueError(f"Multiplication between MultiVariablePolynomial and {type(other)} is undefined")

    def get_latex(self) -> list[NoEscape]:
        # first iteration
        result = [self.terms[0].dumps()]

        # later iterations
        for term in self.terms[1:]:
            if term.sign == 1:
                result.append(NoEscape("+"))
            result.append(term.dumps())

        return result


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
