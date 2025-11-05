from decimal import Decimal
from abc import ABC, abstractmethod
from random import shuffle
from typing import Self
import math

from pylatex import Command, NoEscape
from pylatex.base_classes import LatexObject
import numpy as np


class BaseMathClass(ABC):
    @abstractmethod
    def get_latex(self) -> list[LatexObject | NoEscape]:
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
    def __init__(self, texts: list[str | NoEscape] | None = None):
        """
        A class to wrap raw texts into BaseMathClass.
        get_latex() returns the texts in NoEscape.
        """
        if texts is None:
            self.texts = []
        else:
            self.texts = [NoEscape(t) for t in texts]

    def get_latex(self) -> list[NoEscape]:
        return self.texts.copy()

    def append(self, text: str | NoEscape):
        """
        Adds a text.
        """
        self.texts.append(NoEscape(text))


class BaseMathEntity(BaseMathClass, ABC):
    """
    This class must include the attribute sign and wrap,
    as well as implementing __eq__() and __neg__().
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
        :param big: If True, get_latex() will return a fraction in display mode (bigger text).
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
                return [NoEscape('-'),
                        Command('dfrac' if self.big else 'frac', [self.num, self.denom])]
        else:
            return [Command("dfrac" if self.big else "frac", [self.num, self.denom])]

    def __eq__(self, other) -> bool:
        """
        Checks if this fraction is equal to another.
        Equivalent fractions are considered not equal.
        For example, 4/5 is equal to 4/5, but 4/5 is not equal to 8/9 or 8/10.
        Also, -(-1/2) is not equal to 1/2.
        This ignores the attribute big.

        :return: True if the two are the same fraction, false otherwise.
        """
        if not isinstance(other, Fraction):
            return False
        return self.num == other.num and self.denom == other.denom and self.sign == other.sign

    def __neg__(self):
        return Fraction(self.num, self.denom, -self.sign, self.big, self.wrap)

    def __mul__(self, other):
        if isinstance(other, Number):
            if other.is_int():
                return Fraction(int(self.num * other.mag), self.denom, self.sign * other.sign, self.big, self.wrap)
            else:
                return TypeError("Only an integer value can be multiplied to a Fraction, the value was " + str(other))
        elif isinstance(other, Fraction):
            return Fraction(self.num * other.num, self.denom * other.denom, self.sign * other.sign, self.big or other.big, self.wrap or other.wrap)
        else:
            return NotImplemented

    def simplified(self) -> "Fraction":
        """
        Returns the simplified fraction as a new instance.

        :returns: The simplified fraction.
        """
        sign = (self.sign * self.num * self.denom) // abs(self.num * self.denom)
        d = math.gcd(self.num, self.denom)
        return Fraction(abs(self.num // d), abs(self.denom // d), sign=sign, big=self.big, wrap=self.wrap)

type NumberArgument = int | float | Decimal | str | Number


class Number(BaseMathEntity):
    def __init__(self, num: NumberArgument, wrap=False):
        """
        An integer or a decimal number.
        Passing in a Number gives a new instance with the same value,
        where wrap is inherited unless True is passed in.

        :param num: The number to be represented.
                    For a decimal number, you can enter it as a string to avoid float rounding error.
        :param wrap: If True, the number will be surrounded by parentheses when it's negative.
                     Has no effect if the sign is positive.
                     If num was a Number, True overrides wrap and False inherits it from num.
        """
        if isinstance(num, Number):
            super().__init__(num.sign, num.wrap or wrap)
            self.mag: Decimal = num.mag
        else:
            try:
                num = Decimal(num)
            except Exception:
                raise TypeError(f"{num} could not be converted to Decimal")
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

    def __repr__(self):
        return f"Number('{self.get_signed()}', wrap={self.wrap})"

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
            return NotImplemented

    __rmul__ = __mul__

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
            try:
                return self + Number(other)
            except (ValueError, TypeError):
                return NotImplemented

    __radd__ = __add__

    def __sub__(self, other):
        if isinstance(other, Number):
            return Number(self + -other)
        else:
            try:
                return self - Number(other)
            except (ValueError, TypeError):
                return NotImplemented

    def __lt__(self, other):
        if isinstance(other, Number):
            return (self - other).sign == -1
        else:
            try:
                return self < Number(other)
            except (ValueError, TypeError):
                return NotImplemented

    def __le__(self, other):
        return self < other or self == other

    def __gt__(self, other):
        return -self < -other

    def __ge__(self, other):
        return self > other or self == other

    def __hash__(self):
        return hash((self.sign, self.mag, self.wrap))

    def __abs__(self) -> "Number":
        if self < 0:
            return -self
        else:
            return Number(self)

    def __truediv__(self, other) -> "Number":
        if isinstance(other, (int, float, Decimal)):
            if other == 0:
                raise ZeroDivisionError("Number divided by 0")
            return Number(self.get_signed() * other)
        elif isinstance(other, Number):
            if other == 0:
                raise ZeroDivisionError("Number divided by 0")
            return Number(self.get_signed() / other.get_signed())
        else:
            return NotImplemented



class MultiVariableTerm(BaseMathEntity):
    def __init__(self, coefficient: NumberArgument, *variables: tuple[str, NumberArgument], hide_zero_exponent=False):
        """
        A term of a polynomial with a numerical coefficient and variables raised to a numerical power.
        The stored coefficient will become positive if the input coefficient was negative.

        :param coefficient: The coefficient of the polynomial.
                            You may use string to avoid floating point conversion error.
        :param variables: The variables raised to a numerical power. The first element should be the variable
                          and the second should be the exponent.
                          You may string for exponent to avoid floating point conversion error.
        :param hide_zero_exponent: If True, the variables with exponent 0 are not shown in get_latex().
                                   When two terms are multiplied, this becomes False unless both were True originally.
        """
        self.coefficient = Number(coefficient)
        super().__init__(self.coefficient.sign, False)
        self.coefficient.sign = 1
        self.variables = [(var[0], Number(var[1])) for var in variables]
        self.hide_zero_exponent = hide_zero_exponent

    def get_signed_coefficient(self) -> Number:
        return self.sign * self.coefficient

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
        return MultiVariableTerm(-self.get_signed_coefficient(), *self.variables, hide_zero_exponent=self.hide_zero_exponent)

    def __mul__(self, other):
        if isinstance(other, MultiVariableTerm):
            variables = self.variables.copy()
            all_variables = [v[0] for v in variables]
            for var, exp in other.variables:
                if var not in all_variables:
                    variables.append((var, Number(exp)))
                    all_variables.append(var)
                else:
                    i = all_variables.index(var)
                    variables[i] = (var, variables[i][1] + exp)
            return MultiVariableTerm(self.get_signed_coefficient() * other.get_signed_coefficient(),
                                     *variables,
                                     hide_zero_exponent=self.hide_zero_exponent and other.hide_zero_exponent)
        elif isinstance(other, MultiVariablePolynomial):
            return other * self
        elif type(other) in (int, float, Decimal, Number):
            return MultiVariableTerm(self.get_signed_coefficient() * other,
                                     *self.variables,
                                     hide_zero_exponent=self.hide_zero_exponent)
        else:
            return NotImplemented

    __rmul__ = __mul__

    def __len__(self) -> int:
        return len(self.variables)

    def get_latex(self) -> list[NoEscape]:
        result = []
        if self.get_signed_coefficient() == -1:
            result.append(NoEscape("-"))
        elif self.coefficient != 1:
            result.append(self.get_signed_coefficient().dumps())
        no_variable = True
        for var, exp in self.variables:
            if exp == 0 and self.hide_zero_exponent:
                continue
            elif exp != 1:
                result.append(NoEscape(f"{var}^{{{exp.dumps()}}}"))
            else:
                result.append(NoEscape(var))
            no_variable = False
        if no_variable and self.coefficient == 1:
            # coefficient is +-1 and no variables to display
            result.append(NoEscape("1"))

        return result

    def remove_ones(self) -> Self:
        """
        Removes the variables with exponent 0.

        :returns: Self for chained method calls
        """
        i = 0
        while i < len(self.variables):
            if self.variables[i][1] == 0:
                self.variables.pop(i)
                i -= 1  # adjust index
            i += 1
        return self


class MultiVariablePolynomial(BaseMathClass):
    def __init__(self, terms: list[MultiVariableTerm], mix=False, wrap=False):
        """
        A polynomial with numerical coefficients and multiple variables raised to numerical powers.

        :param terms: The terms of the polynomial.
        :param mix: If True, the terms will be shuffled upon initialization.
        :param wrap: If True, get_latex will return the polynomial enclosed in parentheses.
        """
        self.terms = terms.copy()
        if mix:
            self.mix()
        self.wrap = wrap

    def __add__(self, other) -> Self:
        if isinstance(other, MultiVariableTerm):
            return MultiVariablePolynomial(self.terms + [other], wrap=self.wrap)
        if isinstance(other, (int, float, Decimal, Number)):
            return MultiVariablePolynomial(self.terms + [MultiVariableTerm(other)], wrap=self.wrap)
        if isinstance(other, MultiVariablePolynomial):
            return MultiVariablePolynomial(self.terms + other.terms, wrap=self.wrap or other.wrap)
        return NotImplemented

    __radd__ = __add__

    def __mul__(self, other) -> Self:
        if isinstance(other, MultiVariableTerm) or type(other) in (int, float, Decimal, Number):
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
            return NotImplemented

    __rmul__ = __mul__

    def get_latex(self) -> list[Command | NoEscape]:
        if not self.terms:
            return []
        result = []
        if self.wrap:
            result.append(Command("left("))

        # first iteration
        result.append(self.terms[0].dumps())

        # later iterations
        for term in self.terms[1:]:
            if term.sign == 1:
                result.append(NoEscape("+"))
            result.append(term.dumps())

        if self.wrap:
            result.append(Command("right)"))

        return result

    def append(self, term: MultiVariableTerm):
        """
        Adds the term to the end of the polynomial.

        :param term: The term to be added.
        """
        self.terms.append(term)

    def pop(self, index: int) -> MultiVariableTerm:
        """
        Removes the term at the index.

        :return: The removed term.
        """
        return self.terms.pop(index)

    def __len__(self) -> int:
        return len(self.terms)

    def __getitem__(self, index) -> MultiVariableTerm:
        return self.terms[index]

    def __setitem__(self, index, value):
        if not isinstance(value, MultiVariableTerm):
            raise TypeError(f"{type(value).__name__} cannot be stored in {type(self).__name__}")
        self.terms[index] = value

    def mix(self) -> Self:
        """
        Mixes the order of the terms in place.

        :return: self for chained method calls
        """
        shuffle(self.terms)
        return self

    def remove_zeros(self) -> Self:
        """
        Removes the terms with 0 coefficient.

        :returns: self for chained method calls
        """
        i = 0
        while i < len(self):
            term = self.terms[i]
            if term.coefficient == 0:
                self.terms.pop(i)
                i -= 1   # adjust index
            i += 1
        return self

    def simplify(self) -> Self:
        """
        Simplifies the polynomial in place, combining like terms.

        :returns: self for chained method calls
        """
        self.remove_zeros()
        # reorder variables within terms in alphabetical ordering
        for term in self.terms:
            term.variables.sort(key=lambda t: t[0])
            # if there are duplicate variables, combine them
            i = 0
            while i + 1 < len(term):
                if term.variables[i][0] == term.variables[i+1][0]:
                    term.variables[i] = term.variables[i][0], (term.variables[i][1] + term.variables.pop(i+1)[1])
                    i -= 1   # adjust index
                i += 1
        # sort terms by lexicographic ordering
        self.terms.sort(key=lambda t: (-sum((tt[1] for tt in t.variables), Number(0)), [(va[0], -va[1]) for va in t.variables]))
        # combine like terms and remove if they cancel out
        i = 0
        while 0 < i + 1 < len(self):
            if self[i].variables == self[i+1].variables:
                # combine coefficients, remembering the sign is taken out
                new = self[i].get_signed_coefficient() + self[i+1].get_signed_coefficient()
                self[i].sign = new.sign
                self[i].coefficient = abs(new)
                self.pop(i+1)
                if self[i].coefficient == 0:
                    self.pop(i)
                    i -= 1   # adjust index
                i -= 1   # adjust index
            i += 1

        return self


class Term(MultiVariableTerm):
    def __init__(self, variable: str, coefficient: NumberArgument, exponent: int):
        """
        A term of a polynomial with a numerical coefficient with one variable raised to an integer power.
        The stored coefficient will always be non-negative.

        :param variable: The variable to be used.
        :param coefficient: The coefficient of the polynomial. Use string for any non-integer values.
        :param exponent: The exponent of this term. Use 0 to represent a constant term.
        """
        super().__init__(coefficient, (variable, exponent), hide_zero_exponent=True)

    @staticmethod
    def from_dict(var: str, dic: dict[str, NumberArgument]) -> "Term":
        """
        Initialize Term from a dictionary.

        Each dictionary must be in the following structure:
        {'coefficient': coefficient value; NumberArgument,
         'exponent': exponent of the variable; int}

        For example, the term -4x^3 is represented by the dictionary:
        {'coefficient': -4,
         'exponent': 3
        }

        For a constant term, the exponent must be 0 and the coefficient must record the value of the constant.

        :param var: The variable to be used.
        :param dic: The dictionary to be converted to Term, following the structure above.
        """
        return Term(var, dic["coefficient"], dic["exponent"])

    @staticmethod
    def singlify(term: MultiVariableTerm) -> "Term":
        """
        Returns a copy of the MultiVariableTerm instance as Term.
        The term must have exactly one variable with integer exponent.
        """
        if len(term.variables) != 1:
            raise ValueError("The term doesn't have one variable, variables: " + str(term.variables))
        if not term.variables[0][1].is_int():
            raise ValueError("The term variable is raised to a non-integer exponent, exponent: " + str(term.variables[0][1]))
        return Term(term.variables[0][0], Number(term.get_signed_coefficient()), int(term.variables[0][1]))

    def __mul__(self, other):
        if ((isinstance(other, Term) and other.get_variable() == self.get_variable()) or
            type(other) in (int, float, Decimal, Number)):
            return Term.singlify(super().__mul__(other))
        else:
            return super().__mul__(other)

    def get_variable(self) -> str:
        """
        Returns the variable the term uses.
        """
        return self.variables[0][0]

    def get_exponent(self) -> Number:
        """
        Returns the exponent of the variable in the term.
        """
        return Number(self.variables[0][1])

    def change_variable(self, new_var: str) -> Self:
        """
        Changes the variable used in this term.

        :returns: The object instance for chained method calls.
        """
        self.variables = [(new_var, self.variables[0][1])]
        return self


class SingleVariablePolynomial(MultiVariablePolynomial):
    def __init__(self, variable: str, data: list[dict[str, NumberArgument] | Term], mix=False, wrap=False):
        """
        A polynomial with numerical coefficients and one variable raised to an integer power.
        The input data can be a list of dictionaries,
        where each dictionary represents a term in the polynomial.

        Each dictionary must be in the following structure:
        {'coefficient': coefficient value; NumberArgument,
         'exponent': exponent of the variable; int}

        For example, the term -4x^3 is represented by the dictionary:
        {'coefficient': -4,
         'exponent': 3}

        For a constant term, the exponent must be 0 and the coefficient must record the value of the constant.

        Each term is assumed to be added. That is, the polynomial 4x - 2 is
        uniquely identified by the data:

        [
         {'coefficient': 4,
          'exponent': 1},
         {'coefficient': -2,
          'exponent': 0}
        ]

        :param variable: The variable to be used.
        :param data: The list of data as explained above, or as a Term object.
        :param mix: If True, the terms will be shuffled upon initialization.
        :param wrap: If True, get_latex will return the polynomial enclosed in parentheses.
        """
        self.variable = variable
        terms = []
        degree = float('-inf')
        for t in data:
            if isinstance(t, Term):
                coef = t.get_signed_coefficient()
                exp = t.get_exponent()
                if variable != t.get_variable():
                    raise ValueError("Polynomial variable doesn't agree with term variable")
                if not exp.is_int():
                    raise ValueError("Exponent must be integer for SingleVariablePolynomial")
                terms.append(Term(variable, coef, int(exp)))
            else:
                if not Number(t["exponent"]).is_int():
                    raise ValueError("Exponent must be integer for SingleVariablePolynomial")
                terms.append(Term.from_dict(variable, t))
            if terms[-1].variables:
                degree = max(degree, int(terms[-1].get_exponent()))
        self._degree = degree if isinstance(degree, int) else 0
        super().__init__(terms, mix, wrap)

    @staticmethod
    def singlify(poly: MultiVariablePolynomial) -> "SingleVariablePolynomial":
        """
        Returns a copy of the MultiVariablePolynomial instance as SingleVariablePolynomial.
        The polynomial must have at least one term.
        The polynomial must use exactly one variable with integer exponents.
        """
        if len(poly.terms) == 0:
            raise ValueError("The polynomial has no terms")
        terms = []
        for t in poly.terms:
            s = Term.singlify(t)
            if not terms or s.get_variable() == terms[-1].get_variable():
                terms.append(s)
            else:
                raise ValueError(f"Different variables seen: {s.get_variable()} and {terms[-1].get_variable()}")
        return SingleVariablePolynomial(terms[0].get_variable(), terms, wrap=poly.wrap)

    @property
    def degree(self) -> int:
        return self._degree

    def __add__(self, other) -> Self:
        if ((isinstance(other, Term) and self.variable == other.get_variable()) or
            (isinstance(other, SingleVariablePolynomial) and self.variable == other.variable) or
            isinstance(other, (int, float, Decimal, Number))):
            return SingleVariablePolynomial.singlify(super().__add__(other))
        return super().__add__(other)

    __radd__ = __add__

    def __mul__(self, other) -> Self:
        if ((isinstance(other, SingleVariablePolynomial) and self.variable == other.variable) or
            (isinstance(other, Term) and self.variable == other.get_variable()) or
            type(other) in (int, float, Decimal, Number)
        ):
            return SingleVariablePolynomial.singlify(super().__mul__(other))
        else:
            return super().__mul__(other)

    __rmul__ = __mul__

    def __divmod__(self, other) -> tuple[Self, Self]:
        if not isinstance(other, SingleVariablePolynomial):
            return NotImplemented
        if self.variable != other.variable:
            raise ValueError(f"variables do not agree between operands: {self.variable} and {other.variable}")

        var = self.variable
        # copy operands to simplify
        dividend = SingleVariablePolynomial(var, self.terms).simplify()
        divisor = SingleVariablePolynomial(var, other.terms).simplify()
        wrap = self.wrap or other.wrap   # wrap unless both don't
        if dividend.degree < divisor.degree:
            # quotient is 0 and the remainder is the entire dividend
            dividend.wrap = wrap
            return SingleVariablePolynomial(var, [Term(var, 0, 0)], wrap=wrap), dividend
        if len(divisor) == 0:
            raise ArithmeticError("division by empty polynomial")
        if len(divisor) == 1:
            # monomial division
            d = divisor[0]
            if not isinstance(d, Term):
                raise AssertionError("divisor contained a MultiVariableTerm, not Term")
            i = 0
            t = dividend[i]
            while isinstance(t, Term) and t.get_exponent() >= d.get_exponent():
                dividend[i] = Term(var,
                                   t.get_signed_coefficient() / d.get_signed_coefficient(),
                                   t.get_exponent() - d.get_exponent())
                i += 1
                try:
                    t = dividend[i]
                except IndexError:
                    break
            quotient = SingleVariablePolynomial(var, dividend.terms[:i])
            remainder = SingleVariablePolynomial(var, dividend.terms[i:])
            return quotient, remainder

        # otherwise, use synthetic division
        # https://en.wikipedia.org/wiki/Synthetic_division#Compact_Expanded_Synthetic_Division
        above_bar = np.zeros((1, dividend.degree + 1), dtype=object)   # 2D array to expand above later
        for t in dividend.terms:
            if isinstance(t, Term):
                above_bar[0, dividend.degree - int(t.get_exponent())] = t.get_signed_coefficient()
            else:
                raise AssertionError("dividend contained a MultiVariableTerm, not Term")
        lead = divisor[0].get_signed_coefficient()   # normalizing factor
        left = np.zeros(divisor.degree, dtype=object)   # negated divisor coefficients except the first
        for t in divisor.terms[1:]:
            if isinstance(t, Term):
                left[divisor.degree - int(t.get_exponent()) - 1] = -t.get_signed_coefficient()
            else:
                raise AssertionError("divisor contained a MultiVariableTerm, not Term")
        right_of_bar = dividend.degree + 1 - left.size   # everything in above_bar at index >= this is to the right of the bar
        bottom = np.zeros(dividend.degree + 1, dtype=object)
        for i in range(dividend.degree + 1):
            q = sum(above_bar[:, i])
            if i >= right_of_bar:
                # this is a remainder, drop down without normalizing
                bottom[i] = q
            else:
                # this is a quotient, normalize and prepare the next products
                bottom[i] = q / lead
                if q != 0:
                    # add the product row above the array
                    new_row = np.zeros((1, dividend.degree + 1), dtype=object)
                    new_row[0, i+1 : i+1+left.size] = bottom[i] * left
                    above_bar = np.vstack((new_row, above_bar))
        quotient = SingleVariablePolynomial(var, [], wrap=wrap)
        quotient_degree = right_of_bar - 1
        remainder = SingleVariablePolynomial(var, [], wrap=wrap)
        for i in range(bottom.size):
            if bottom[i] == 0:
                continue
            if i < right_of_bar:
                quotient.append(Term(var, bottom[i], quotient_degree - i))
            else:
                remainder.append(Term(var, bottom[i], bottom.size - i - 1))
        return quotient, remainder

    def __floordiv__(self, other) -> Self:
        if not isinstance(other, SingleVariablePolynomial):
            return NotImplemented
        return divmod(self, other)[0]

    def __mod__(self, other) -> Self:
        if not isinstance(other, SingleVariablePolynomial):
            return NotImplemented
        return divmod(self, other)[1]

    def append(self, term: Term):
        """
        Adds the term to the end of the polynomial.
        The term must have the same variable as this polynomial.
        """
        if term.get_variable() != self.variable:
            raise ValueError("Term with different variable appended to SingleVariablePolynomial")
        super().append(term)
        if term.variables:
            self._degree = max(self._degree, int(term.get_exponent()))

    def change_variable(self, new_var: str) -> Self:
        """
        Changes the variable used in this polynomial.

        :returns: The object instance for chained method calls.
        """
        self.variable = new_var
        for i in range(len(self.terms)):
            old_term = self.terms[i]
            if isinstance(old_term, Term):
                self.terms[i] = Term(new_var, old_term.get_signed_coefficient(), int(old_term.get_exponent()))
            else:
                raise ValueError("SingleVariablePolynomial contained a MultiVariableTerm, not Term")
        return self

    def _update_degree(self) -> Self:
        self._degree = float('-inf')   # reset degree
        for t in self.terms:
            if isinstance(t, Term):
                if t.variables:
                    self._degree = max(self._degree, int(t.get_exponent()))
            else:
                raise AssertionError("SingleVariablePolynomial contained something other than Term")
        if isinstance(self._degree, float):
            # no term, set degree to 0
            self._degree = 0
        return self

    def remove_zeros(self) -> Self:
        # override to update degree
        return super().remove_zeros()._update_degree()

    def simplify(self) -> Self:
        # override to update degree
        return super().simplify()._update_degree()

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
        result = []
        if self.wrap:
            result.append(Command('left('))
        if self.sign == -1:
            result.append(NoEscape('-'))
        result.append(Command('dfrac', [self.num.dumps(), self.denom.dumps()]))
        if self.wrap:
            result.append(Command('right)'))
        return result


class UnsafeFraction(BaseMathClass):
    def __init__(self,
                 num: str | NumberArgument | BaseMathClass,
                 denom: str | NumberArgument | BaseMathClass,
                 big=False):
        """
        A fraction with arbitrary numerator and denominator.
        If str is given as an argument, it will be converted to NoEscape.

        :param num: The numerator of the fraction.
        :param denom: The denominator of the fraction.
        :param big: If True, get_latex() will return a fraction in display mode (bigger text).
        """
        def convert(o, name: str) -> BaseMathClass:
            if isinstance(o, str):
                return TextWrapper([o])
            elif isinstance(o, BaseMathClass):
                return o
            else:
                try:
                    return Number(o)
                except TypeError:
                    raise ValueError(f"argument {name} with value {o} cannot be converted to BaseMathClass")

        self.num = convert(num, "num")
        self.denom = convert(denom, "denom")
        self.big = big

    def get_latex(self) -> list[Command]:
        return [Command("dfrac" if self.big else "frac", [self.num.dumps(), self.denom.dumps()])]

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
        'value' is always stored as a NoEscape instance.

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
                raise TypeError(f"type {type(term).__name__} is not allowed for a term")
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
