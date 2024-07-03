from pylatex import Command


class BaseMathClass:
    def get_command(self) -> Command:
        """
        Returns the Command representation of the object.
        """
        pass


class Fraction(BaseMathClass):
    def __init__(self, num: int, denom: int, big=True):
        """
        A fraction with integer numerator/denominator.
        :param num: Numerator of the fraction.
        :param denom: Denominator of the fraction.
        :param big: If True, get_command() will return a big fraction.
        """
        self.num = num
        self.denom = denom
        self.big = big

    def get_command(self) -> Command:
        return Command("dfrac" if self.big else "frac", [self.num, self.denom])
