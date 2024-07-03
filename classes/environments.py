from pylatex import Package
from pylatex.base_classes import Environment


class Multicols(Environment):
    def __init__(self, arguments):
        super().__init__(arguments=arguments)

    _latex_name = "multicols"
    packages = [Package('multicol')]
