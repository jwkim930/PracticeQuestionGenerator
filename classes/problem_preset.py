# Pre-made problems with appropriate ranges
# Just set num_quest as desired

import classes.problems as problems


class RectangleAreaProblem(problems.WordProblem):
    def __init__(self, num_quest: int):
        super().__init__(num_quest, '4cm', 'RectangleArea', [(2, 12), (2, 12)])


class TriangleAreaProblem(problems.WordProblem):
    def __init__(self, num_quest: int):
        super().__init__(num_quest, '4cm', 'TriangleArea', [(4, 8), (2, 6)])


class CircleCircAreaProblem(problems.WordProblem):
    def __init__(self, num_quest: int):
        super().__init__(num_quest, '4cm', 'CircleCircArea', [(2, 8)])


class ParalleloAreaProblem(problems.WordProblem):
    def __init__(self, num_quest: int):
        super().__init__(num_quest, '4cm', 'ParalleloArea', [(4, 8), (2, 6)])

linear_relation_mix = [problems.LinearRelationProblem(4, (-3, 3), (-2, 2)),
                       problems.LinearRelationProblem(1, (-3, 3), (-2, 2), (-2, 2)),
                       problems.LinearRelationProblem(1, (0, 0), (-2, 2), no_constant=False)]

single_operation_equation_mix = [problems.EquationSingleOperation(3, (-9, 9), 'add'),
                                 problems.EquationSingleOperation(3, (-9, 9), 'sub'),
                                 problems.EquationSingleOperation(3, (-9, 9), 'mul'),
                                 problems.EquationSingleOperation(3, (-9, 9), 'div')]
