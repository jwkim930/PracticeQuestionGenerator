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


multi_operation_double_test = [problems.EquationMultiOperation(2, (-5, 5), 'double'),
                        problems.EquationMultiOperation(2, (-5, 5), 'double_dist'),
                        problems.EquationMultiOperation(2, (-5, 5), 'double_frac'),
                        problems.EquationMultiOperation(2, (-5, 5), 'double_frac_dist'),
                        problems.EquationMultiOperation(2, (-5, 5), 'rational')]


class MultiOperationBasicMix(problems.EquationMultiOperation):
    def __init___(self, num_quest: int, nrange: tuple[int, int], var=('x',), inequality=False):
        super().__init__(num_quest, nrange, 'simple', 'simple_div', 'simple_dist', var=var, inequality=inequality)


class MultiOperationAdvancedMix(problems.EquationMultiOperation):
    def __init___(self, num_quest: int, nrange: tuple[int, int], var=('x',), inequality=False):
        super().__init__(num_quest, nrange, 'double', 'double_dist', 'double_frac', 'double_frac_dist', 'rational', var=var, inequality=inequality)

class MultiOperationChallengingMix(problems.EquationMultiOperation):
    def __init__(self, num_quest: int, nrange: tuple[int, int], var=('x',), inequality=False):
        super().__init__(num_quest, nrange, 'frac_const', 'bino_frac', 'double_bino_frac', 'bino_frac_const', 'double_bino_frac_large',var=var, inequality=inequality)
