import os

from pylatex import Document, Command, Section, Subsection, VerticalSpace, Package, NoEscape
from pylatex import MiniPage
from random import randint

import classes.problems as problems
import classes.problem_preset as preset
from classes.environments import Multicols


# Parameters (edit here)
probs = [
    [problems.ExponentRulePractice(10, (-9, 9), (2, 8))],
    [
        problems.PolynomialAdd(3, (-9, 9), (1, 3), *preset.variables, min_term_count=2),
        problems.PolynomialSubtract(3, (-9, 9), (1, 3), *preset.variables, min_term_count=2),
        problems.PolynomialMultiply(3, (-9, 9), (1, 1), *preset.variables, min_term_count = 2, max_term_count=2),
        problems.PolynomialDivide(3, (-9, 9), (1, 3), *preset.variables, no_constant=True)
    ],
    [
        preset.MultiOperationAdvancedMix(2, (-9, 9), preset.variables),
        preset.MultiOperationChallengingMix(2, (-9, 9), preset.variables),
        preset.MultiOperationInsaneMix(1, (-9, 9), preset.variables)
    ],
    [preset.MultiOperationAdvancedMix(3, (-9, 9), preset.variables, True)],
    [problems.LinearGraphingProblem(5, (-4, 4))]
]
prob_names = ["Exponent Rules", "Polynomial Simplification", "Linear Equations", "Linear Inequalities", "Linear Graphing"]
prob_insts = [NoEscape("Simplify the following expressions as much as possible."),
              NoEscape("Simplify the following polynomials."),
              NoEscape("Solve the following linear equations for the unknown variable. Verify your solution."),
              NoEscape("Solve the following inequalities for the unknown variable. Verify your solution both at and beyond the bound."),
              NoEscape("Graph the following linear relations.")]
prob_cols = [2, 1, 1, 1, 1]
# If mix_up is True, questions are generated in mixed order for that section.
mix_up = [False, True, True, False, False]
title = "Grade 9 Review"   # ignored if there's only one section

# Don't edit below
if len({len(probs), len(prob_names), len(prob_insts), len(prob_cols)}) > 1:
    raise ValueError("the number of sections do not agree")
nsec = len(probs)
doc = Document(geometry_options={"paper": "letterpaper", "margin": "0.8in"})
doc.packages.append(Package("graphicx"))
if nsec == 1:
    title = prob_names[0] + " Practice"
doc.preamble.append(Command("title", title))
doc.preamble.append(Command("date", ""))
doc.append(Command("maketitle"))

def print_problems(i: int):
    def helper():
        q = 0
        while len(probs[i]) > 0:
            q += 1
            probi = randint(0, len(probs[i]) - 1) if mix_up[i] else 0
            prob = probs[i][probi]
            with doc.create(Subsection(f"Q {i+1}.{q}", False)):
                with doc.create(MiniPage()):
                    for elem in prob.get_problem():
                        if isinstance(elem, problems.DocInjector):
                            elem.inject(doc)
                        else:
                            doc.append(elem)
                    doc.append(VerticalSpace(prob.vspace))
            if prob.num_quest == 0:
                probs[i].pop(probi)
    if prob_cols[i] == 1:
        helper()
    else:
        with doc.create(Multicols(prob_cols[i])):
            helper()

if len(probs) == 1:
    doc.append(prob_insts[0])
    print_problems(0)
else:
    for sec in range(nsec):
        with doc.create(Section(prob_names[sec], True)):
            doc.append(prob_insts[sec])
            print_problems(sec)
            doc.append(VerticalSpace("1cm"))
doc.generate_tex(os.path.join(os.getcwd(), "document_output", prob_names[0] if len(probs) == 1 else title))
