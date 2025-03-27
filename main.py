import os

from pylatex import Document, Command, Subsection, VerticalSpace, Package, NoEscape
from pylatex import MiniPage
from random import randint

import classes.problems as problems
import classes.problem_preset as preset
from classes.environments import Multicols


# Parameters (edit here)
probs = [problems.PolynomialDivide(100, (-9, 9), (0, 6), no_constant=True)]
prob_name = "Test"
prob_inst = NoEscape("Simplify the following polynomials.")
prob_cols = 2
mix_up = False   # If True, questions are generated in mixed order.


# Don't edit below
doc = Document(geometry_options={"paper": "letterpaper", "margin": "0.8in"})
doc.packages.append(Package("graphicx"))
doc.preamble.append(Command("title", prob_name + " Practice"))
doc.preamble.append(Command("date", ""))
doc.append(Command("maketitle"))

doc.append(prob_inst)


def print_problems():
    q = 0
    while len(probs) > 0:
        q += 1
        probi = randint(0, len(probs) - 1) if mix_up else 0
        prob = probs[probi]
        with doc.create(Subsection("Q" + str(q), False)):
            with doc.create(MiniPage()):
                doc.append(prob.get_problem())
                doc.append(VerticalSpace(prob.vspace))
        if prob.num_quest == 0:
            probs.pop(probi)


if prob_cols == 1:
    print_problems()
else:
    with doc.create(Multicols(prob_cols)):
        print_problems()

doc.generate_tex(os.path.join(os.getcwd(), "document_output", prob_name))
