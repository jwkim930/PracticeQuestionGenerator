import os

from pylatex import Document, Command, Subsection, VerticalSpace
from random import randint

import classes.problems as problems
from classes.environments import Multicols


probs = [problems.FractionAddition(15, (1, 10), (2, 10)),
         problems.FractionAddition(15, (1, 10), (2, 10), neg=True)]
prob_name = "Fraction Addition"
prob_inst = ""
prob_cols = 3
mix_up = True   # If True, questions are generated in mixed order.


doc = Document(geometry_options={"paper": "letterpaper", "margin": "0.8in"})
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
