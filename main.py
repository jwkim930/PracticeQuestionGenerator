import os

from pylatex import Document, Command, Subsection

import classes.problems as problems
from classes.environments import Multicols


probs = [problems.FractionAddition(30, (1, 10), (1, 10))]
prob_name = probs[0].name
prob_inst = probs[0].instruction
prob_cols = probs[0].num_col

doc = Document(geometry_options={"paper": "letterpaper", "margin": "0.8in"})
doc.preamble.append(Command("title", prob_name + " Practice"))
doc.preamble.append(Command("date", ""))
doc.append(Command("maketitle"))

doc.append(prob_inst)


def print_problems():
    q = 0
    for prob in probs:
        for i in range(prob.num_quest):
            q += 1
            with doc.create(Subsection("Q"+str(q), False)):
                doc.append(prob.get_problem())
                doc.append(Command("vspace", prob.vspace))


if prob_cols == 1:
    print_problems()
else:
    with doc.create(Multicols(prob_cols)):
        print_problems()

doc.generate_tex(os.path.join(os.getcwd(), "document_output", prob_name))
