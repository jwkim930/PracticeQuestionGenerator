import os

from pylatex import Document, Command, Subsection, VerticalSpace, Package, NoEscape
from pylatex import MiniPage
from random import randint

import classes.problems as problems
import classes.problem_preset as preset
from classes.environments import Multicols

# Parameters (edit here)
probs = [preset.MultiOperationChallengingMix(100, (-9, 9), var=('x', 'y', 'z', 'a', 'b', 'c'))]
prob_name = "Equation"
prob_inst = NoEscape("Solve each of the following equations for the missing variable.")
prob_cols = 1
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
