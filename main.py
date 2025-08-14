import os

from pylatex import Document, Command, Section, Subsection, VerticalSpace, Package, NoEscape
from pylatex import MiniPage
from random import randint

import classes.problems as problems
import classes.problem_preset as preset
from classes.environments import Multicols


# Parameters (edit here)
probs = [
    [problems.LinearGraphingProblem(10, (-5, 5))]
]
prob_names = ["Linear Graphing"]
prob_insts = [preset.graphing_instruction]
prob_cols = [1]
# If mix_up is True, questions are generated in mixed order for that section.
mix_up = [False]
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
