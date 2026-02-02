import os

from pylatex import Document, Command, Section, Subsection, VerticalSpace, Package, NoEscape
from pylatex import MiniPage
from random import randint

import classes.problems as problems
import classes.problem_preset as preset
from classes.environments import Multicols


# Parameters (edit here)
probs = [
    [problems.WordProblem(6, "10cm", "Grade11TrigSequencesQuadratics", [
        (10000, 15000),
        (5, 7),
        (30, 35),
        (1449, 1549),
        (65, 75),
        (16000, 20000),
        (50, 60),
        (400, 600),
        (3500, 7000),
        (5, 10)
    ])]
]  # outer list is section, inner list is problems within the section
prob_names = ["Test"]   # name of each section
prob_insts = ["Testing"]   # instruction for each section
prob_cols = [1]   # number of columns for each section
# If mix_up is True, questions are generated in mixed order for that section.
mix_up = [False]
title = "Test"   # ignored if there's only one section



# Don't edit below
if len({len(probs), len(prob_names), len(prob_insts), len(prob_cols), len(mix_up)}) > 1:
    raise ValueError("the number of sections do not agree")
nsec = len(probs)
doc = Document(geometry_options={"paper": "letterpaper", "margin": "0.8in"})
doc.packages.append(Package("graphicx,tikz"))
if nsec == 1:
    title = prob_names[0] + " Practice"
doc.preamble.append(Command("usetikzlibrary", "angles,quotes,calc"))
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
            qnum = f"Q {i+1}.{q}" if nsec > 1 else f"Q{q}"
            with doc.create(Subsection(qnum, False)):
                if not isinstance(prob, problems.WordProblem):
                    with doc.create(MiniPage()):
                        for elem in prob.get_problem():
                            if isinstance(elem, problems.DocInjector):
                                elem.inject(doc)
                            else:
                                doc.append(elem)
                        doc.append(VerticalSpace(prob.vspace))
                else:
                    # page overflow should happen with word problems
                    doc.append(prob.get_problem()[0])
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

output_dir = os.path.join(os.getcwd(), "document_output")
if not os.path.isdir(output_dir):
    os.mkdir(output_dir)
doc.generate_tex(os.path.join(output_dir, prob_names[0] if len(probs) == 1 else title))
