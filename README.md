# Math Practice Question Generator

A Python-based tool that generates randomized mathematics practice worksheets using LaTeX. The problem set covers primary/secondary school mathematics.

## Features

This project supports a wide variety of problem types. Some of the currently implemented options are:

* **Arithmetic & Operations:** Integer and decimal arithmetic, order of operations (BEDMAS), and exponent rules.
* **Algebra:**
    * Simplifying polynomials (addition, subtraction, multiplication, division).
    * Factoring polynomials (common factors, quadratics, special cases).
    * Solving linear and quadratic equations (factoring, quadratic formula).
    * Solving systems of linear equations.
    * Simplifying rational exponents and radicals.
* **Graphing & Functions:**
    * Linear relations (finding equations, graphing lines).
    * Quadratic functions (graphing, identifying equations from graphs).
* **Trigonometry:** Solving right-angled triangles (SOH CAH TOA).
* **Word Problems:** Templated word problems for area, perimeter, etc.

## Prerequisites

1.  **Python 3.10+**: This project uses pattern matching that was added in Python 3.10. Ensure you have a modern version of Python installed.
2.  **LaTeX Distribution**: This project uses [`PyLaTeX`](https://github.com/JelteF/PyLaTeX). While this project itself does not require a LaTeX compiler, as it exports the result in a `.tex` file, you will still need one to convert it to a PDF file for your use.
    * **Windows**: [MiKTeX](https://miktex.org/) or [TeX Live](https://www.tug.org/texlive/).
    * **macOS**: [MacTeX](https://www.tug.org/mactex/).
    * **Linux**: `texlive-full` (e.g., `sudo apt-get install texlive-full`).

## Installation

1.  Clone or download this repository.
2.  Install the required Python packages:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Configure the Worksheet**:
    Open `main.py` and modify the "Parameters" section to define what problems you want to generate.
    * Each list in the parameters define a section in the output document. Each section has a title (`prob_names`), a set of problems (`probs`), and instruction (`prob_insts`).
    * A section can have a mix of different problem types. If `mix_up` is `False`, they will appear in the order they were written in the list. Note that `prob` must always be a list of lists even if there's only one problem type for the section or there's only one section.
    * If there's only one section, the section title (`prob_names`) is used for the document title.
    * For the list of available problems, please refer to the classes in `problems.py` and their docstrings. There are also some problems in `preset.py`, which uses pre-defined parameters for convenience.

    ```python
    # Example configuration in main.py
    probs = [
     [
         problems.EquationSingleOperation(5, (-9, 9), 'add', 'sub'),
         problems.EquationMultiOperation(5, (-9, 9), 'simple', 'simple_dist', var=preset.variables)
     ],
     [problems.BEDMASPractice(5, (-10, 10))]
    ]
    prob_names = ["Linear Equation", "Order of Operations"]   # name of each section
    prob_insts = [preset.equation_instruction, "Evaluate the following expressions."]   # instruction for each section
    prob_cols = [1, 2]   # Number of columns for each section
    # If mix_up is True, questions are generated in mixed order for that section.
    mix_up = [True, False]
    title = "Math Worksheet"   # ignored if there's only one section
    ```

2.  **Run the Generator**:
    Execute the script:

    ```bash
    python main.py
    ```

3.  **Output**:
    The script will generate a `.tex` file. Check the `document_output` directory for the result.

## Project Structure

* `main.py`: The entry point of the application. Handles configuration and document generation.
* `classes/`:
    * `problems.py`: Contains the logic for generating specific math problems.
    * `math_objects.py`: Defines mathematical entities like `Fraction`, `SingleVariablePolynomial`, `Term`, etc., and handles their LaTeX string representation.
    * `problem_preset.py`: Helper file containing predefined problem sets, variables, and problem instructions.
    * `environments.py`: Custom LaTeX environments (e.g., Multicols).
