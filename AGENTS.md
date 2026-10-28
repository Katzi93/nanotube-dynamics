# Coding Guidelines

## Repository modules

- `tube_tracker/` contains the tube tracking software and graphical interface.
- `tracking_analysis/` contains the current analysis of tube tracking data.
- `ito_light_response/` contains the ITO light-response analysis.
- `legacy_scripts/` contains the older exploratory analysis formerly in `test_analysis/`.
- `tests/` contains interactive tracking tests and development scripts; `run_tracking_interface.py` launches the interface.

## Mathematical writeup and LaTeX

- The paper's mathematical SI is in `supplementary_information/main.tex`.
- Before editing or compiling it, read `supplementary_information/AGENTS.md` and `supplementary_information/README.md`.
- Use the existing `supplementary_information/build.ps1` build command. The verified MiKTeX setup does not need to be rediscovered on each turn.

## Python Environment

- This project runs in a conda environment named `tube-tracking-clean`.
- Commands here execute in Git Bash, where `python`, `py`, `python3`, and `conda` are not on `PATH`; bare `python` hits the Windows Store stub and fails with "Python wurde nicht gefunden".
- Call the interpreter by full path: `/c/Users/Flori/anaconda3/envs/tube-tracking-clean/python.exe`.
- Running `conda init bash` once from an Anaconda Prompt would let `conda activate tube-tracking-clean` work in Git Bash and resolve bare `python`.

## General Style

- This repository contains data-analysis code. Prefer simple, explicit code that is easy to inspect and modify.
- Keep functions, modules, and individual operations small and narrowly scoped.
- Optimize for readability over abstraction, reuse, or architectural complexity.
- Use clear, descriptive names. Keep related variable definitions together near the beginning of the scope where they are used.
- Use the available line width instead of aggressively wrapping code. Keep ordinary calculations, assignments, conditions, and short function calls on one line when they fit within approximately 120 characters.
- Long function definitions and argument lists may place one parameter per line, but do not split a short expression across several lines merely to satisfy a narrow formatter default.
- Calculate reported values before print statements. Print statements should format named values rather than perform calculations inside formatted strings.
- Explain spatial selections and fitted geometry with exact coordinates, variables, and inequalities. Do not use vague phrases such as "near the point" or "within the region" when the code implements a specific mathematical rule.
- Do not use ASCII or plain-text diagrams to explain geometry. Use concise prose, equations, and concrete numerical examples.

## Fail Fast

- Let invalid data, missing files, incorrect shapes, missing dependencies, and violated assumptions raise errors immediately.
- Do not add defensive code merely to keep a script running.
- Do not silently skip failures, substitute defaults, return sentinel values, or continue with partial results.
- Avoid broad exception handling. Catch an exception only when there is a specific, useful recovery action; otherwise let it propagate.
- Do not clutter analysis code with assertions for routine paths, shapes, types, dependencies, or intermediate conditions. Compact assertions are appropriate for genuine mathematical parameter constraints. Otherwise, let Python and library operations fail naturally unless an invalid scientific assumption could produce a plausible but incorrect result.

## Script Structure

- Put imports at the top of the file.
- Define reusable functions above the executable script block.
- Put executable analysis code directly under:

```python
if __name__ == "__main__":
    ...
```

- Do not create a `main()` function solely to call it from the `if __name__ == "__main__":` block.
- Define all analysis parameters together in one clearly marked block near the beginning of the `__main__` block.
- Add a brief comment to every analysis parameter explaining what it controls.
- Put a parameter's brief explanation inline after its assignment. Reserve separate preceding comment lines for explanations that apply to several lines or require more than one line.
- Do not hide unexplained numeric or string literals inside functions; expose values that affect the analysis in the parameter block.
- Keep intermediate values local to the `__main__` block or to the function that uses them.

## Scope and Global State

- Do not use module-level variables as a substitute for passing values explicitly or keeping them local.
- Module-level constants are acceptable when they genuinely describe module-wide configuration or fixed domain values.
- Library modules may expose deliberate module-level definitions, but mutable global state should be avoided.
- Functions should receive the data and configuration they need through explicit parameters.

## Analysis Workflow

- Treat everything in `raw_data/` and `test_data/` as read-only. Never modify, rename, move, overwrite, or delete files in these directories.
- Write processed data, generated figures, and other outputs somewhere outside `raw_data/` and `test_data/`.
- Add concise console feedback to long-running code so the current stage and meaningful progress are visible.
- Report progress at useful intervals rather than printing once per item in large loops.
- Make data transformations explicit and easy to follow in execution order.
- Prefer a short sequence of clear operations over a framework or deeply layered pipeline.
- Preserve useful errors and tracebacks. A visible failure is preferable to a plausible but incorrect analysis result.
- Add abstractions only when they reduce real repetition or make the analysis easier to understand.

## Large Downloads and Installations

- The human, not the AI agent, performs installations and downloads involving large files.
- Before a large download or installation, explain what is needed and provide the exact command for the human to run.
- Wait for the human to confirm that the command finished before continuing with dependent work.
- The AI agent may inspect versions and verify completed installations, but must not start or continue a large download or installation itself.
