# Coding Guidelines

## General Style

- This repository contains data-analysis code. Prefer simple, explicit code that is easy to inspect and modify.
- Keep functions, modules, and individual operations small and narrowly scoped.
- Optimize for readability over abstraction, reuse, or architectural complexity.
- Use clear, descriptive names. Keep related variable definitions together near the beginning of the scope where they are used.

## Fail Fast

- Let invalid data, missing files, incorrect shapes, missing dependencies, and violated assumptions raise errors immediately.
- Do not add defensive code merely to keep a script running.
- Do not silently skip failures, substitute defaults, return sentinel values, or continue with partial results.
- Avoid broad exception handling. Catch an exception only when there is a specific, useful recovery action; otherwise let it propagate.
- Assertions and explicit validation are appropriate when they make an analysis assumption clear, but do not duplicate errors already raised naturally by Python or a library.

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
