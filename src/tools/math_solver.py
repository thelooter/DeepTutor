#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Math Solver Tool - Symbolic mathematics execution tool

Execute mathematical statements with deterministic, verified results using SymPy.
Supports simplification, derivatives, integrals, equation solving, matrix operations,
and numerical evaluation. Output is formatted in LaTeX for proper frontend rendering.
"""

import asyncio
from dataclasses import dataclass
import logging
from pathlib import Path
import time
from typing import Any

import sympy as sp
from sympy import (
    Matrix,
    N,
    diff,
    integrate,
    simplify,
    solve,
    symbols,
)
from sympy.parsing.sympy_parser import (
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

logger = logging.getLogger("MathSolver")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _load_config() -> dict[str, Any]:
    """Load math_solver configuration from config files"""
    try:
        from src.services.config import load_config_with_main

        config = load_config_with_main("main.yaml", PROJECT_ROOT)
        math_config = config.get("tools", {}).get("math_solver", {})
        if math_config:
            logger.debug("Loaded math_solver config from main.yaml")
            return math_config
    except Exception as e:
        logger.debug(f"Failed to load math_solver config: {e}")

    return {}


def _parse_expression(expr_str: str):
    """Parse a string expression into a SymPy expression"""
    transformations = standard_transformations + (implicit_multiplication_application,)
    expr_str = expr_str.strip()

    if "=" in expr_str:
        left, right = expr_str.split("=", 1)
        return parse_expr(left.strip(), transformations=transformations), parse_expr(
            right.strip(), transformations=transformations
        )

    return parse_expr(expr_str, transformations=transformations)


def _to_latex(expr) -> str:
    """Convert SymPy expression to LaTeX"""
    return sp.latex(expr)


def _auto_detect_operation(statement: str) -> str:
    """Auto-detect the type of mathematical operation from the statement"""
    statement_lower = statement.lower().strip()

    if "=" in statement:
        return "solve"

    keywords = {
        "derivative": ["derivative", "differentiate", "diff"],
        "integral": ["integral", "integrate", "antiderivative"],
        "simplify": ["simplify", "expand", "factor"],
        "matrix": ["matrix", "determinant", "eigenvalue", "inverse"],
        "evaluate": ["evaluate", "compute", "calculate", "numerical"],
    }

    for op, keywords_list in keywords.items():
        for keyword in keywords_list:
            if keyword in statement_lower:
                return op

    return "simplify"


class MathSolverError(Exception):
    """Math solver error"""


@dataclass
class MathSolverResult:
    """Result container for math solver computation"""

    status: str
    operation: str
    input_latex: str
    result: str
    steps: list[str]
    latex: str
    elapsed_ms: float
    error: str = ""


async def _execute_computation(
    statement: str,
    operation: str,
    variables: dict[str, Any] | None,
    limits: tuple[Any, Any] | None,
    matrix_operation: str | None,
) -> MathSolverResult:
    """Execute the mathematical computation (runs inside timeout wrapper)"""
    if operation == "auto":
        operation = _auto_detect_operation(statement)

    input_expr, output_expr = None, None
    steps: list[str] = []
    input_latex = ""
    result_latex = ""

    if operation == "simplify":
        input_expr = _parse_expression(statement)
        input_latex = _to_latex(input_expr)
        output_expr = simplify(input_expr)
        result_latex = _to_latex(output_expr)
        steps = [f"\\text{{Simplify: }} {input_latex} = {result_latex}"]

    elif operation == "derivative":
        input_expr = _parse_expression(statement)
        input_latex = _to_latex(input_expr)

        var = list(input_expr.free_symbols)[0] if input_expr.free_symbols else symbols("x")

        output_expr = diff(input_expr, var)
        result_latex = _to_latex(output_expr)

        if limits is not None:
            output_expr = integrate(input_expr, (var, limits[0], limits[1]))
            result_latex = _to_latex(output_expr)
            steps = [f"\\int_{{{limits[0]}}}^{{{limits[1]}}} {input_latex} \\, dx = {result_latex}"]
        else:
            steps = [f"\\frac{{d}}{{d{var}}} \\left( {input_latex} \\right) = {result_latex}"]

    elif operation == "integral":
        input_expr = _parse_expression(statement)
        input_latex = _to_latex(input_expr)

        var = list(input_expr.free_symbols)[0] if input_expr.free_symbols else symbols("x")

        if limits is not None:
            output_expr = integrate(input_expr, (var, limits[0], limits[1]))
            result_latex = _to_latex(output_expr)
            steps = [f"\\int_{{{limits[0]}}}^{{{limits[1]}}} {input_latex} \\, dx = {result_latex}"]
        else:
            output_expr = integrate(input_expr, var)
            result_latex = _to_latex(output_expr)
            steps = [f"\\int {input_latex} \\, dx = {result_latex} + C"]

    elif operation == "solve":
        input_expr, rhs = _parse_expression(statement)
        input_latex = f"{_to_latex(input_expr)} = {_to_latex(rhs)}"
        equation = sp.Eq(input_expr, rhs)
        solutions = solve(equation, input_expr.free_symbols)
        output_expr = solutions
        result_latex = ", ".join([_to_latex(s) for s in solutions])
        steps = [f"\\text{{Solve: }} {input_latex}", f"x = {result_latex}"]

    elif operation == "matrix":
        matrix_str = statement.strip()
        if matrix_str.startswith("[") and matrix_str.endswith("]"):
            matrix_str = matrix_str[1:-1]
        rows = matrix_str.split("],")
        matrix_data = []
        for row in rows:
            row = row.strip().strip("[]")
            if row:
                values = [float(x.strip()) for x in row.split(",")]
                matrix_data.append(values)

        mat = Matrix(matrix_data)
        input_latex = _to_latex(mat)

        if matrix_operation == "determinant":
            output_expr = mat.det()
            result_latex = _to_latex(output_expr)
            steps = [f"\\det \\begin{{bmatrix}} {input_latex} \\end{{bmatrix}} = {result_latex}"]
        elif matrix_operation == "inverse":
            output_expr = mat.inv()
            result_latex = _to_latex(output_expr)
            steps = [
                f"\\left( \\begin{{bmatrix}} {input_latex} \\end{{bmatrix}} \\right)^{{-1}} = {result_latex}"
            ]
        elif matrix_operation == "eigenvalues":
            output_expr = mat.eigenvals()
            result_latex = ", ".join(
                [f"\\lambda_{{{i + 1}}} = {_to_latex(v)}" for i, v in enumerate(output_expr)]
            )
            steps = [f"\\text{{Eigenvalues: }} {result_latex}"]
        else:
            output_expr = mat
            result_latex = _to_latex(mat)
            steps = [f"\\begin{{bmatrix}} {input_latex} \\end{{bmatrix}}"]

    elif operation == "evaluate":
        input_expr = _parse_expression(statement)
        input_latex = _to_latex(input_expr)

        if variables:
            subs_expr = input_expr
            for var_name, value in variables.items():
                var = symbols(var_name)
                subs_expr = subs_expr.subs(var, value)
            output_expr = N(subs_expr, 10)
        else:
            output_expr = N(input_expr, 10)

        result_latex = str(output_expr)
        steps = [f"\\text{{Evaluate: }} {input_latex} = {result_latex}"]

    else:
        raise MathSolverError(f"Unsupported operation: {operation}")

    if output_expr is None:
        output_expr = input_expr

    return MathSolverResult(
        status="success",
        operation=operation,
        input_latex=input_latex if input_latex else _to_latex(input_expr) if input_expr else "",
        result=result_latex if result_latex else _to_latex(output_expr),
        steps=steps,
        latex=f"$$ {steps[-1] if steps else _to_latex(output_expr)} $$",
        elapsed_ms=0,
    )


async def solve_math(
    statement: str,
    operation: str = "auto",
    timeout: int = 30,
    variables: dict[str, Any] | None = None,
    limits: tuple[Any, Any] | None = None,
    matrix_operation: str | None = None,
) -> dict[str, Any]:
    """
    Execute a mathematical statement and return the result.

    Args:
        statement: Mathematical statement to execute (e.g., "x**2 + 2*x + 1")
        operation: Type of operation (auto, simplify, derivative, integral, solve, matrix, evaluate)
        timeout: Execution timeout in seconds
        variables: Dictionary of variable values for substitution
        limits: Tuple of (lower, upper) limits for definite integration
        matrix_operation: Specific matrix operation (determinant, inverse, eigenvalues)

    Returns:
        dict with keys:
            - status: "success" or "error"
            - operation: The operation that was performed
            - input_latex: LaTeX representation of input
            - result: Final answer (LaTeX formatted)
            - steps: List of solution steps (LaTeX formatted)
            - latex: Full LaTeX output
            - elapsed_ms: Execution time in milliseconds
            - error: Error message if failed
    """
    start_time = time.time()

    if not statement or not statement.strip():
        return {
            "status": "error",
            "operation": operation,
            "input_latex": "",
            "result": "",
            "steps": [],
            "latex": "",
            "elapsed_ms": (time.time() - start_time) * 1000,
            "error": "Empty statement provided",
        }

    try:
        result = await asyncio.wait_for(
            _execute_computation(statement, operation, variables, limits, matrix_operation),
            timeout=timeout,
        )

        elapsed_ms = (time.time() - start_time) * 1000

        return {
            "status": result.status,
            "operation": result.operation,
            "input_latex": result.input_latex,
            "result": result.result,
            "steps": result.steps,
            "latex": result.latex,
            "elapsed_ms": elapsed_ms,
        }

    except asyncio.TimeoutError:
        elapsed_ms = (time.time() - start_time) * 1000
        logger.error(f"Math solver timed out after {timeout}s: {statement[:50]}...")
        return {
            "status": "error",
            "operation": operation,
            "input_latex": statement,
            "result": "",
            "steps": [],
            "latex": "",
            "elapsed_ms": elapsed_ms,
            "error": f"Computation timed out after {timeout} seconds",
        }

    except Exception as e:
        elapsed_ms = (time.time() - start_time) * 1000
        error_msg = str(e)

        logger.error(f"Math solver error: {e}", exc_info=True)

        return {
            "status": "error",
            "operation": operation,
            "input_latex": statement,
            "result": "",
            "steps": [],
            "latex": "",
            "elapsed_ms": elapsed_ms,
            "error": error_msg,
        }


def solve_math_sync(
    statement: str,
    operation: str = "auto",
    timeout: int = 30,
    **kwargs,
) -> dict[str, Any]:
    """
    Synchronous version of solve_math (for non-async environments)
    """
    return asyncio.run(solve_math(statement, operation, timeout, **kwargs))


if __name__ == "__main__":

    async def _demo():
        print("==== 1. Test simplify ====")
        result = await solve_math("x**2 + 2*x + 1", operation="simplify")
        print(f"Status: {result['status']}")
        print(f"Result: {result['result']}")
        print(f"LaTeX: {result['latex']}")
        print("-" * 40)

        print("==== 2. Test derivative ====")
        result = await solve_math("x**3", operation="derivative")
        print(f"Status: {result['status']}")
        print(f"Result: {result['result']}")
        print(f"Steps: {result['steps']}")
        print("-" * 40)

        print("==== 3. Test solve equation ====")
        result = await solve_math("x**2 - 4 = 0", operation="solve")
        print(f"Status: {result['status']}")
        print(f"Result: {result['result']}")
        print("-" * 40)

        print("==== 4. Test integral ====")
        result = await solve_math("x**2", operation="integral")
        print(f"Status: {result['status']}")
        print(f"Result: {result['result']}")
        print("-" * 40)

        print("==== 5. Test matrix determinant ====")
        result = await solve_math(
            "[[1, 2], [3, 4]]", operation="matrix", matrix_operation="determinant"
        )
        print(f"Status: {result['status']}")
        print(f"Result: {result['result']}")
        print("-" * 40)

        print("==== 6. Test auto-detect ====")
        result = await solve_math("x + 2 = 10", operation="auto")
        print(f"Detected operation: {result['operation']}")
        print(f"Result: {result['result']}")
        print("-" * 40)

    asyncio.run(_demo())
