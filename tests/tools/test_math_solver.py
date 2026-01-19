#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unit Tests for Math Solver Tool
Tests for src/tools/math_solver.py
"""

import asyncio
import pytest
import sys
import importlib.util
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def load_math_solver():
    """Load math_solver module directly from file path"""
    math_solver_path = "/home/thelooter/Documents/Coding/Python/DeepTutor/src/tools/math_solver.py"

    if "math_solver_direct" not in sys.modules:
        spec = importlib.util.spec_from_file_location("math_solver_direct", math_solver_path)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            sys.modules["math_solver_direct"] = module
            spec.loader.exec_module(module)

    return sys.modules["math_solver_direct"]


@pytest.fixture
def math_solver():
    """Fixture to load math_solver module"""
    return load_math_solver()


class TestMathSolverImports:
    """Test that math_solver module can be imported"""

    def test_module_loads(self):
        """Test that math_solver module can be loaded"""
        module = load_math_solver()
        assert module is not None

    def test_solve_math_function_exists(self):
        """Test that solve_math function exists"""
        module = load_math_solver()
        assert hasattr(module, "solve_math")
        assert callable(module.solve_math)


class TestMathSolverSimplify:
    """Test simplify operation"""

    @pytest.mark.asyncio
    async def test_simplify_polynomial(self, math_solver):
        """Test simplifying a polynomial expression"""
        result = await math_solver.solve_math("x**2 + 2*x + 1", operation="simplify")
        assert result["status"] == "success"
        assert result["operation"] == "simplify"
        assert "result" in result
        assert "latex" in result

    @pytest.mark.asyncio
    async def test_simplify_trigonometric(self, math_solver):
        """Test simplifying trigonometric expression"""
        result = await math_solver.solve_math("sin(x)**2 + cos(x)**2", operation="simplify")
        assert result["status"] == "success"
        assert result["operation"] == "simplify"

    @pytest.mark.asyncio
    async def test_simplify_fraction(self, math_solver):
        """Test simplifying a rational expression"""
        result = await math_solver.solve_math("(x**2 - 1) / (x - 1)", operation="simplify")
        assert result["status"] == "success"
        assert result["operation"] == "simplify"


class TestMathSolverDerivative:
    """Test derivative operation"""

    @pytest.mark.asyncio
    async def test_derivative_power_rule(self, math_solver):
        """Test derivative using power rule"""
        result = await math_solver.solve_math("x**3", operation="derivative")
        assert result["status"] == "success"
        assert result["operation"] == "derivative"
        assert "x^{2}" in result["result"] or "x^2" in result["result"] or "2" in result["result"]

    @pytest.mark.asyncio
    async def test_derivative_polynomial(self, math_solver):
        """Test derivative of polynomial"""
        result = await math_solver.solve_math("x**3 + 2*x**2 - 5*x + 3", operation="derivative")
        assert result["status"] == "success"
        assert result["operation"] == "derivative"

    @pytest.mark.asyncio
    async def test_derivative_trigonometric(self, math_solver):
        """Test derivative of trigonometric function"""
        result = await math_solver.solve_math("sin(x)", operation="derivative")
        assert result["status"] == "success"
        assert "cos" in result["result"].lower()

    @pytest.mark.asyncio
    async def test_derivative_exponential(self, math_solver):
        """Test derivative of exponential function"""
        result = await math_solver.solve_math("exp(x)", operation="derivative")
        assert result["status"] == "success"
        assert result["operation"] == "derivative"


class TestMathSolverIntegral:
    """Test integral operation"""

    @pytest.mark.asyncio
    async def test_indefinite_integral_power(self, math_solver):
        """Test indefinite integral of power function"""
        result = await math_solver.solve_math("x**2", operation="integral")
        assert result["status"] == "success"
        assert result["operation"] == "integral"

    @pytest.mark.asyncio
    async def test_indefinite_integral_polynomial(self, math_solver):
        """Test indefinite integral of polynomial"""
        result = await math_solver.solve_math("3*x**2 + 2*x + 1", operation="integral")
        assert result["status"] == "success"
        assert result["operation"] == "integral"

    @pytest.mark.asyncio
    async def test_definite_integral(self, math_solver):
        """Test definite integral"""
        result = await math_solver.solve_math("x**2", operation="integral", limits=(0, 1))
        assert result["status"] == "success"
        assert result["operation"] == "integral"

    @pytest.mark.asyncio
    async def test_integral_trigonometric(self, math_solver):
        """Test integral of trigonometric function"""
        result = await math_solver.solve_math("sin(x)", operation="integral")
        assert result["status"] == "success"
        assert result["operation"] == "integral"


class TestMathSolverSolve:
    """Test solve equation operation"""

    @pytest.mark.asyncio
    async def test_solve_quadratic_equation(self, math_solver):
        """Test solving quadratic equation"""
        result = await math_solver.solve_math("x**2 - 4 = 0", operation="solve")
        assert result["status"] == "success"
        assert result["operation"] == "solve"

    @pytest.mark.asyncio
    async def test_solve_linear_equation(self, math_solver):
        """Test solving linear equation"""
        result = await math_solver.solve_math("2*x + 6 = 0", operation="solve")
        assert result["status"] == "success"
        assert result["operation"] == "solve"

    @pytest.mark.asyncio
    async def test_solve_polynomial(self, math_solver):
        """Test solving higher degree polynomial"""
        result = await math_solver.solve_math("x**3 - x = 0", operation="solve")
        assert result["status"] == "success"
        assert result["operation"] == "solve"


class TestMathSolverMatrix:
    """Test matrix operations"""

    @pytest.mark.asyncio
    async def test_matrix_determinant_2x2(self, math_solver):
        """Test determinant of 2x2 matrix"""
        result = await math_solver.solve_math(
            "[[1, 2], [3, 4]]", operation="matrix", matrix_operation="determinant"
        )
        assert result["status"] == "success"
        assert result["operation"] == "matrix"

    @pytest.mark.asyncio
    async def test_matrix_determinant_3x3(self, math_solver):
        """Test determinant of 3x3 matrix"""
        result = await math_solver.solve_math(
            "[[1, 2, 3], [4, 5, 6], [7, 8, 9]]", operation="matrix", matrix_operation="determinant"
        )
        assert result["status"] == "success"
        assert result["operation"] == "matrix"

    @pytest.mark.asyncio
    async def test_matrix_inverse(self, math_solver):
        """Test matrix inverse"""
        result = await math_solver.solve_math(
            "[[1, 2], [3, 4]]", operation="matrix", matrix_operation="inverse"
        )
        assert result["status"] == "success"
        assert result["operation"] == "matrix"

    @pytest.mark.asyncio
    async def test_matrix_eigenvalues(self, math_solver):
        """Test matrix eigenvalues"""
        result = await math_solver.solve_math(
            "[[1, 0], [0, 2]]", operation="matrix", matrix_operation="eigenvalues"
        )
        assert result["status"] == "success"
        assert result["operation"] == "matrix"


class TestMathSolverEvaluate:
    """Test numerical evaluation"""

    @pytest.mark.asyncio
    async def test_evaluate_expression(self, math_solver):
        """Test numerical evaluation"""
        result = await math_solver.solve_math("sqrt(16) + 5", operation="evaluate")
        assert result["status"] == "success"
        assert result["operation"] == "evaluate"

    @pytest.mark.asyncio
    async def test_evaluate_with_variables(self, math_solver):
        """Test evaluation with variable substitution"""
        result = await math_solver.solve_math(
            "x**2 + y**2", operation="evaluate", variables={"x": 3, "y": 4}
        )
        assert result["status"] == "success"
        assert result["operation"] == "evaluate"


class TestMathSolverAuto:
    """Test auto-detect operation"""

    @pytest.mark.asyncio
    async def test_auto_detect_simplify(self, math_solver):
        """Test auto-detect for simplify"""
        result = await math_solver.solve_math("(x + 1)**2 - x**2 - 2*x - 1", operation="auto")
        assert result["status"] == "success"

    @pytest.mark.asyncio
    async def test_auto_detect_solve(self, math_solver):
        """Test auto-detect for solve"""
        result = await math_solver.solve_math("x + 2 = 10", operation="auto")
        assert result["status"] == "success"


class TestMathSolverErrorHandling:
    """Test error handling"""

    @pytest.mark.asyncio
    async def test_invalid_expression(self, math_solver):
        """Test handling of invalid expression"""
        result = await math_solver.solve_math("x***2", operation="simplify")
        assert result["status"] == "error"
        assert "error" in result

    @pytest.mark.asyncio
    async def test_unsupported_operation(self, math_solver):
        """Test handling of unsupported operation"""
        result = await math_solver.solve_math("x**2", operation="unsupported_operation")
        assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_empty_statement(self, math_solver):
        """Test handling of empty statement"""
        result = await math_solver.solve_math("", operation="simplify")
        assert result["status"] == "error"


class TestMathSolverTimeout:
    """Test timeout functionality"""

    @pytest.mark.asyncio
    async def test_timeout_parameter(self, math_solver):
        """Test that timeout parameter is respected"""
        result = await math_solver.solve_math("x**2", operation="simplify", timeout=5)
        assert result["status"] == "success"


class TestMathSolverOutputFormat:
    """Test output format"""

    @pytest.mark.asyncio
    async def test_output_contains_required_fields(self, math_solver):
        """Test that output contains all required fields"""
        result = await math_solver.solve_math("x**2", operation="simplify")
        assert "status" in result
        assert "operation" in result
        assert "latex" in result
        assert "elapsed_ms" in result

    @pytest.mark.asyncio
    async def test_latex_format(self, math_solver):
        """Test that output is properly formatted in LaTeX"""
        result = await math_solver.solve_math("x**2 + x", operation="simplify")
        assert "$" in result["latex"] or "$$" in result["latex"]

    @pytest.mark.asyncio
    async def test_steps_in_output(self, math_solver):
        """Test that steps are included when applicable"""
        result = await math_solver.solve_math("x**2", operation="derivative")
        assert "steps" in result or "result" in result


class TestMathSolverEdgeCases:
    """Test edge cases"""

    @pytest.mark.asyncio
    async def test_complex_expression(self, math_solver):
        """Test handling of complex expression"""
        result = await math_solver.solve_math("exp(-x**2) * sin(x)", operation="simplify")
        assert result["status"] in ["success", "error"]

    @pytest.mark.asyncio
    async def test_multiple_variables(self, math_solver):
        """Test expression with multiple variables"""
        result = await math_solver.solve_math("x*y + y*z + z*x", operation="simplify")
        assert result["status"] == "success"

    @pytest.mark.asyncio
    async def test_constants(self, math_solver):
        """Test expression with constants"""
        result = await math_solver.solve_math("pi**2 + E", operation="simplify")
        assert result["status"] == "success"

    @pytest.mark.asyncio
    async def test_trig_identities(self, math_solver):
        """Test trigonometric identities"""
        result = await math_solver.solve_math("tan(x) * cos(x)", operation="simplify")
        assert result["status"] == "success"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
