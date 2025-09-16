from langchain_core.tools import StructuredTool
from pydantic import BaseModel, field_validator
import sympy as sp

_TOOL_DESCRIPTION = (
    "evaluate_expression(expression: str) -> float:\n"
    " - Evaluates a mathematical expression (e.g., '(1 + 2) * 3 / 4') involving addition (+), subtraction (-), multiplication (*), division (/), and parentheses.\n"
    " - Supports numbers (e.g., '1', '3.14') and percentages (e.g., '110%' -> 1.1).\n"
    " - Returns the result as a float.\n"
)

class MathExpressionInput(BaseModel):
    expression: str

    @field_validator('expression')
    def validate_expression(cls, value):
        if not value.strip():
            raise ValueError("Expression cannot be empty")
        return value

def evaluate_expression(expression: str) -> float:
    """
    Evaluate a mathematical expression.
    Example: '(1 + 2) * 3 / 4' -> 2.25
    """
    inputs = MathExpressionInput(expression=expression)

    # Replace percentages (e.g., '110%' -> 1.1)
    import re
    for perc in re.findall(r'\d+%', expression):
        try:
            resolved_value = str(float(perc[:-1]) / 100.0)
            expression = expression.replace(perc, resolved_value)
        except ValueError:
            raise ValueError(f"Invalid percentage format: {perc}")

    try:
        result = sp.sympify(expression, evaluate=True)
        return float(result)
    except sp.SympifyError:
        raise ValueError(f"Invalid expression: {expression}")
    except ZeroDivisionError:
        raise ValueError("Division by zero is not allowed")
    except Exception as e:
        raise ValueError(f"Error evaluating expression: {str(e)}")

def get_math_tool():
    return StructuredTool.from_function(
        name="evaluate_expression",
        func=evaluate_expression,
        description=_TOOL_DESCRIPTION,
        args_schema=MathExpressionInput
    )