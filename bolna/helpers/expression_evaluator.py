"""Deterministic expression evaluator for graph agent routing.

No eval(), no side effects. Evaluates expression conditions against
context_data for instant routing decisions.
"""

import operator as op
from typing import Any

from bolna.enums import ExpressionOperator, ExpressionLogic, EdgeConditionType

_MISSING = object()

_COMPARISON_OPS = {
    ExpressionOperator.EQ: op.eq,
    ExpressionOperator.NEQ: op.ne,
    ExpressionOperator.GT: op.gt,
    ExpressionOperator.GTE: op.ge,
    ExpressionOperator.LT: op.lt,
    ExpressionOperator.LTE: op.le,
}


def resolve_variable(context_data: dict, path: str) -> Any:
    """Dot-notation lookup. Returns _MISSING if not found."""
    current = context_data
    for segment in path.split("."):
        if isinstance(current, dict) and segment in current:
            current = current[segment]
        else:
            return _MISSING
    return current


def _coerce_for_comparison(actual: Any, expected: Any):
    """Coerce actual/expected to comparable types (e.g. string "3" vs int 3)."""
    if type(actual) == type(expected):
        return actual, expected
    if isinstance(actual, (str, int, float)) and isinstance(expected, (str, int, float)):
        try:
            return float(actual), float(expected)
        except (ValueError, TypeError):
            pass
    return actual, expected


def evaluate_condition(condition: dict, context_data: dict) -> bool:
    """Evaluate a single ExpressionCondition dict against context_data."""
    variable = condition.get("variable", "")
    operator = condition.get("operator", "")
    expected = condition.get("value")

    actual = resolve_variable(context_data, variable)

    if operator == ExpressionOperator.EXISTS:
        return actual is not _MISSING
    if operator == ExpressionOperator.NOT_EXISTS:
        return actual is _MISSING
    if actual is _MISSING:
        return False

    cmp_fn = _COMPARISON_OPS.get(operator)
    if cmp_fn:
        try:
            coerced_actual, coerced_expected = _coerce_for_comparison(actual, expected)
            return cmp_fn(coerced_actual, coerced_expected)
        except TypeError:
            return False

    if operator == ExpressionOperator.IN:
        return actual in expected if isinstance(expected, list) else False
    if operator == ExpressionOperator.NOT_IN:
        return actual not in expected if isinstance(expected, list) else True
    if operator == ExpressionOperator.CONTAINS:
        if isinstance(actual, str) and isinstance(expected, str):
            return expected in actual
        if isinstance(actual, (list, tuple)):
            return expected in actual
        return False

    return False


def evaluate_expression_group(group: dict, context_data: dict) -> bool:
    """Evaluate an AND/OR group of conditions."""
    logic = group.get("logic", ExpressionLogic.AND)
    conditions = group.get("conditions", [])

    if not conditions:
        return False

    if logic == ExpressionLogic.OR:
        return any(evaluate_condition(c, context_data) for c in conditions)
    return all(evaluate_condition(c, context_data) for c in conditions)


def evaluate_edge_expression(edge: dict, context_data: dict) -> bool:
    """Evaluate an edge's expression. Returns True if it matches."""
    condition_type = edge.get("condition_type")

    if condition_type == EdgeConditionType.UNCONDITIONAL:
        return True
    if condition_type != EdgeConditionType.EXPRESSION:
        return False

    expression = edge.get("expression")
    if not expression:
        return False

    return evaluate_expression_group(expression, context_data)
