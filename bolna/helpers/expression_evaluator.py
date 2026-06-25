"""Deterministic expression evaluator for graph agent routing.

No eval(), no side effects. Evaluates expression conditions against
context_data for instant routing decisions.
"""

import operator as op
from typing import Any, Optional

from bolna.enums import ExpressionOperator, ExpressionLogic, EdgeConditionType, VariableType
from bolna.helpers.logger_config import configure_logger

logger = configure_logger(__name__)

_MISSING = object()
MISSING = _MISSING  # public alias so sibling modules can test for "path not found"

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


def set_variable(context_data: dict, path: str, value: Any) -> None:
    """Dot-notation setter (the write twin of resolve_variable), creating intermediate
    dicts as needed."""
    segments = path.split(".")
    current = context_data
    for segment in segments[:-1]:
        nxt = current.get(segment)
        if not isinstance(nxt, dict):
            nxt = {}
            current[segment] = nxt
        current = nxt
    current[segments[-1]] = value


_TRUE_TOKENS = {"true", "1", "yes"}
_FALSE_TOKENS = {"false", "0", "no"}
_BOOL_TOKENS = _TRUE_TOKENS | _FALSE_TOKENS


def _is_bool_like(value: Any) -> bool:
    """True for a bool or a string that names one, so inference only treats genuine
    booleans as boolean and leaves numbers/other strings to numeric comparison."""
    return isinstance(value, bool) or (isinstance(value, str) and value.strip().lower() in _BOOL_TOKENS)


def _to_bool(value: Any) -> bool:
    """Coerce a value to bool. Raises ValueError on an unrecognized string."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        token = value.strip().lower()
        if token in _TRUE_TOKENS:
            return True
        if token in _FALSE_TOKENS:
            return False
    raise ValueError(f"cannot coerce {value!r} to boolean")


def _coerce_value(value: Any, declared_type: VariableType) -> Any:
    """Coerce a single value into a declared type. Raises on failure."""
    if declared_type == VariableType.STRING:
        return str(value)
    if declared_type == VariableType.NUMBER:
        return float(value)
    if declared_type == VariableType.BOOLEAN:
        return _to_bool(value)
    raise ValueError(f"unsupported variable type: {declared_type!r}")


def _resolve_declared_type(
    variable: str, variable_types: Optional[dict], log_unknown: bool = True
) -> Optional[VariableType]:
    """Resolve a variable's declared type. Keys match the condition's variable path
    exactly. An unrecognized type name falls back to inference, warning once when
    log_unknown (the trace/describe pass passes False to avoid duplicate warnings).
    """
    if not variable_types:
        return None
    raw = variable_types.get(variable)
    if raw is None:
        return None
    try:
        return VariableType(raw)
    except ValueError:
        if log_unknown:
            logger.warning(f"Unknown variable type {raw!r} for {variable!r}; falling back to inference")
        return None


def coerce_to_type(value: Any, variable: str, variable_types: Optional[dict]) -> Any:
    """Coerce value to the type declared for `variable`, or return it unchanged if no
    known type is declared. Raises TypeError/ValueError if coercion fails."""
    declared = _resolve_declared_type(variable, variable_types, log_unknown=False)
    return _coerce_value(value, declared) if declared is not None else value


def _coerce_operands(actual: Any, expected: Any, operator: str, declared_type: Optional[VariableType]):
    """Bring actual/expected into the same domain before the operator runs.

    A declared type coerces both operands (list elements too, for membership) into that
    type, so every operator compares in the declared domain. Without one, only scalar
    comparisons get inference coercion (same type as-is, a boolean literal vs a
    boolean-like value -> boolean, else numeric); membership/substring keep raw values.
    """
    if declared_type is not None:
        actual = _coerce_value(actual, declared_type)
        if isinstance(expected, list):
            expected = [_coerce_value(item, declared_type) for item in expected]
        elif expected is not None:
            expected = _coerce_value(expected, declared_type)
        return actual, expected

    if operator not in _COMPARISON_OPS:
        return actual, expected
    if type(actual) is type(expected):
        return actual, expected
    if isinstance(expected, bool) and _is_bool_like(actual):
        return _to_bool(actual), expected
    if isinstance(actual, (str, int, float)) and isinstance(expected, (str, int, float)):
        try:
            return float(actual), float(expected)
        except (ValueError, TypeError):
            pass
    return actual, expected


def evaluate_condition(
    condition: dict, context_data: dict, variable_types: Optional[dict] = None, log_failures: bool = True
) -> bool:
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

    declared_type = _resolve_declared_type(variable, variable_types, log_unknown=log_failures)
    try:
        actual, expected = _coerce_operands(actual, expected, operator, declared_type)
    except (TypeError, ValueError):
        if log_failures and declared_type is not None:
            logger.warning(
                f"Expression coercion failed: {variable}={actual!r} {operator} {expected!r} as {declared_type.value}"
            )
        return False

    cmp_fn = _COMPARISON_OPS.get(operator)
    if cmp_fn:
        try:
            return cmp_fn(actual, expected)
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


def evaluate_expression_group(group: dict, context_data: dict, variable_types: Optional[dict] = None) -> bool:
    """Evaluate an AND/OR group of conditions."""
    logic = group.get("logic", ExpressionLogic.AND)
    conditions = group.get("conditions", [])

    if not conditions:
        return False

    if logic == ExpressionLogic.OR:
        return any(evaluate_condition(c, context_data, variable_types) for c in conditions)
    return all(evaluate_condition(c, context_data, variable_types) for c in conditions)


def evaluate_edge_expression(edge: dict, context_data: dict, variable_types: Optional[dict] = None) -> bool:
    """Evaluate an edge's expression. Returns True if it matches."""
    condition_type = edge.get("condition_type")

    if condition_type == EdgeConditionType.UNCONDITIONAL:
        return True
    if condition_type != EdgeConditionType.EXPRESSION:
        return False

    expression = edge.get("expression")
    if not expression:
        return False

    return evaluate_expression_group(expression, context_data, variable_types)


def describe_condition(condition: dict, context_data: dict, variable_types: Optional[dict] = None) -> str:
    """Trace one condition for routing logs: "variable operator value (actual=<resolved>) -> <result>"."""
    variable = condition.get("variable", "")
    operator = condition.get("operator", "")
    actual = resolve_variable(context_data, variable)
    actual_repr = "<missing>" if actual is _MISSING else repr(actual)
    result = evaluate_condition(condition, context_data, variable_types, log_failures=False)
    declared = _resolve_declared_type(variable, variable_types, log_unknown=False)
    type_note = f", as {declared.value}" if declared else ""

    if operator in (ExpressionOperator.EXISTS, ExpressionOperator.NOT_EXISTS):
        return f"{variable} {operator} (actual={actual_repr}{type_note}) -> {result}"
    return f"{variable} {operator} {condition.get('value')!r} (actual={actual_repr}{type_note}) -> {result}"


def describe_edge_expression(edge: dict, context_data: dict, variable_types: Optional[dict] = None) -> str:
    """Trace an edge's deterministic evaluation (variable/operator/expected vs actual) for routing logs."""
    condition_type = edge.get("condition_type")

    if condition_type == EdgeConditionType.UNCONDITIONAL:
        return "unconditional"
    if condition_type != EdgeConditionType.EXPRESSION:
        return f"{condition_type} (non-deterministic)"

    expression = edge.get("expression") or {}
    conditions = expression.get("conditions", [])
    if not conditions:
        return "expression (no conditions)"

    logic = expression.get("logic") or "and"
    separator = f" {logic.upper()} "
    return separator.join(describe_condition(c, context_data, variable_types) for c in conditions)
