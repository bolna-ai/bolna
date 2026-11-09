"""infer_type maps a JSON value to a Pydantic field type for json_to_pydantic_schema.

bool is a subclass of int in Python, so checking int before bool made the bool branch
unreachable and typed every boolean field as int in the generated schema.
"""

from bolna.helpers.utils import infer_type


def test_bool_is_inferred_as_bool_not_int():
    # Regression: int was checked first, so isinstance(True, int) matched and the
    # bool branch never ran.
    assert infer_type(True) == (bool, ...)
    assert infer_type(False) == (bool, ...)


def test_int_is_still_inferred_as_int():
    assert infer_type(42) == (int, ...)
    assert infer_type(0) == (int, ...)


def test_other_scalar_and_container_types_are_unchanged():
    assert infer_type(3.14) == (float, ...)
    assert infer_type("hello") == (str, ...)
    assert infer_type([1, 2]) == (list, ...)
    assert infer_type({"a": 1}) == (dict, ...)
