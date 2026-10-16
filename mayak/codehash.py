"""Отпечаток исходного кода по синтаксическому дереву без докстрингов."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os

from mayak.provenance import REPO_ROOT

DIGEST_CHARS = 16
_DEFINITIONS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
_MEMO: dict = {}


def _is_docstring(stmt):
    return (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant)
            and isinstance(stmt.value.value, str))


def _strip_docstrings(tree):
    """Убирает докстринги модуля, классов и функций на любой глубине вложенности."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, *_DEFINITIONS)) and node.body and \
                _is_docstring(node.body[0]):
            node.body = node.body[1:]
    return tree


def _canonical(value):
    """Дерево в виде вложенных списков без пустых полей и без номеров строк."""
    if isinstance(value, ast.AST):
        fields = {}
        for name in value._fields:
            item = getattr(value, name, None)
            if item is None or (isinstance(item, list) and not item):
                continue
            fields[name] = _canonical(item)
        return [type(value).__name__, fields]
    if isinstance(value, list):
        return [_canonical(v) for v in value]
    return repr(value)


def _digest(obj):
    blob = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:DIGEST_CHARS]


def _target_names(target):
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.Tuple, ast.List)):
        return set().union(*(_target_names(e) for e in target.elts))
    if isinstance(target, (ast.Starred, ast.Subscript, ast.Attribute)):
        return _target_names(target.value)
    return set()


def defined_names(stmt):
    """Имена, которые определяет или меняет инструкция верхнего уровня.

    Args:
        stmt: узел синтаксического дерева.

    Returns:
        Множество имён: имя функции или класса, цели присваивания. Для прочих
        инструкций - пустое множество.
    """
    if isinstance(stmt, _DEFINITIONS):
        return {stmt.name}
    if isinstance(stmt, ast.Assign):
        return set().union(*(_target_names(t) for t in stmt.targets))
    if isinstance(stmt, (ast.AnnAssign, ast.AugAssign)):
        return _target_names(stmt.target)
    return set()


def source_digest(source, names=None):
    """Отпечаток исходного текста.

    Args:
        source: исходный текст файла на Python.
        names: имена определений верхнего уровня, которые входят в отпечаток; None
            означает весь файл. Порядок имён не важен, берётся порядок в файле.

    Returns:
        Строка из шестнадцатеричных цифр.

    Raises:
        ValueError: какого-то из имён нет среди определений верхнего уровня.
        SyntaxError: текст не разбирается.
    """
    tree = _strip_docstrings(ast.parse(source))
    body = tree.body
    if names is not None:
        want = set(names)
        body = [stmt for stmt in tree.body if defined_names(stmt) & want]
        found = set().union(*(defined_names(stmt) for stmt in body))
        if want - found:
            raise ValueError(f"в исходнике нет определений верхнего уровня "
                             f"{sorted(want - found)}")
    return _digest([_canonical(stmt) for stmt in body])


def source_path(unit):
    """Путь к файлу единицы кода.

    Args:
        unit: имя модуля через точку или путь к файлу с расширением py, абсолютный
            либо от корня репозитория.

    Returns:
        Абсолютный путь к файлу.

    Raises:
        ModuleNotFoundError: модуль не найден или у него нет исходного файла.
    """
    if unit.endswith(".py"):
        return unit if os.path.isabs(unit) else os.path.join(REPO_ROOT, unit)
    spec = importlib.util.find_spec(unit)
    if spec is None or not spec.origin or not spec.origin.endswith(".py"):
        raise ModuleNotFoundError(f"нет исходного файла модуля {unit}")
    return spec.origin


def read_unit(unit):
    """Исходный текст единицы кода.

    Args:
        unit: имя модуля через точку или путь к файлу.

    Returns:
        Текст файла.
    """
    with open(source_path(unit), encoding="utf-8") as f:
        return f.read()


def unit_digest(unit, names=None):
    """Отпечаток одной единицы кода.

    Args:
        unit: имя модуля через точку или путь к файлу.
        names: имена определений верхнего уровня; None означает весь файл.

    Returns:
        Строка из шестнадцатеричных цифр.
    """
    text = read_unit(unit)
    key = (hashlib.sha256(text.encode("utf-8")).hexdigest(),
           None if names is None else tuple(sorted(names)))
    if key not in _MEMO:
        _MEMO[key] = source_digest(text, names)
    return _MEMO[key]


def code_digests(spec):
    """Отпечатки набора единиц кода.

    Args:
        spec: словарь: единица кода и кортеж имён определений, None означает весь файл.

    Returns:
        Словарь: единица кода и её отпечаток, по алфавиту единиц.
    """
    return {unit: unit_digest(unit, spec[unit]) for unit in sorted(spec)}


def combined_digest(digests):
    """Один отпечаток на весь набор.

    Args:
        digests: словарь отпечатков по единицам кода.

    Returns:
        Строка из шестнадцатеричных цифр.
    """
    return _digest(sorted(digests.items()))


__all__ = ["code_digests", "combined_digest", "defined_names", "read_unit", "source_digest",
           "source_path", "unit_digest"]
