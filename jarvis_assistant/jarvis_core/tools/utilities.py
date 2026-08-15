"""
tools/utilities.py
Small utility tools carried over from the original JARVIS script: time,
date, weather, jokes, and a *safe* arithmetic calculator. The calculator
replaces the legacy raw eval() with an AST-based evaluator that only
permits numeric literals and arithmetic operators.
"""
from __future__ import annotations

import ast
import datetime
import operator

import requests

from ..config import settings

_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv, ast.USub: operator.neg, ast.UAdd: operator.pos,
}


def _safe_eval(node: ast.AST):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("Expression contains disallowed operations.")


def calculate(expression: str) -> str:
    """Safely evaluate an arithmetic expression (no code execution, unlike eval())."""
    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree.body)
        return f"The answer is {result}"
    except Exception:
        return "I couldn't calculate that. Please provide a plain arithmetic expression (e.g. (12 + 3) * 2)."


def get_time() -> str:
    """Return the current local time."""
    return f"The current time is {datetime.datetime.now().strftime('%I:%M %p')}"


def get_date() -> str:
    """Return today's date."""
    return f"Today's date is {datetime.datetime.now().strftime('%B %d, %Y')}"


def get_weather(city: str = "London") -> str:
    """Fetch current weather for a city via OpenWeatherMap (requires OPENWEATHER_API_KEY)."""
    api_key = settings.openweather_api_key
    if not api_key:
        return "Weather API key not configured (set OPENWEATHER_API_KEY in .env)."
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
        response = requests.get(url, timeout=10)
        data = response.json()
        if response.status_code == 200:
            temp = data["main"]["temp"]
            condition = data["weather"][0]["description"]
            return f"In {city}, it's {temp}\u00b0C with {condition}"
        return f"Unable to fetch weather for {city} (status {response.status_code})."
    except Exception as exc:
        return f"Error getting weather: {exc}"


def tell_joke() -> str:
    """Fetch a random joke."""
    try:
        response = requests.get("https://official-joke-api.appspot.com/random_joke", timeout=10)
        if response.status_code == 200:
            joke = response.json()
            return f"{joke['setup']} ... {joke['punchline']}"
        return "I couldn't find a joke right now."
    except Exception:
        return "Error fetching joke."
