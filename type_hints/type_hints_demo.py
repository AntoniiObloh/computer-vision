"""Python type hints — runnable notes.

Covers annotations on variables and functions, unions and `Optional`,
overrides in subclasses, `NewType`, dataclasses and generics.

Two examples here are DELIBERATELY wrong. They are marked with `# mypy:`
comments quoting what a type checker reports. The file still runs: CPython
does not enforce annotations at runtime, which is exactly the point — the
errors are only visible to a checker.

    Run:    .venv/bin/python type_hints/type_hints_demo.py
    Check:  mypy type_hints/type_hints_demo.py      (pip install mypy first)

Needs Python 3.12+: the `def f[T](...)` generic syntax is PEP 695.
"""

import random
import typing as t
from dataclasses import dataclass
from typing import NewType

# ---------------------------------------------------------------------------
# 1. Variables and functions
# ---------------------------------------------------------------------------

# An annotated variable. Nothing checks this at runtime — `x = "five"` would
# run just as happily; only a checker objects.
x: int = 5


def display_bio(name: str, age: int, job: t.Optional[str] = None) -> None:
    """`t.Optional[str]` is the older spelling of `str | None` — same meaning.

    Both styles are kept in this file on purpose, so the two are easy to
    compare. In new code the `|` form is preferred.
    """
    job_text = f"I am a {job}" if job else "I am unemployed"
    print(f"My name is {name}, I am {age} years old, and {job_text}")


def add_numbers(n1: int | None = None, n2: int = 0) -> int:
    # Watch out: `not n1` is also True for 0, so add_numbers(0, 5) returns 0
    # instead of 5. The annotation says the special case is None, so
    # `if n1 is None` is what actually matches the signature.
    if not n1:
        return 0
    return n1 + n2


# ---------------------------------------------------------------------------
# 2. Overrides: a subclass must accept everything the parent accepts
# ---------------------------------------------------------------------------


class Parent:
    def capitalize(self, text: str) -> str:
        return text.upper()


class Child(Parent):
    # mypy: Argument 1 of "capitalize" is incompatible with supertype "Parent";
    #       supertype defines the argument type as "str"
    #
    # The Liskov substitution principle: anywhere a Parent works, a Child must
    # work too. The parent takes str, this override takes only int, so passing
    # a Child where a Parent is expected breaks. An override may WIDEN a
    # parameter type (str -> str | int), never narrow it.
    def capitalize(self, text: int) -> str:
        return str(text).upper()


# ---------------------------------------------------------------------------
# 3. NewType: distinct names for a shape that is otherwise identical
# ---------------------------------------------------------------------------

# Both are plain tuple[int, int, int] at runtime, with no wrapper object and
# no cost. To a checker they are two separate types, so an RGB cannot be
# passed where an HSL is expected — which is the whole reason to declare them.
RGB = NewType("RGB", tuple[int, int, int])
HSL = NewType("HSL", tuple[int, int, int])


# ---------------------------------------------------------------------------
# 4. Dataclass and a factory function
# ---------------------------------------------------------------------------


@dataclass
class User:
    first_name: str
    last_name: str
    email: str
    age: int | None
    fav_color: RGB | HSL | None


def create_user(
    first_name: str,
    last_name: str,
    age: int | None = None,
    fav_color: RGB | HSL | None = None,
) -> User:
    email = f"{first_name.lower()}_{last_name.lower()}@example.com"
    return User(
        first_name=first_name,
        last_name=last_name,
        email=email,
        age=age,
        fav_color=fav_color,
    )


# ---------------------------------------------------------------------------
# 5. Generics (PEP 695)
# ---------------------------------------------------------------------------


def random_choice[T](items: list[T]) -> T:
    """`T` binds to whatever the list holds: list[User] in, User out.

    Without `T` the return type would have to be `object`, and every caller
    would need a cast to get a usable type back.
    """
    return random.choice(items)


if __name__ == "__main__":
    # mypy: Argument 2 to "display_bio" has incompatible type "str";
    #       expected "int"
    # Prints fine anyway — the annotation is not enforced at runtime.
    display_bio("Antonii", "23")

    print(add_numbers())

    user1 = create_user("Antonii", "Oblog", age=38, fav_color=RGB((100, 100, 100)))
    user2 = create_user("Antonii2", "Oblog2", age=32, fav_color=RGB((100, 100, 100)))
    users = [user1, user2]

    # list[User] -> User
    random_user = random_choice(users)
    print(random_user)

    # The same function on list[str] -> str. One definition, T rebinds.
    emails = [user.email for user in users]
    random_email = random_choice(emails)
    print(random_email)
