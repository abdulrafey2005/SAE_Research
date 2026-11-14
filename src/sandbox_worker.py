
"""
Lightweight worker for executing generated MBPP code.

Input:
{
    "code": "...",
    "test_imports": "...",
    "test_list": [...]
}

Output:
{
    "passed": true/false,
    "error": "..."
}
"""

import contextlib
import io
import json
import sys
import traceback


def main():

    result = {
        "passed": False,
        "error": "",
    }

    try:

        # Read and parse input.
        payload = json.loads(sys.stdin.read())

        code = payload["code"]
        test_imports = payload.get("test_imports", "")
        test_list = payload.get("test_list", [])

        if not isinstance(code, str):
            raise TypeError("Generated code must be a string")

        if not isinstance(test_imports, str):
            raise TypeError("test_imports must be a string")

        if not isinstance(test_list, list):
            raise TypeError("test_list must be a list")

        # Separate namespace for execution.
        namespace = {
            "__name__": "__sandbox__"
        }

        output_buffer = io.StringIO()

        with (
            contextlib.redirect_stdout(output_buffer),
            contextlib.redirect_stderr(output_buffer),
        ):

            # Execute imports.
            if test_imports.strip():
                exec(test_imports, namespace)

            # Execute generated solution.
            exec(code, namespace)

            # Execute MBPP tests.
            for statement in test_list:

                exec(statement, namespace)

        result = {
            "passed": True,
            "error": "",
        }

    except BaseException as e:

        result = {
            "passed": False,
            "error": (
                f"{type(e).__name__}: {e}"
            )[:500],
        }

    # Always print one JSON response.
    sys.__stdout__.write(
        json.dumps(result) + "\n"
    )

    sys.__stdout__.flush()


if __name__ == "__main__":
    main()
