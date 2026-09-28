"""Answer a PD question by calling one API tool.

    python -m agent "Calculate the default probability for business B1001."
"""

from __future__ import annotations

import argparse

from agent.run import answer


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the business PD agent")
    parser.add_argument("question", nargs="+", help="Question in plain language")
    args = parser.parse_args()
    print(answer(" ".join(args.question)))


if __name__ == "__main__":
    main()
