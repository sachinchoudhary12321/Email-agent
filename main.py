"""CLI entry: python main.py [--limit N] [--all]"""

import argparse

from agent import EmailAgent


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize useful inbox content.")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument(
        "--all", action="store_true", help="Include read mail, not only unread."
    )
    parser.add_argument("--mailbox", default="INBOX")
    args = parser.parse_args()

    agent = EmailAgent()
    emails, summary, insights = agent.run(
        limit=args.limit,
        unread_only=not args.all,
        mailbox=args.mailbox,
    )
    print(f"\nFetched {len(emails)} email(s).\n")
    print(summary)
    if insights:
        print("\n--- Per-email insights ---\n")
        print(insights)


if __name__ == "__main__":
    main()
