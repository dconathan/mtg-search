import argparse


parser = argparse.ArgumentParser()
parser.add_argument("command")
parser.add_argument("args", nargs="*")


def main():
    args = parser.parse_args()

    if args.command == "search":
        from mtg_search.api import search

        query = " ".join(args.args)
        print(f"query is: {query}")
        print()
        for c in search(query):
            print(c.name)
            print(c.text)
            print()
    if args.command == "index":
        from mtg_search.api import create_index

        create_index()
    else:
        print("unknown command")


if __name__ == "__main__":
    main()
