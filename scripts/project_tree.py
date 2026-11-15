
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "project_tree.txt"


def generate_tree(root):
    lines = [f"{root.name}/"]

    def walk(directory, prefix=""):
        entries = sorted(
            directory.iterdir(),
            key=lambda path: (path.is_file(), path.name.lower())
        )

        for index, entry in enumerate(entries):
            last = index == len(entries) - 1
            connector = "└── " if last else "├── "
            suffix = "/" if entry.is_dir() else ""
            lines.append(f"{prefix}{connector}{entry.name}{suffix}")

            if entry.is_dir() and not entry.is_symlink():
                extension = "    " if last else "│   "
                walk(entry, prefix + extension)

    walk(root)
    return "\n".join(lines) + "\n"


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(generate_tree(ROOT), encoding="utf-8")
    print(f"Full project tree saved to: {OUTPUT}")


if __name__ == "__main__":
    main()
