import pdfplumber
import re
import json
import hashlib
import unicodedata


def reconstruct_lines(page, y_tol=2):
    chars = sorted(page.chars, key=lambda c: c["top"])
    clusters = []

    for ch in chars:
        if clusters and abs(ch["top"] - clusters[-1][0]) <= y_tol:
            clusters[-1][1].append(ch)
        else:
            clusters.append((ch["top"], [ch]))

    lines = []

    for y, group in clusters:
        group = sorted(group, key=lambda c: c["x0"])
        text = "".join(c["text"] for c in group).strip()

        if not text:
            continue

        lines.append({
            "text": text,
            "y": y,
            "bold": any("Bold" in c["fontname"] for c in group),
            "size": max(c["size"] for c in group)
        })

    return sorted(lines, key=lambda x: x["y"])


def detect_title(lines):
    title = []

    for line in lines:
        if line["bold"] and line["size"] >= 20:
            title.append(line["text"])
        elif title:
            break

    return " ".join(title)


def is_heading(line):
    if not line["bold"]:
        return False

    return bool(
        re.match(
            r"^\d+(?:\.\d+)*\.?\s+",
            line["text"]
        )
    )


def create_node(number, title, page, y):
    return {
        "id": f"REQ-{number}",
        "number": number,
        "title": title,
        "parent": None,
        "source": {
            "page": page,
            "y": y
        },
        "text": [],
        "tables": [],
        "children": [],
        "test_cases": [],
        "version": "v1",
        "content_hash": ""
    }


def build_requirement_tree(pages):
    tree = []
    stack = []
    nodes = {}

    for page in pages:
        for line in page["lines"]:

            if not is_heading(line):
                continue

            match = re.match(
                r"^(\d+(?:\.\d+)*)\.?\s+(.+)$",
                line["text"]
            )

            if not match:
                continue

            number = match.group(1)
            title = match.group(2)

            node = create_node(
                number,
                title,
                page["page"],
                line["y"]
            )

            nodes[node["id"]] = node

            level = number.count(".") + 1

            while len(stack) >= level:
                stack.pop()

            if stack:
                node["parent"] = stack[-1]["id"]
                stack[-1]["children"].append(node)
            else:
                tree.append(node)

            stack.append(node)

    return tree, nodes


def dedupe_tables(tables):
    def contains(outer, inner):
        ox0, oy0, ox1, oy1 = outer
        ix0, iy0, ix1, iy1 = inner

        return (
            ox0 <= ix0
            and oy0 <= iy0
            and ox1 >= ix1
            and oy1 >= iy1
        )

    kept = []

    for table in tables:
        if any(
            other is not table
            and contains(other.bbox, table.bbox)
            for other in tables
        ):
            continue

        kept.append(table)

    return kept


def attach_content(pages, tree, nodes):
    current = None
    paragraph = []

    for page in pages:
        for line in page["lines"]:

            if is_heading(line):

                if current and paragraph:
                    current["text"].append(
                        " ".join(paragraph)
                    )
                    paragraph = []

                match = re.match(
                    r"^(\d+(?:\.\d+)*)",
                    line["text"]
                )

                if match:
                    current = nodes[
                        f"REQ-{match.group(1)}"
                    ]

                continue

            if current:
                paragraph.append(line["text"])

        if current and paragraph:
            current["text"].append(
                " ".join(paragraph)
            )
            paragraph = []

    return tree


def attach_tables(pages, tree, nodes):

    for page in pages:
        headings = []

        for line in page["lines"]:

            if is_heading(line):

                match = re.match(
                    r"^(\d+(?:\.\d+)*)",
                    line["text"]
                )

                if match:
                    headings.append({
                        "y": line["y"],
                        "node": nodes[
                            f"REQ-{match.group(1)}"
                        ]
                    })

        for table in page["tables"]:

            table_y = table.bbox[1]
            target = None

            for heading in headings:
                if heading["y"] <= table_y:
                    target = heading["node"]
                else:
                    break

            if target:
                target["tables"].append(
                    table.extract()
                )

    return tree


def normalize_text(value):
    """
    Convert extracted PDF text into a stable representation
    before hashing.

    Handles:
    - Unicode ligatures
    - different whitespace
    - page-break fragmentation
    - leading/trailing whitespace
    """

    if value is None:
        return ""

    value = str(value)

    # Convert characters such as ﬁ -> fi
    value = unicodedata.normalize("NFKC", value)

    # Normalize all whitespace
    value = re.sub(r"\s+", " ", value)

    return value.strip()


def normalize_table(value):
    """
    Recursively normalize table contents so that harmless
    whitespace/Unicode differences do not change the hash.
    """

    if isinstance(value, list):
        return [
            normalize_table(item)
            for item in value
        ]

    if isinstance(value, tuple):
        return [
            normalize_table(item)
            for item in value
        ]

    if isinstance(value, dict):
        return {
            key: normalize_table(val)
            for key, val in value.items()
        }

    if value is None:
        return ""

    return normalize_text(value)


def compute_hashes(tree):

    def traverse(node):

        normalized_text = normalize_text(
            " ".join(node.get("text", []))
        )

        normalized_tables = normalize_table(
            node.get("tables", [])
        )

        content = {
            "text": normalized_text,
            "tables": normalized_tables
        }

        serialized = json.dumps(
            content,
            sort_keys=True,
            ensure_ascii=False
        )

        node["content_hash"] = hashlib.sha256(
            serialized.encode("utf-8")
        ).hexdigest()

        for child in node["children"]:
            traverse(child)

    for root in tree:
        traverse(root)

    return tree


def sort_requirement_tree(tree):

    def number_key(node):
        return tuple(
            int(part)
            for part in node["number"].split(".")
        )

    tree.sort(key=number_key)

    for node in tree:
        if node.get("children"):
            sort_requirement_tree(
                node["children"]
            )

    return tree


def parse_manual(pdf_path):

    with pdfplumber.open(pdf_path) as pdf:

        pages = []

        for page in pdf.pages:
            pages.append({
                "page": page.page_number,
                "lines": reconstruct_lines(page),
                "tables": dedupe_tables(
                    page.find_tables()
                )
            })

    tree, nodes = build_requirement_tree(pages)

    tree = attach_content(
        pages,
        tree,
        nodes
    )

    tree = attach_tables(
        pages,
        tree,
        nodes
    )

    # Fix physical PDF ordering such as 3.4 appearing before 3.3
    tree = sort_requirement_tree(tree)

    # Create stable hashes after text normalization
    tree = compute_hashes(tree)

    return tree