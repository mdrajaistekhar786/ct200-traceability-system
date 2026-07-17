import pdfplumber
import re
import json
import hashlib


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


#Creating the Heading function
def is_heading(line):
    if not line["bold"]:
        return False

    return bool(re.match(r'^\d+(?:\.\d+)*\.?\s+', line["text"]))


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

            m = re.match(r'^(\d+(?:\.\d+)*)\.?\s+(.+)$', line["text"])

            number = m.group(1)
            title = m.group(2)

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
            ox0 <= ix0 and
            oy0 <= iy0 and
            ox1 >= ix1 and
            oy1 >= iy1
        )

    kept = []

    for table in tables:
        if any(
            other is not table and contains(other.bbox, table.bbox)
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
                    current["text"].append(" ".join(paragraph))
                    paragraph = []

                m = re.match(r'^(\d+(?:\.\d+)*)', line["text"])
                current = nodes[f"REQ-{m.group(1)}"]
                continue

            if current:
                paragraph.append(line["text"])

        if current and paragraph:
            current["text"].append(" ".join(paragraph))
            paragraph = []

    return tree




def attach_tables(pages, tree, nodes):

    for page in pages:

        headings = []

        # Collect headings with their Y positions
        for line in page["lines"]:
            if is_heading(line):
                m = re.match(r'^(\d+(?:\.\d+)*)', line["text"])
                if m:
                    headings.append({
                        "y": line["y"],
                        "node": nodes[f"REQ-{m.group(1)}"]
                    })

        # Attach each table to the nearest heading above it
        for table in page["tables"]:

            table_y = table.bbox[1]   # top Y of table

            target = None

            for h in headings:
                if h["y"] <= table_y:
                    target = h["node"]
                else:
                    break

            if target:
                target["tables"].append(table.extract())

    return tree




def compute_hashes(tree):

    def traverse(node):

        content = {
            "text": node["text"],
            "tables": node["tables"]
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


def parse_manual(pdf_path):

    with pdfplumber.open(pdf_path) as pdf:

        pages = []

        for p in pdf.pages:
            pages.append({
                "page": p.page_number,
                "lines": reconstruct_lines(p),
                "tables": dedupe_tables(p.find_tables())
            })

    tree, nodes = build_requirement_tree(pages)

    tree = attach_content(pages, tree, nodes)

    tree = attach_tables(pages, tree, nodes)

    tree = compute_hashes(tree)

    return tree