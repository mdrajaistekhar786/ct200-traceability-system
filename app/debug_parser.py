from app.parser import parse_manual

paths = [
    r"D:\ct200-traceability-system\data\ct200_manual.pdf",
    r"D:\ct200-traceability-system\data\ct200_manual_v2.pdf",
]

target_ids = {
    "REQ-1.1",
    "REQ-3.1",
    "REQ-3.3",
    "REQ-6.1",
    "REQ-8.1",
}


def walk(nodes):
    result = []

    for node in nodes:
        result.append(node)
        result.extend(walk(node.get("children", [])))

    return result


for path in paths:
    print("\n" + "=" * 80)
    print("PDF:", path)
    print("=" * 80)

    tree = parse_manual(path)

    for node in walk(tree):
        if node["id"] in target_ids:
            print("\nID:", node["id"])
            print("TITLE:", node["title"])
            print("HASH:", node["content_hash"])
            print("TEXT:", node["text"])
            print("TABLES:", node["tables"])