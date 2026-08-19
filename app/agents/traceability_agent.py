"""
Traceability Agent.

Creates/updates the requirement <-> test case mapping for every requirement
touched in a pipeline run. Runs after validation so the matrix always
reflects the latest validated test cases, and marks prior mappings for a
node "stale" when the requirement itself was flagged stale by Version
Detection.
"""

from app.models import TestCase, Traceability, RequirementNode


def build_traceability(db, document_id: int, requirement_node_ids: list):
    for node_id in requirement_node_ids:
        req_node = (
            db.query(RequirementNode)
            .filter(RequirementNode.node_id == node_id, RequirementNode.document_id == document_id)
            .first()
        )

        test_cases = (
            db.query(TestCase)
            .filter(TestCase.requirement_node_id == node_id)
            .all()
        )

        mapped_status = "stale" if (req_node and req_node.is_stale) else "active"

        for tc in test_cases:
            existing = (
                db.query(Traceability)
                .filter(
                    Traceability.requirement_node_id == node_id,
                    Traceability.test_case_id == tc.test_case_id,
                )
                .first()
            )

            if existing:
                existing.document_id = document_id
                existing.status = mapped_status
            else:
                db.add(
                    Traceability(
                        document_id=document_id,
                        requirement_node_id=node_id,
                        test_case_id=tc.test_case_id,
                        status=mapped_status,
                    )
                )

    db.commit()
