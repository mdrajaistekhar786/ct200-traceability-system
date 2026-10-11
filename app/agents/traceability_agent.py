"""
Traceability Agent.

Creates/updates the requirement <-> test case mapping for every requirement
touched in a pipeline run.

Runs after validation so the matrix reflects only validated test cases.
"""

from datetime import datetime

from app.database import (
    requirements_collection,
    test_cases_collection,
    traceability_collection,
)


def build_traceability(
    db,
    document_id: str,
    requirement_node_ids: list,
):
    """
    Create or update requirement -> test case mappings in MongoDB.
    """

    for node_id in requirement_node_ids:

        # ---------------------------------------------------------------
        # Find the requirement in the current document version
        # ---------------------------------------------------------------

        req_node = requirements_collection.find_one(
            {
                "document_id": document_id,
                "node_id": node_id,
            }
        )

        # ---------------------------------------------------------------
        # Find test cases generated for this requirement AND
        # this specific document version.
        # ---------------------------------------------------------------

        test_cases = test_cases_collection.find(
            {
                "document_id": document_id,
                "requirement_node_id": node_id,
            }
        )

        mapped_status = (
            "stale"
            if (
                req_node
                and req_node.get("is_stale", False)
            )
            else "active"
        )

        for tc in test_cases:

            test_case_id = tc.get(
                "test_case_id"
            )

            # -----------------------------------------------------------
            # Check whether mapping already exists
            # -----------------------------------------------------------

            existing = traceability_collection.find_one(
                {
                    "document_id": document_id,
                    "requirement_node_id": node_id,
                    "test_case_id": test_case_id,
                }
            )

            if existing:

                traceability_collection.update_one(
                    {
                        "_id": existing["_id"]
                    },
                    {
                        "$set": {
                            "status": mapped_status,
                            "updated_at": datetime.utcnow(),
                        }
                    },
                )

            else:

                traceability_collection.insert_one(
                    {
                        "document_id": document_id,
                        "requirement_node_id": node_id,
                        "test_case_id": test_case_id,
                        "status": mapped_status,
                        "updated_at": datetime.utcnow(),
                    }
                )