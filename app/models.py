from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    Boolean,
    ForeignKey,
    DateTime
)
from sqlalchemy.orm import relationship, declarative_base
from datetime import datetime

Base = declarative_base()


# ==========================
# Document Table
# ==========================
class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    version = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Set by the Document Gate (CRAG) node
    is_relevant = Column(Boolean, default=True)
    relevance_reason = Column(Text, nullable=True)

    # "first" | "unchanged" | "changed" -- set by Version Detection
    version_status = Column(String, nullable=True)

    # Relationships
    nodes = relationship(
        "RequirementNode",
        back_populates="document",
        cascade="all, delete-orphan"
    )


# ==========================
# Requirement Nodes
# ==========================
class RequirementNode(Base):
    __tablename__ = "requirement_nodes"

    id = Column(Integer, primary_key=True, index=True)

    # Example: REQ-2.1.3
    node_id = Column(String, nullable=False, index=True)

    document_id = Column(
        Integer,
        ForeignKey("documents.id"),
        nullable=False
    )

    # Example: REQ-2
    parent_node_id = Column(String, nullable=True)

    number = Column(String)
    title = Column(String)

    level = Column(Integer)

    text = Column(Text)

    content_hash = Column(String)

    page = Column(Integer, nullable=True)

    # Set by Version Detection / Requirement Agent when compared to the
    # previous version of the same document
    is_stale = Column(Boolean, default=False)

    # Relationship
    document = relationship("Document", back_populates="nodes")


class TestCase(Base):
    __tablename__ = "test_cases"

    id = Column(Integer, primary_key=True, index=True)

    requirement_node_id = Column(
        String,
        ForeignKey("requirement_nodes.node_id"),
        nullable=False
    )

    test_case_id = Column(String)
    title = Column(String)
    preconditions = Column(Text)
    steps = Column(Text)
    expected_result = Column(Text)
    priority = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)


# ==========================
# Validation Results (Self-RAG)
# ==========================
class ValidationResult(Base):
    __tablename__ = "validation_results"

    id = Column(Integer, primary_key=True, index=True)

    requirement_node_id = Column(
        String,
        ForeignKey("requirement_nodes.node_id"),
        nullable=False
    )

    verdict = Column(String, nullable=False)  # "pass" | "fail"
    feedback = Column(Text, nullable=True)
    evidence_detail = Column(Text, nullable=True)

    # Comma-separated node_ids of other requirements retrieved as
    # cross-requirement evidence for this validation attempt.
    related_requirement_ids = Column(Text, nullable=True)

    attempt = Column(Integer, default=1)
    created_at = Column(DateTime, default=datetime.utcnow)


# ==========================
# Traceability (Requirement <-> Test Case)
# ==========================
class Traceability(Base):
    __tablename__ = "traceability"

    id = Column(Integer, primary_key=True, index=True)

    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    requirement_node_id = Column(String, nullable=False, index=True)
    test_case_id = Column(String, nullable=False, index=True)

    status = Column(String, default="active")  # "active" | "stale"
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
