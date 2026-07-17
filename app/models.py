from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
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

    # Relationship
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