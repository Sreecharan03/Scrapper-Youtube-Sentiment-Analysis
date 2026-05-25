"""
app/db/repositories/base.py
============================
Generic base repository with common CRUD operations.

WHY THE REPOSITORY PATTERN:
  - Business logic (API endpoints, Celery tasks) should not contain raw
    MongoDB queries. If you need to change an index or query strategy,
    you change it in ONE place: the repository.
  - Each repository can be injected as a dependency in FastAPI, making
    it trivial to swap with an in-memory fake for unit tests.
  - Type annotations on return values prevent entire classes of runtime bugs.

USAGE:
  class VideoRepository(BaseRepository):
      collection_name = "videos"

  repo = VideoRepository(get_database())
  doc = await repo.find_one({"video_id": "abc123"})
"""

from typing import Any, Optional

from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorCollection, AsyncIOMotorDatabase

from app.core.exceptions import DatabaseOperationError
from app.core.logging import get_logger

logger = get_logger(__name__)


class BaseRepository:
    """
    Abstract base for all MongoDB repositories.

    Subclasses must set `collection_name` as a class variable.
    All methods are async and safe to call from FastAPI route handlers.
    """

    collection_name: str  # Subclasses must override this

    def __init__(self, database: AsyncIOMotorDatabase) -> None:
        if not hasattr(self, "collection_name") or not self.collection_name:
            raise NotImplementedError(
                f"{self.__class__.__name__} must define `collection_name`"
            )
        self._db = database
        self._collection: AsyncIOMotorCollection = database[self.collection_name]

    # ------------------------------------------------------------------ #
    # Read operations                                                      #
    # ------------------------------------------------------------------ #

    async def find_one(self, filter_: dict) -> Optional[dict]:
        """
        Find and return a single document matching the filter.

        Returns:
            The document as a dict, or None if not found.
        """
        try:
            result = await self._collection.find_one(filter_)
            return dict(result) if result else None
        except Exception as exc:
            raise DatabaseOperationError(
                f"find_one failed on {self.collection_name}",
                detail=str(exc),
            ) from exc

    async def find_by_id(self, doc_id: str) -> Optional[dict]:
        """Find a document by its MongoDB ObjectId string."""
        try:
            oid = ObjectId(doc_id)
        except Exception:
            return None   # Invalid ObjectId format — not found
        return await self.find_one({"_id": oid})

    async def find_many(
        self,
        filter_: dict,
        *,
        skip: int = 0,
        limit: int = 100,
        sort: Optional[list[tuple[str, int]]] = None,
    ) -> list[dict]:
        """
        Find multiple documents with pagination support.

        Args:
            filter_: MongoDB query filter
            skip: Number of documents to skip (for pagination)
            limit: Maximum documents to return (capped at 1000 for safety)
            sort: List of (field, direction) tuples, e.g. [("created_at", -1)]

        Returns:
            List of matching documents as dicts.
        """
        safe_limit = min(limit, 1000)  # Hard cap — never return unbounded results
        try:
            cursor = self._collection.find(filter_).skip(skip).limit(safe_limit)
            if sort:
                cursor = cursor.sort(sort)
            return [dict(doc) async for doc in cursor]
        except Exception as exc:
            raise DatabaseOperationError(
                f"find_many failed on {self.collection_name}",
                detail=str(exc),
            ) from exc

    async def count(self, filter_: dict) -> int:
        """Count documents matching the filter."""
        try:
            return await self._collection.count_documents(filter_)
        except Exception as exc:
            raise DatabaseOperationError(
                f"count failed on {self.collection_name}",
                detail=str(exc),
            ) from exc

    # ------------------------------------------------------------------ #
    # Write operations                                                     #
    # ------------------------------------------------------------------ #

    async def insert_one(self, document: dict) -> str:
        """
        Insert a single document.

        Returns:
            The inserted document's ObjectId as a string.
        """
        try:
            result = await self._collection.insert_one(document)
            inserted_id = str(result.inserted_id)
            logger.debug(
                "document_inserted",
                collection=self.collection_name,
                id=inserted_id,
            )
            return inserted_id
        except Exception as exc:
            raise DatabaseOperationError(
                f"insert_one failed on {self.collection_name}",
                detail=str(exc),
            ) from exc

    async def insert_many(self, documents: list[dict]) -> list[str]:
        """
        Bulk insert documents. More efficient than multiple insert_one calls.

        Returns:
            List of inserted ObjectId strings, in the same order as input.
        """
        if not documents:
            return []
        try:
            result = await self._collection.insert_many(documents, ordered=False)
            ids = [str(oid) for oid in result.inserted_ids]
            logger.debug(
                "documents_bulk_inserted",
                collection=self.collection_name,
                count=len(ids),
            )
            return ids
        except Exception as exc:
            raise DatabaseOperationError(
                f"insert_many failed on {self.collection_name}",
                detail=str(exc),
            ) from exc

    async def update_one(
        self,
        filter_: dict,
        update: dict,
        *,
        upsert: bool = False,
    ) -> int:
        """
        Update first matching document.

        Args:
            filter_: Query to identify the document
            update: MongoDB update operators (e.g. {"$set": {...}})
            upsert: If True, insert if not found

        Returns:
            Number of documents modified (0 or 1).
        """
        try:
            result = await self._collection.update_one(filter_, update, upsert=upsert)
            return result.modified_count
        except Exception as exc:
            raise DatabaseOperationError(
                f"update_one failed on {self.collection_name}",
                detail=str(exc),
            ) from exc

    async def delete_one(self, filter_: dict) -> int:
        """
        Delete first matching document.

        Returns:
            Number of documents deleted (0 or 1).
        """
        try:
            result = await self._collection.delete_one(filter_)
            return result.deleted_count
        except Exception as exc:
            raise DatabaseOperationError(
                f"delete_one failed on {self.collection_name}",
                detail=str(exc),
            ) from exc

    # ------------------------------------------------------------------ #
    # Utility                                                              #
    # ------------------------------------------------------------------ #

    async def exists(self, filter_: dict) -> bool:
        """Return True if at least one document matches the filter."""
        return await self.count(filter_) > 0
