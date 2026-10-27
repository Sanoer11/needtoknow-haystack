"""Permission-aware document filtering for Haystack pipelines."""

from collections.abc import Iterable
from typing import Any

from haystack import Document, component, default_from_dict, default_to_dict, logging

logger = logging.getLogger(__name__)


def _as_set(value: Any) -> set[str]:
    """Normalise an ACL metadata value (a single string or an iterable of strings) to a set."""
    if value is None:
        return set()
    if isinstance(value, str):
        return {value}
    if isinstance(value, Iterable):
        return {str(item) for item in value}
    return {str(value)}


@component
class AccessFilter:
    """
    Drops retrieved documents that the requesting user is not allowed to see.

    Place it after any retriever. Each document declares who may read it in its metadata,
    and the filter keeps a document only if one of these holds:

    - the user's ID is listed in `meta[users_field]`,
    - one of the user's groups is listed in `meta[groups_field]`,
    - `public_field` is configured and `meta[public_field]` is `True`.

    The filter fails closed: a document that carries no access-control metadata is denied
    unless `allow_unlabeled=True`, and an empty `user_id` is rejected.

    Because filtering happens after retrieval, ask the retriever for more candidates than you
    need (for example `top_k=20`) and let this component trim the allowed ones with `top_k`.

    ### Usage example

    ```python
    from haystack import Document, Pipeline
    from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
    from haystack.document_stores.in_memory import InMemoryDocumentStore

    from needtoknow_haystack import AccessFilter

    store = InMemoryDocumentStore()
    store.write_documents([
        Document(content="Q3 salary bands", meta={"allowed_groups": ["hr"]}),
        Document(content="Office wifi password policy", meta={"public": True}),
    ])

    pipeline = Pipeline()
    pipeline.add_component("retriever", InMemoryBM25Retriever(document_store=store, top_k=20))
    pipeline.add_component("access_filter", AccessFilter(public_field="public", top_k=5))
    pipeline.connect("retriever.documents", "access_filter.documents")

    result = pipeline.run({
        "retriever": {"query": "salary bands"},
        "access_filter": {"user_id": "alice", "groups": ["engineering"]},
    })
    # alice is not in "hr", so the salary document is filtered out.
    print([doc.content for doc in result["access_filter"]["documents"]])
    ```
    """

    def __init__(
        self,
        users_field: str = "allowed_users",
        groups_field: str = "allowed_groups",
        public_field: str | None = None,
        allow_unlabeled: bool = False,
        top_k: int | None = None,
    ) -> None:
        """
        Create an AccessFilter.

        :param users_field:
            Metadata key listing the user IDs allowed to read a document.
        :param groups_field:
            Metadata key listing the groups allowed to read a document.
        :param public_field:
            Optional metadata key that marks a document as readable by everyone when its value is `True`.
            Disabled when `None`.
        :param allow_unlabeled:
            Whether documents with none of the access-control fields are readable by everyone.
            Defaults to `False` so that unlabeled documents are denied (fail closed).
        :param top_k:
            Maximum number of allowed documents to return, keeping the retriever's order.
            Must be `None` or greater than 0.
        :raises ValueError:
            If `top_k` is not `None` and is less than or equal to 0, or if `users_field` and
            `groups_field` are the same key.
        """
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be greater than 0.")
        if users_field == groups_field:
            raise ValueError("users_field and groups_field must be different metadata keys.")

        self.users_field = users_field
        self.groups_field = groups_field
        self.public_field = public_field
        self.allow_unlabeled = allow_unlabeled
        self.top_k = top_k

    def _is_labeled(self, document: Document) -> bool:
        fields = [self.users_field, self.groups_field]
        if self.public_field is not None:
            fields.append(self.public_field)
        return any(field in document.meta for field in fields)

    def _is_allowed(self, document: Document, user_id: str, groups: set[str]) -> bool:
        if not self._is_labeled(document):
            return self.allow_unlabeled
        if self.public_field is not None and document.meta.get(self.public_field) is True:
            return True
        if user_id in _as_set(document.meta.get(self.users_field)):
            return True
        return bool(groups & _as_set(document.meta.get(self.groups_field)))

    @component.output_types(documents=list[Document], denied_count=int)
    def run(self, documents: list[Document], user_id: str, groups: list[str] | None = None) -> dict[str, Any]:
        """
        Keep only the documents that `user_id` (or one of their `groups`) may read.

        :param documents:
            Documents returned by a retriever.
        :param user_id:
            ID of the user the documents are being retrieved for.
        :param groups:
            Groups the user belongs to.
        :returns:
            A dictionary with:
            - `documents`: The allowed documents, in their original order, trimmed to `top_k`.
            - `denied_count`: How many documents were removed because the user may not read them.
        :raises ValueError:
            If `user_id` is empty.
        """
        if not user_id or not user_id.strip():
            raise ValueError("user_id must be a non-empty string; refusing to filter for an unknown user.")

        group_set = _as_set(groups)
        allowed = [doc for doc in documents if self._is_allowed(doc, user_id, group_set)]
        denied_count = len(documents) - len(allowed)

        if denied_count:
            logger.debug(
                "AccessFilter removed {denied_count} of {total} documents for user {user_id}",
                denied_count=denied_count,
                total=len(documents),
                user_id=user_id,
            )

        if self.top_k is not None:
            allowed = allowed[: self.top_k]

        return {"documents": allowed, "denied_count": denied_count}

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize this component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            users_field=self.users_field,
            groups_field=self.groups_field,
            public_field=self.public_field,
            allow_unlabeled=self.allow_unlabeled,
            top_k=self.top_k,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AccessFilter":
        """
        Deserialize the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
        """
        return default_from_dict(cls, data)
