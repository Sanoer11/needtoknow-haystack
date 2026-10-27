import inspect

import pytest
from haystack import Document, Pipeline
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.document_stores.in_memory import InMemoryDocumentStore

from needtoknow_haystack import AccessFilter


def _contents(result: dict) -> list[str]:
    return [doc.content for doc in result["documents"]]


class TestInit:
    def test_defaults(self):
        access_filter = AccessFilter()

        assert access_filter.users_field == "allowed_users"
        assert access_filter.groups_field == "allowed_groups"
        assert access_filter.public_field is None
        assert access_filter.allow_unlabeled is False
        assert access_filter.top_k is None

    @pytest.mark.parametrize("top_k", [0, -1])
    def test_rejects_non_positive_top_k(self, top_k):
        with pytest.raises(ValueError, match="top_k must be greater than 0"):
            AccessFilter(top_k=top_k)

    def test_rejects_same_users_and_groups_field(self):
        with pytest.raises(ValueError, match="must be different"):
            AccessFilter(users_field="acl", groups_field="acl")


class TestRun:
    def test_allows_listed_user(self):
        docs = [Document(content="mine", meta={"allowed_users": ["alice", "bob"]})]

        result = AccessFilter().run(documents=docs, user_id="alice")

        assert _contents(result) == ["mine"]
        assert result["denied_count"] == 0

    def test_denies_unlisted_user(self):
        docs = [Document(content="bob only", meta={"allowed_users": ["bob"]})]

        result = AccessFilter().run(documents=docs, user_id="alice")

        assert result["documents"] == []
        assert result["denied_count"] == 1

    def test_allows_member_of_listed_group(self):
        docs = [Document(content="hr doc", meta={"allowed_groups": ["hr"]})]

        result = AccessFilter().run(documents=docs, user_id="alice", groups=["engineering", "hr"])

        assert _contents(result) == ["hr doc"]

    def test_denies_when_no_group_matches(self):
        docs = [Document(content="hr doc", meta={"allowed_groups": ["hr"]})]

        result = AccessFilter().run(documents=docs, user_id="alice", groups=["engineering"])

        assert result["documents"] == []

    def test_accepts_single_string_acl_values(self):
        docs = [
            Document(content="user string", meta={"allowed_users": "alice"}),
            Document(content="group string", meta={"allowed_groups": "hr"}),
        ]

        result = AccessFilter().run(documents=docs, user_id="alice", groups=["hr"])

        assert _contents(result) == ["user string", "group string"]

    def test_user_ids_are_matched_exactly(self):
        docs = [Document(content="case", meta={"allowed_users": ["Alice"]})]

        result = AccessFilter().run(documents=docs, user_id="alice")

        assert result["documents"] == []

    def test_denies_unlabeled_documents_by_default(self):
        docs = [Document(content="no acl")]

        result = AccessFilter().run(documents=docs, user_id="alice")

        assert result["documents"] == []
        assert result["denied_count"] == 1

    def test_allow_unlabeled_opt_in(self):
        docs = [Document(content="no acl"), Document(content="bob only", meta={"allowed_users": ["bob"]})]

        result = AccessFilter(allow_unlabeled=True).run(documents=docs, user_id="alice")

        assert _contents(result) == ["no acl"]
        assert result["denied_count"] == 1

    def test_public_field_allows_everyone(self):
        docs = [
            Document(content="public", meta={"public": True}),
            Document(content="truthy string is not public", meta={"public": "yes"}),
        ]

        result = AccessFilter(public_field="public").run(documents=docs, user_id="alice")

        assert _contents(result) == ["public"]

    def test_public_field_ignored_when_not_configured(self):
        docs = [Document(content="public", meta={"public": True})]

        result = AccessFilter().run(documents=docs, user_id="alice")

        assert result["documents"] == []

    def test_custom_field_names(self):
        docs = [Document(content="custom", meta={"readers": ["alice"]})]

        result = AccessFilter(users_field="readers", groups_field="teams").run(documents=docs, user_id="alice")

        assert _contents(result) == ["custom"]

    def test_keeps_order_scores_and_trims_to_top_k(self):
        docs = [
            Document(content="a", score=0.9, meta={"allowed_users": ["alice"]}),
            Document(content="blocked", score=0.8, meta={"allowed_users": ["bob"]}),
            Document(content="b", score=0.7, meta={"allowed_users": ["alice"]}),
            Document(content="c", score=0.6, meta={"allowed_users": ["alice"]}),
        ]

        result = AccessFilter(top_k=2).run(documents=docs, user_id="alice")

        assert _contents(result) == ["a", "b"]
        assert [doc.score for doc in result["documents"]] == [0.9, 0.7]
        assert result["denied_count"] == 1

    @pytest.mark.parametrize("user_id", ["", "   "])
    def test_rejects_empty_user_id(self, user_id):
        with pytest.raises(ValueError, match="user_id must be a non-empty string"):
            AccessFilter().run(documents=[], user_id=user_id)

    def test_empty_input(self):
        result = AccessFilter().run(documents=[], user_id="alice")

        assert result == {"documents": [], "denied_count": 0}


class TestSerialization:
    def test_round_trip(self):
        original = AccessFilter(
            users_field="readers", groups_field="teams", public_field="public", allow_unlabeled=True, top_k=3
        )

        data = original.to_dict()
        restored = AccessFilter.from_dict(data)

        assert data["init_parameters"] == {
            "users_field": "readers",
            "groups_field": "teams",
            "public_field": "public",
            "allow_unlabeled": True,
            "top_k": 3,
        }
        assert restored.to_dict() == data


class TestPipeline:
    def test_filters_retriever_output_per_user(self):
        store = InMemoryDocumentStore()
        store.write_documents(
            [
                Document(content="Q3 salary bands for engineering", meta={"allowed_groups": ["hr"]}),
                Document(content="Engineering onboarding guide", meta={"allowed_groups": ["engineering"]}),
                Document(content="Company holiday calendar", meta={"public": True}),
            ]
        )
        pipeline = Pipeline()
        pipeline.add_component("retriever", InMemoryBM25Retriever(document_store=store, top_k=10))
        pipeline.add_component("access_filter", AccessFilter(public_field="public"))
        pipeline.connect("retriever.documents", "access_filter.documents")

        def run_for(groups):
            result = pipeline.run(
                {
                    "retriever": {"query": "engineering salary calendar"},
                    "access_filter": {"user_id": "alice", "groups": groups},
                }
            )
            return set(_contents(result["access_filter"]))

        assert run_for(["engineering"]) == {"Engineering onboarding guide", "Company holiday calendar"}
        assert run_for(["hr"]) == {"Q3 salary bands for engineering", "Company holiday calendar"}

    def test_pipeline_serialization_round_trip(self):
        pipeline = Pipeline()
        pipeline.add_component("access_filter", AccessFilter(public_field="public", top_k=4))

        # Haystack 3 only deserializes components from allowlisted modules; Haystack 2 has no allowlist.
        kwargs = {}
        if "allowed_modules" in inspect.signature(Pipeline.loads).parameters:
            kwargs["allowed_modules"] = ["needtoknow_haystack"]
        restored = Pipeline.loads(pipeline.dumps(), **kwargs)

        assert restored.get_component("access_filter").to_dict() == pipeline.get_component("access_filter").to_dict()
