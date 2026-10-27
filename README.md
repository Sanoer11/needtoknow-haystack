# needtoknow-haystack

**Permission-aware retrieval for [Haystack](https://haystack.deepset.ai/).** `AccessFilter` drops retrieved documents that the requesting user is not allowed to see, so a RAG pipeline never hands one user's restricted documents to another.

```
pipeline                      results   leaked  leak rate   queries w/ leak
retriever only                    250      145      58.0%         50/50
retriever + AccessFilter          240        0       0.0%          0/50
```
<sub>Output of [`examples/leak_benchmark.py`](examples/leak_benchmark.py) on Haystack 3.4.1: 5 users × 10 probing queries over a synthetic company corpus, BM25 top-5. A result is a leak when the user is not entitled to the document. Runs offline, no LLM or API key.</sub>

## Why

Most RAG pipelines retrieve from one shared index. Retrieval ranks by relevance, not by who is asking, so a question like *"What are the salary bands?"* returns HR documents to anyone who asks. Whatever the retriever returns ends up in the LLM prompt, and from there in the answer.

`AccessFilter` enforces document-level permissions inside the pipeline, between the retriever and the prompt.

## Installation

```bash
pip install git+https://github.com/Sanoer11/needtoknow-haystack.git
```

Works with Haystack 2.31+ and Haystack 3.x (tested on 2.31.0 and 3.4.1).

## Usage

Label each document with who may read it, then put `AccessFilter` after any retriever:

```python
from haystack import Document, Pipeline
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.document_stores.in_memory import InMemoryDocumentStore

from needtoknow_haystack import AccessFilter

store = InMemoryDocumentStore()
store.write_documents(
    [
        Document(content="Q3 salary bands for engineering", meta={"allowed_groups": ["hr"]}),
        Document(content="Engineering onboarding guide", meta={"allowed_groups": ["engineering"]}),
        Document(content="Company holiday calendar", meta={"public": True}),
        Document(content="Alice's performance review", meta={"allowed_users": ["alice"]}),
    ]
)

pipeline = Pipeline()
pipeline.add_component("retriever", InMemoryBM25Retriever(document_store=store, top_k=20))
pipeline.add_component("access_filter", AccessFilter(public_field="public", top_k=5))
pipeline.connect("retriever.documents", "access_filter.documents")

result = pipeline.run(
    {
        "retriever": {"query": "salary bands, onboarding guide, holiday calendar and performance review"},
        "access_filter": {"user_id": "bob", "groups": ["engineering"]},
    }
)

for doc in result["access_filter"]["documents"]:
    print(doc.content)
print("denied:", result["access_filter"]["denied_count"])
```

```
Engineering onboarding guide
Company holiday calendar
denied: 2
```

The retriever matched all four documents. Bob is in `engineering`, so he gets the onboarding guide and the public calendar, and the HR salary document and Alice's review are removed.

### How access is decided

A document is returned when **any** of these is true:

| Rule | Metadata (default key) | Example |
|---|---|---|
| The user is listed | `allowed_users` | `{"allowed_users": ["alice", "bob"]}` |
| One of the user's groups is listed | `allowed_groups` | `{"allowed_groups": ["hr"]}` |
| The document is public (opt-in via `public_field`) | e.g. `public` | `{"public": True}` |

Values can be a list or a single string. Matching is exact and case-sensitive.

### Secure defaults

- **Unlabeled documents are denied.** A document with none of the access fields is treated as restricted. Set `allow_unlabeled=True` only if unlabeled means public in your data.
- **An empty `user_id` raises `ValueError`** instead of silently returning nothing or everything.
- **Public means exactly `True`.** A value like `"yes"` does not make a document public.

### Parameters

| Parameter | Default | Description |
|---|---|---|
| `users_field` | `"allowed_users"` | Metadata key listing allowed user IDs |
| `groups_field` | `"allowed_groups"` | Metadata key listing allowed groups |
| `public_field` | `None` | Metadata key marking public documents (disabled when `None`) |
| `allow_unlabeled` | `False` | Whether documents without access metadata are readable by everyone |
| `top_k` | `None` | Maximum number of allowed documents to return |

`run(documents, user_id, groups=None)` returns `documents` (allowed, original order, trimmed to `top_k`) and `denied_count`.

### Over-fetch, then trim

Filtering happens after retrieval, so if the retriever returns only 5 documents and 3 are restricted, the user gets 2. Ask the retriever for more candidates (`top_k=20`) and let `AccessFilter(top_k=5)` trim the allowed ones. This is why the protected run in the benchmark returns 240 results instead of 250: for a few queries, fewer than 5 of the 20 candidates were allowed.

### Loading saved pipelines (Haystack 3)

Haystack 3 only deserializes components from allowlisted modules. Allow this package when loading a pipeline from YAML:

```python
Pipeline.loads(yaml_text, allowed_modules=["needtoknow_haystack"])
```

## Limitations

- **Your metadata is the source of truth.** The filter trusts `allowed_users` / `allowed_groups` written at indexing time. Sync them from your real permission system, and resolve `user_id` and `groups` from an authenticated session, never from the user's prompt.
- **Post-retrieval filtering.** Restricted documents are still searched, just never returned. For very large or highly restricted corpora, also use your document store's native filters to narrow the search.
- **Retrieval-level protection only.** It stops restricted documents reaching the prompt. It does not stop an LLM from repeating what an allowed document says, or from inferring facts by combining allowed documents.

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check . && ruff format --check .
mypy src
python examples/leak_benchmark.py
```

## Background

This component comes from [NeedToKnow](https://github.com/Sanoer11/needtoknow), a permission-aware RAG system. With 5 users × 10 adversarial probes (250 retrieved results), it measured a 43.2% leak rate without the permission lock and 0% with it.

## License

Apache-2.0
