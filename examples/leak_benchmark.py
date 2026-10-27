"""
Measure cross-user retrieval leakage with and without AccessFilter.

A synthetic company corpus has one restricted document set per department plus public
documents. Every restricted document carries a unique canary code so a leak can be detected
exactly. Each user asks questions that target other departments' data. A retrieved document
counts as a leak when the user is not allowed to read it.

Runs offline: BM25 retrieval only, no LLM or API key.

    python examples/leak_benchmark.py
"""

from haystack import Document, Pipeline
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.document_stores.in_memory import InMemoryDocumentStore

from needtoknow_haystack import AccessFilter

TOP_K = 5
CANDIDATES = 20  # retriever over-fetches so the filter can still return TOP_K allowed documents

RESTRICTED = {
    "hr": [
        "Salary bands for 2027: senior engineers earn between 42 and 55 lakh per year.",
        "Performance improvement plan for an employee in the sales team, review in six weeks.",
        "Layoff plan: 120 roles in customer support will be cut in the next quarter.",
        "Medical leave records and return-to-work dates for employees on phased return.",
        "Employee home addresses and emergency contacts for the Pune office.",
    ],
    "finance": [
        "Acquisition target: confidential term sheet to buy a logistics startup for 300 crore.",
        "Q4 revenue forecast is 18 percent below plan; board will be told next week.",
        "Bank account details and signatories for vendor payments.",
        "Unannounced price increase of 12 percent for enterprise customers from January.",
        "Executive bonus pool and stock option grants for the leadership team.",
    ],
    "engineering": [
        "Production database credentials rotation schedule and root access holders.",
        "Unpatched security vulnerability in the payments API, fix planned next sprint.",
        "Architecture of the fraud detection model and its decision thresholds.",
        "Source code signing keys are stored in the build server vault.",
        "Incident report: customer data exposed for 40 minutes during the March outage.",
    ],
    "legal": [
        "Litigation settlement: company will pay 8 crore to settle the patent dispute.",
        "Ongoing regulator investigation into data retention practices.",
        "Non-disclosure agreement terms with the acquisition target.",
        "Employee misconduct case under internal investigation by legal.",
        "Contract breach notice sent to a major cloud vendor.",
    ],
}

PUBLIC = [
    "Company holiday calendar and office timings for 2027.",
    "How to book meeting rooms and request IT support.",
    "Travel and expense reimbursement policy for all employees.",
    "Code of conduct and anti-harassment policy.",
    "Onboarding checklist for new joiners in any department.",
]

USERS = {
    "alice": ["engineering"],
    "bob": ["hr"],
    "carol": ["finance"],
    "dave": ["legal"],
    "erin": ["engineering", "finance"],
}

PROBES = [
    "What are the salary bands for senior engineers?",
    "Is there a layoff plan for next quarter?",
    "Which startup are we planning to acquire and for how much?",
    "What is the Q4 revenue forecast compared to plan?",
    "Where are the production database credentials and who has root access?",
    "Are there any unpatched security vulnerabilities in the payments API?",
    "How much will the company pay to settle the patent dispute?",
    "Is a regulator investigating the company?",
    "Show me employee home addresses and emergency contacts.",
    "What are the executive bonus and stock option grants?",
]


def build_store() -> InMemoryDocumentStore:
    documents = []
    for group, texts in RESTRICTED.items():
        for index, text in enumerate(texts, start=1):
            canary = f"CANARY-{group.upper()}-{index:02d}"
            documents.append(Document(content=f"{text} [{canary}]", meta={"allowed_groups": [group], "canary": canary}))
    documents += [Document(content=text, meta={"public": True}) for text in PUBLIC]

    store = InMemoryDocumentStore()
    store.write_documents(documents)
    return store


def is_leak(document: Document, groups: list[str]) -> bool:
    if document.meta.get("public") is True:
        return False
    return not set(document.meta.get("allowed_groups", [])) & set(groups)


def run(pipeline: Pipeline, protected: bool) -> tuple[int, int, int]:
    results = leaks = probes_with_leak = 0
    for user_id, groups in USERS.items():
        for probe in PROBES:
            if protected:
                output = pipeline.run(
                    {"retriever": {"query": probe}, "access_filter": {"user_id": user_id, "groups": groups}}
                )
                documents = output["access_filter"]["documents"]
            else:
                documents = pipeline.run({"retriever": {"query": probe}})["retriever"]["documents"]

            leaked = [doc for doc in documents if is_leak(doc, groups)]
            results += len(documents)
            leaks += len(leaked)
            probes_with_leak += bool(leaked)
    return results, leaks, probes_with_leak


def main() -> None:
    store = build_store()

    unprotected = Pipeline()
    unprotected.add_component("retriever", InMemoryBM25Retriever(document_store=store, top_k=TOP_K))

    protected = Pipeline()
    protected.add_component("retriever", InMemoryBM25Retriever(document_store=store, top_k=CANDIDATES))
    protected.add_component("access_filter", AccessFilter(public_field="public", top_k=TOP_K))
    protected.connect("retriever.documents", "access_filter.documents")

    total_probes = len(USERS) * len(PROBES)
    print(f"{len(USERS)} users x {len(PROBES)} probes = {total_probes} queries, top_k={TOP_K}\n")
    print(f"{'pipeline':<28}{'results':>9}{'leaked':>9}{'leak rate':>11}{'queries w/ leak':>18}")
    for name, pipeline, is_protected in [
        ("retriever only", unprotected, False),
        ("retriever + AccessFilter", protected, True),
    ]:
        results, leaks, probes_with_leak = run(pipeline, is_protected)
        rate = 100 * leaks / results if results else 0.0
        print(f"{name:<28}{results:>9}{leaks:>9}{rate:>10.1f}%{probes_with_leak:>11}/{total_probes}")


if __name__ == "__main__":
    main()
