from app.internal.vector_store import search_hits

for question in ["what is a flu?", "What is a cold?", "What is alzheimer?"]:
    print("=" * 20, question)
    for index, hit in enumerate(search_hits(question), start=1):
        print("---", index, hit.source, hit.page, "len", len(hit.text))
        print(hit.text)
        print()
