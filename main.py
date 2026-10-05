import os

from chunking import recursive_chunking
from embedding import embed
from vector_indexing import load_index, normalize, save_index, search

INDEX_PATH = "bank_index"

# build the index once, then reuse the saved one so we don't pay for embeddings again
if os.path.exists(f"{INDEX_PATH}.npy"):
    vectors, texts = load_index(INDEX_PATH)
    print(f"Loaded saved index: {len(texts)} chunks")
else:
    with open("corpus.txt", encoding="utf-8") as f:
        texts = recursive_chunking(f.read())
    vectors = normalize(embed(texts))
    save_index(INDEX_PATH, vectors, texts)
    print(f"Built and saved index: {len(texts)} chunks")

# each question and the exact text the correct chunk must contain
test_questions = [
    ("How much did the bank earn last year?", "NPR 4.2 billion"),
    ("How much money did customers keep in the bank?", "NPR 312 billion"),
    ("What share of loans went bad?", "2.7 percent"),
    ("Is the bank's capital strong enough?", "13.1 percent"),
    ("How many people use the phone app?", "two million"),
    ("What will shareholders receive?", "10 percent"),
    ("How many new employees joined?", "300 graduates"),
    ("What are the plans for clean energy lending?", "5 percent of its loan portfolio"),
    ("How fast did housing loans grow?", "18 percent"),
    ("How does the bank stop card fraud?", "fraud monitoring system"),
]

questions = [q for q, _ in test_questions]
question_vectors = embed(questions)   # one API call for all questions

hits = 0
for (question, answer), vector in zip(test_questions, question_vectors):
    results = search(vector, vectors, texts, top_k=3)
    found = answer in results[0][1]
    hits += found
    print(f"\n{'PASS' if found else 'FAIL'}  {question}")
    for score, text in results:
        print(f"   {score:.3f}  {text[:80]}...")

print(f"\nTop result contained the answer for {hits}/{len(test_questions)} questions")