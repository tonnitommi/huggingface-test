"""Huggingface models locally."""

from transformers import pipeline

def minimal_task():

    nlp = pipeline(
        "document-question-answering",
        model="impira/layoutlm-document-qa",
    )

    r = nlp(
        "content/invoice.png",
        "What is the recipient's name?",
    )

    print(r)


if __name__ == "__main__":
    minimal_task()
