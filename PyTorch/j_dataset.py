import datasets

datasets.Features(
    {
        "id": datasets.Value("string"),
        "s1": datasets.Value("string"),
        "s2": datasets.Value("string"),
        "question": datasets.Value("string"),
        "answers": datasets.Sequence(
            {
                "text": datasets.Value("string"),
                "answer_start": datasets.Value("int32"),
            }
        ),
    }
)