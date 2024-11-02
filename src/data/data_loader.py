from datasets import load_dataset


class DataLoader:
    def __init__(self, dataset_name, tokenizer, sample_size=1000):
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.sample_size = sample_size

    def load_data(self):
        dataset = load_dataset(self.dataset_name, split="all")
        dataset = dataset.shuffle(seed=65).select(range(self.sample_size))
        return dataset

    def format_chat_template(self, row):
        row_json = [{"role": "user", "content": row["input"]},
                    {"role": "assistant", "content": row["output"]}]
        row["text"] = self.tokenizer.apply_chat_template(row_json, tokenize=False)
        return row

    def preprocess(self, dataset):
        dataset = dataset.map(self.format_chat_template, num_proc=4)
        return dataset.train_test_split(test_size=0.1)
