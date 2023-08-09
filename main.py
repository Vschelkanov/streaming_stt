from datasets import load_dataset, DatasetDict


def main():
    common_voice = DatasetDict()

    common_voice["train"] = load_dataset("google/fleurs", "tg_tj", split="train+validation")
    common_voice["test"] = load_dataset("google/fleurs", "tg_tj", split="test")

    print(common_voice)


if __name__ == '__main__':
    main()