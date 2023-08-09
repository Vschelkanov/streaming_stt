from datasets import Audio
from datasets import load_dataset, DatasetDict
from functools import partial


def prepare_dataset(batch, feature_extractor, tokenizer):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch


def get_common_voice_dataset(feature_extractor, tokenizer):
    common_voice = DatasetDict()

    common_voice["train"] = load_dataset("google/fleurs", "tg_tj", split="train+validation", )
    common_voice["test"] = load_dataset("google/fleurs", "tg_tj", split="test")

    common_voice = common_voice.remove_columns(
        ['id', 'num_samples', 'path', 'transcription', 'gender', 'lang_id', 'language', 'lang_group_id']
    )
    common_voice = common_voice.rename_column('raw_transcription', 'sentence')
    common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))

    common_voice = common_voice.map(partial(prepare_dataset, feature_extractor=feature_extractor, tokenizer=tokenizer),
                                    remove_columns=common_voice.column_names["train"], num_proc=2)

    return common_voice


def main():
    pass


if __name__ == '__main__':
    main()