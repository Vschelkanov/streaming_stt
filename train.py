from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer
from transformers import WhisperProcessor
from transformers import WhisperForConditionalGeneration

from dataset import get_common_voice_dataset
from datacollator import DataCollatorSpeechSeq2SeqWithPadding

import evaluate
from transformers import Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer

from functools import partial


def compute_metrics(pred, tokenizer, metric):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


def main():
    # model_checkpoint = 'muhtasham/whisper-medium-tg_tj'
    model_checkpoint = 'muhtasham/whisper-small-tg'

    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_checkpoint)
    tokenizer = WhisperTokenizer.from_pretrained(model_checkpoint, language='Tajik', task='transcribe')
    processor = WhisperProcessor.from_pretrained(model_checkpoint, language='Tajik', task='transcribe')
    model = WhisperForConditionalGeneration.from_pretrained(model_checkpoint)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    asr_datasets = get_common_voice_dataset(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer
    )

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    wer_metric = evaluate.load("wer")

    training_args = Seq2SeqTrainingArguments(
        output_dir="./models",  # change to a repo name of your choice
        per_device_train_batch_size=16,
        gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
        learning_rate=1e-5,
        warmup_steps=500,
        max_steps=4000,
        gradient_checkpointing=True,
        fp16=False,
        evaluation_strategy="steps",
        per_device_eval_batch_size=8,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=1000,
        eval_steps=1000,
        logging_steps=25,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=asr_datasets["train"],
        eval_dataset=asr_datasets["test"],
        data_collator=data_collator,
        compute_metrics=partial(compute_metrics, tokenizer=tokenizer, metric=wer_metric),
        tokenizer=processor.feature_extractor,
    )
    processor.save_pretrained(training_args.output_dir)

    trainer.train()


if __name__ == '__main__':
    main()