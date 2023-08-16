from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer
from transformers import WhisperProcessor
from transformers import WhisperForConditionalGeneration
from dataset import prep_asr_test_dataset, get_assignment_dataset

import torch
from functools import partial
import evaluate


def map_to_pred(batch, processor, model):
    audio = batch["audio"]
    input_features = processor(
        audio["array"],
        sampling_rate=audio["sampling_rate"],
        return_tensors="pt").input_features

    batch["reference"] = processor.tokenizer._normalize(batch['transcription'])

    with torch.no_grad():
        predicted_ids = model.generate(input_features.to("mps"))[0]

    transcription = processor.decode(predicted_ids)
    batch["prediction"] = processor.tokenizer._normalize(transcription)

    return batch


def main():
    model_checkpoint = './models/checkpoint-300'

    processor = WhisperProcessor.from_pretrained(model_checkpoint, language='Tajik', task='transcribe')
    model = WhisperForConditionalGeneration.from_pretrained(model_checkpoint)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.to("mps")

    test_dataset = get_assignment_dataset()['test']

    wer = evaluate.load("wer")

    results = test_dataset.map(partial(map_to_pred, processor=processor, model=model))
    wer_score = 100 * wer.compute(references=results["reference"], predictions=results["prediction"])
    print('Test Dataset WER: ', wer_score)


if __name__ == '__main__':
    main()