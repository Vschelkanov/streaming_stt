import soundfile as sf
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor


class Wave2Vec2Inference:
    def __init__(self, model_name):
        self.device = "cpu"

        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name)
        self.model.to(self.device)

    def buffer_to_text(self, audio_buffer):
        if len(audio_buffer) == 0:
            return ""

        inputs = self.processor(torch.tensor(audio_buffer), sampling_rate=16_000, return_tensors="pt", padding=True)

        with torch.no_grad():
            logits = self.model(
                inputs.input_values.to(self.device),
                attention_mask=inputs.attention_mask.to(self.device)
            ).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)[0]
        confidence = self.confidence_score(logits, predicted_ids)

        return transcription, confidence

    def confidence_score(self, logits, predicted_ids):
        scores = torch.nn.functional.softmax(logits, dim=-1)
        pred_scores = scores.gather(-1, predicted_ids.unsqueeze(-1))[:, :, 0]
        mask = torch.logical_and(
            predicted_ids.not_equal(self.processor.tokenizer.word_delimiter_token_id),
            predicted_ids.not_equal(self.processor.tokenizer.pad_token_id))

        character_scores = pred_scores.masked_select(mask)
        total_average = torch.sum(character_scores) / len(character_scores)
        return total_average

    def file_to_text(self, filename):
        audio_input, samplerate = sf.read(filename)
        assert samplerate == 16000
        return self.buffer_to_text(audio_input)


if __name__ == "__main__":
    print("Model test")
    asr = Wave2Vec2Inference('jonatasgrosman/wav2vec2-large-xlsr-53-english')
    text = asr.file_to_text('test.wav')
    print(text)
