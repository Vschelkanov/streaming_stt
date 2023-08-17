import sys
import faster_whisper
import numpy as np
import nltk
import argparse
from functools import lru_cache
import librosa
import time

SAMPLING_RATE = 16000


@lru_cache
def load_audio(file_name):
    audio, _ = librosa.load(file_name, sr=16000)
    return audio


def load_audio_chunk(file_name, beg, end):
    audio = load_audio(file_name)
    beg_s = int(beg * 16000)
    end_s = int(end * 16000)
    return audio[beg_s:end_s]


class ASRWithFasterWhisper:
    def __init__(self, model_dir):
        self.separator = ''
        self.asr_language = 'tg'
        self.model = self.load_model(model_dir)
        self.transcribe_args = {}

    @staticmethod
    def load_model(model_dir=None):
        if model_dir is not None:
            print(f'Loading whisper model from model_dir {model_dir}.', file=sys.stderr)
            model_size_or_path = model_dir
        else:
            raise ValueError('model_dir parameter must be set')

        model = faster_whisper.WhisperModel(model_size_or_path, device='cpu', compute_type='int8')
        return model

    def transcribe(self, audio, init_prompt=''):
        segments, info = self.model.transcribe(
            audio,
            language=self.asr_language,
            initial_prompt=init_prompt,
            beam_size=5,
            word_timestamps=True,
            condition_on_previous_text=True,
            **self.transcribe_args
        )
        return list(segments)

    @staticmethod
    def ts_words(segments):
        outputs = []
        for segment in segments:
            for word in segment.words:
                word_text = word.word
                info_tuple = (word.start, word.end, word_text)
                outputs.append(info_tuple)
        return outputs

    @staticmethod
    def segments_end_ts(res):
        return [s.end for s in res]

    def use_vad(self):
        self.transcribe_args["vad_filter"] = True


class BufferHypothesis:
    def __init__(self):
        self.committed_in_buffer = []
        self.buffer = []
        self.new = []

        self.last_committed_time = 0
        self.last_committed_word = None

    def insert(self, new, offset):
        new = [(a + offset, b + offset, t) for a, b, t in new]
        self.new = [(a, b, t) for a, b, t in new if a > self.last_committed_time - 0.1]

        if len(self.new) >= 1:
            a, b, t = self.new[0]
            if abs(a - self.last_committed_time) < 1:
                if self.committed_in_buffer:
                    # it's going to search for 1, 2, ..., 5 consecutive words (n-grams)
                    # that are identical in committed and new. If they are, they're dropped.

                    cn = len(self.committed_in_buffer)
                    nn = len(self.new)
                    for i in range(1, min(min(cn, nn), 5) + 1):  # 5 is the maximum
                        c = " ".join([self.committed_in_buffer[-j][2] for j in range(1, i + 1)][::-1])
                        tail = " ".join(self.new[j - 1][2] for j in range(1, i + 1))
                        if c == tail:
                            print("removing last", i, "words:", file=sys.stderr)
                            for j in range(i):
                                print("\t", self.new.pop(0), file=sys.stderr)
                            break

    def flush(self):
        # returns committed chunk = the longest common prefix of 2 last inserts.

        commit = []
        while self.new:
            na, nb, nt = self.new[0]

            if len(self.buffer) == 0:
                break

            if nt == self.buffer[0][2]:
                commit.append((na, nb, nt))
                self.last_committed_word = nt
                self.last_committed_time = nb
                self.buffer.pop(0)
                self.new.pop(0)
            else:
                break
        self.buffer = self.new
        self.new = []
        self.committed_in_buffer.extend(commit)
        return commit

    def pop_committed(self, time):
        while self.committed_in_buffer and self.committed_in_buffer[0][1] <= time:
            self.committed_in_buffer.pop(0)

    def complete(self):
        return self.buffer


class ASRStreamingProcessor:
    SAMPLING_RATE = 16000

    def __init__(self, asr):
        self.asr = asr
        self.init()

    def init(self):
        """run this when starting or restarting processing"""
        self.audio_buffer = np.array([], dtype=np.float32)
        self.buffer_time_offset = 0

        self.transcript_buffer = BufferHypothesis()
        self.committed = []
        self.last_chunked_at = 0
        self.silence_iters = 0

    def insert_audio_chunk(self, audio):
        self.audio_buffer = np.append(self.audio_buffer, audio)

    def prompt(self):
        """Returns a tuple: (prompt, context),
        - "prompt" is a 200-character suffix of committed text that is inside the scrolled away part of audio buffer.
        - "context" is the committed text that is inside the audio buffer. It is transcribed again and skipped.
        It is returned only for debugging and logging reasons.
        """
        k = max(0, len(self.committed) - 1)
        while k > 0 and self.committed[k - 1][1] > self.last_chunked_at:
            k -= 1

        p = self.committed[:k]
        p = [t for _, _, t in p]
        prompt = []
        l = 0
        while p and l < 200:  # 200 characters prompt size
            x = p.pop(-1)
            l += len(x) + 1
            prompt.append(x)
        non_prompt = self.committed[k:]
        return self.asr.separator.join(prompt[::-1]), self.asr.separator.join(t for _, _, t in non_prompt)

    def process_iter(self):
        """Runs on the current audio buffer.
        Returns: a tuple (beg_timestamp, end_timestamp, "text"), or (None, None, "").
        The non-empty text is confirmed (committed) partial transcript.
        """

        prompt, non_prompt = self.prompt()
        print('PROMPT:', prompt, file=sys.stderr)
        print('CONTEXT:', non_prompt, file=sys.stderr)

        print(
            f"Transcribing {len(self.audio_buffer) / self.SAMPLING_RATE:2.2f} seconds "
            f"from {self.buffer_time_offset:2.2f}",
            file=sys.stderr
        )

        result = self.asr.transcribe(
            self.audio_buffer,
            init_prompt=prompt
        )

        # transform to [(beg,end,"word1"), ...]
        tsw = self.asr.ts_words(result)

        self.transcript_buffer.insert(tsw, self.buffer_time_offset)
        output = self.transcript_buffer.flush()
        self.committed.extend(output)

        print("COMPLETE:", self.to_flush(output), file=sys.stderr, flush=True)
        print("INCOMPLETE:", self.to_flush(self.transcript_buffer.complete()), file=sys.stderr, flush=True)

        # there is a newly confirmed text we trim all the completed sentences from the audio buffer
        if output:
            self.chunk_completed_sentence()

        # if the audio buffer is longer than 30s, trim it...
        if len(self.audio_buffer) / self.SAMPLING_RATE > 30:
            # ...on the last completed segment (labeled by Whisper)
            self.chunk_completed_segment(result)
            print(f"Chunking because of len", file=sys.stderr)
            # self.chunk_at(t)

        print(f"Len of buffer now: {len(self.audio_buffer) / self.SAMPLING_RATE:2.2f}", file=sys.stderr)
        return self.to_flush(output)

    def chunk_completed_sentence(self):
        if self.committed == []:
            return

        print(self.committed, file=sys.stderr)
        sentences = self.words_to_sentences(self.committed)
        for s in sentences:
            print("\t\tSENTENCE:", s, file=sys.stderr)

        if len(sentences) < 2:
            return

        while len(sentences) > 2:
            sentences.pop(0)

        # we will continue with audio processing at this timestamp
        chunk_at = sentences[-2][1]

        print(f"--- Sentence chunked at {chunk_at:2.2f}", file=sys.stderr)
        self.chunk_at(chunk_at)

    def chunk_completed_segment(self, res):
        if self.committed == []:
            return

        ends = self.asr.segments_end_ts(res)

        t = self.committed[-1][1]

        if len(ends) > 1:

            e = ends[-2] + self.buffer_time_offset
            while len(ends) > 2 and e > t:
                ends.pop(-1)
                e = ends[-2] + self.buffer_time_offset
            if e <= t:
                print(f"--- Segment chunked at {e:2.2f}", file=sys.stderr)
                self.chunk_at(e)
            else:
                print(f"--- Last segment not within commited area", file=sys.stderr)
        else:
            print(f"--- Not enough segments to chunk", file=sys.stderr)

    def chunk_at(self, time):
        # trims the hypothesis and audio buffer at "time
        self.transcript_buffer.pop_committed(time)
        cut_seconds = time - self.buffer_time_offset
        self.audio_buffer = self.audio_buffer[int(cut_seconds) * self.SAMPLING_RATE:]
        self.buffer_time_offset = time
        self.last_chunked_at = time

    @staticmethod
    def words_to_sentences(words):
        """Uses self.tokenizer for sentence segmentation of words.
        Returns: [(beg,end,"sentence 1"),...]
        """
        c_words = [w for w in words]
        t = " ".join(o[2] for o in c_words)

        # ???
        s = nltk.word_tokenize(t)
        out = []
        while s:
            beg = None
            end = None
            sent = s.pop(0).strip()
            f_sent = sent
            while c_words:
                b, e, w = c_words.pop(0)
                if beg is None and sent.startswith(w):
                    beg = b
                elif end is None and sent == w:
                    end = e
                    out.append((beg, end, f_sent))
                    break
                sent = sent[len(w):].strip()
        return out

    def finish(self):
        """Flush the incomplete text when the whole processing ends.
        Returns: the same format as self.process_iter()
        """
        o = self.transcript_buffer.complete()
        f = self.to_flush(o)
        print("Last, non committed:", f, file=sys.stderr)
        return f

    def to_flush(self, sentences, sep=None, offset=0, ):
        # concatenates the timestamped words or sentences into one sequence that is flushed in one line
        # sentences: [(beg1, end1, "sentence1"), ...] or [] if empty
        # return: (beg1,end-of-last-sentence,"concatenation of sentences") or (None, None, "") if empty
        if sep is None:
            sep = self.asr.separator
        t = sep.join(s[2] for s in sentences)
        if len(sentences) == 0:
            b = None
            e = None
        else:
            b = offset + sentences[0][0]
            e = offset + sentences[-1][1]
        return b, e, t


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('audio_path', type=str,
                        help="Filename of 16kHz mono channel wav, on which live streaming is simulated.")

    parser.add_argument('--min-chunk-size', type=float, default=1.0,
                        help='Minimum audio chunk size in seconds. It waits up to this time to do processing. '
                             'If the processing takes shorter time, it waits, otherwise it processes '
                             'the whole segment that was received by this time.')

    parser.add_argument('--model_dir', type=str, default='./models/ctranslate2-whisper',
                        help="Directory where Whisper model.bin and other files are saved. ")

    parser.add_argument('--task', type=str, default='transcribe', choices=["transcribe", "translate"],
                        help="Transcribe or translate.")
    parser.add_argument('--start_at', type=float, default=0.0,
                        help='Start processing audio at this time.')

    parser.add_argument('--offline', action="store_true", default=False,
                        help='Offline mode.')

    parser.add_argument('--vad', action="store_true", default=False,
                        help='Use VAD = voice activity detection, with the default parameters.')
    args = parser.parse_args()

    return args


def main():
    args = get_args()

    t = time.time()
    asr = ASRWithFasterWhisper(
        model_dir=args.model_dir
    )

    e = time.time()
    print(f"Model loaded in {round(e - t, 2)} seconds.", file=sys.stderr)

    if args.vad:
        print("setting VAD filter", file=sys.stderr)
        asr.use_vad()

    min_chunk = args.min_chunk_size
    online_processor = ASRStreamingProcessor(asr)

    audio_path = args.audio_path
    duration = len(load_audio(audio_path)) / SAMPLING_RATE
    print("Audio duration is: %2.2f seconds" % duration, file=sys.stderr)

    # load the audio into the LRU cache before we start the timer
    audio_chunk = load_audio_chunk(audio_path, 0, 1)

    # warm up the ASR, because the very first transcribe takes much more time than the other
    asr.transcribe(audio_chunk)

    beg = args.start_at
    start = time.time() - beg

    def output_transcript(out, now_t=None):
        # output format in stdout is like:
        # 4186.3606 0 1720 Hello to je
        # - the first three words are:
        #    - emission time from beginning of processing, in milliseconds
        #    - beg and end timestamp of the text segment, as estimated by Whisper model.
        #    The timestamps are not accurate, but they're useful anyway
        # - the next words: segment transcript
        if now_t is None:
            now_t = time.time() - start
        if out[0] is not None:
            print("%1.4f %1.0f %1.0f %s" % (now_t * 1000, out[0] * 1000, out[1] * 1000, out[2]),
                  file=sys.stderr, flush=True)
            print("%1.4f %1.0f %1.0f %s" % (now_t * 1000, out[0] * 1000, out[1] * 1000, out[2]),
                  flush=True)
        else:
            print(out, file=sys.stderr, flush=True)

    if args.offline:  # offline mode processing
        whole_audio = load_audio(audio_path)
        online_processor.insert_audio_chunk(whole_audio)

        try:
            output = online_processor.process_iter()
        except AssertionError:
            print("Assertion error", file=sys.stderr)
            pass
        else:
            output_transcript(output)
        now = None
    else:  # online mode
        end = 0
        while True:
            now = time.time() - start
            if now < end + min_chunk:
                time.sleep(min_chunk + end - now)
            end = time.time() - start
            print(f'Loading audio: {beg} : {end}')
            audio_chunk = load_audio_chunk(audio_path, beg, end)
            beg = end

            online_processor.insert_audio_chunk(audio_chunk)

            try:
                out = online_processor.process_iter()
            except AssertionError:
                print("Assertion error", file=sys.stderr)
                pass
            else:
                output_transcript(out)
            now = time.time() - start

            print(f"# Last processed {end:.2f} s, now is {now:.2f}, the latency is {now - end:.2f}", file=sys.stderr,
                  flush=True)

            if end >= duration:
                break
        now = None

    out = online_processor.finish()
    output_transcript(out, now_t=now)



if __name__ == '__main__':
    main()