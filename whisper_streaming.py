from whisper_online import *
import numpy as np
import threading
import time
from queue import Queue
import pyaudio
import webrtcvad
from functools import lru_cache


class Live:
    exit_event = threading.Event()
    device_name = 'default'
    asr_model = FasterWhisperASR('tg', './models/ctranslate2-whisper')
    online_processor = OnlineASRProcessor(asr_model, None)

    def stop(self):
        Live.exit_event.set()
        self.asr_input_queue.put('close')
        print('ASR Stopped')

    def start(self):
        self.asr_output_queue = Queue()
        self.asr_input_queue = Queue()

        self.asr_process_thread = threading.Thread(
            target=self.asr_process,
            args=(self.asr_input_queue, self.asr_output_queue)
        )
        self.asr_process_thread.start()

        self.vad_process_thread = threading.Thread(
            target=self.vad_process,
            args=(self.device_name, self.asr_input_queue)
        )
        self.vad_process_thread.start()

    def vad_process(self, device_name, asr_input_queue):
        vad = webrtcvad.Vad()
        vad.set_mode(2)

        audio = pyaudio.PyAudio()
        frame_duration = 20
        rate = 16000
        chunk = int(rate * frame_duration / 1000)

        microphones = self.list_microphones(audio)

        selected_input_device_id = self.get_input_device_id(device_name, microphones)

        stream = audio.open(input_device_index=selected_input_device_id,
                            format=pyaudio.paInt16,
                            channels=1,
                            rate=rate,
                            input=True,
                            frames_per_buffer=chunk)

        frames = b''
        while True:
            if self.exit_event.is_set():
                break

            frame = stream.read(chunk, exception_on_overflow=False)
            is_speech = vad.is_speech(frame, rate)
            if is_speech:
                frames += frame
            else:
                if len(frames) > 1:
                    asr_input_queue.put(frames)
                frames = b''
        stream.stop_stream()
        stream.close()
        audio.terminate()

    def asr_process(self, in_queue, output_queue):
        print("\nlistening to your voice\n")
        while True:
            audio_frames = in_queue.get()
            if audio_frames == "close":
                break

            buffer = np.frombuffer(
                audio_frames, dtype=np.int16) / 32767

            start_t = time.perf_counter()

            self.online_processor.insert_audio_chunk(buffer)
            output = self.online_processor.process_iter()

            self.output_transcript(output, start_t)

            text = output[2]

            inference_time = time.perf_counter() - start_t
            sample_length = len(buffer) / 16000

            if text != "":
                output_queue.put([sample_length, inference_time, text])

    @staticmethod
    def get_input_device_id(device_name, microphones):
        for device in microphones:
            if device_name in device[1]:
                return device[0]

    @staticmethod
    def list_microphones(pyaudio_instance):
        info = pyaudio_instance.get_host_api_info_by_index(0)
        num_devices = info.get('deviceCount')

        result = []
        for i in range(0, num_devices):
            if (pyaudio_instance.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
                name = pyaudio_instance.get_device_info_by_host_api_device_index(0, i).get('name')
                result += [[i, name]]
        return result

    def get_last_text(self):
        return self.asr_output_queue.get()

    @staticmethod
    def output_transcript(o, start, now=None):
        # output format in stdout is like:
        # 4186.3606 0 1720 Takhle to je
        # - the first three words are:
        #    - emission time from beginning of processing, in milliseconds
        #    - beg and end timestamp of the text segment, as estimated by Whisper model.
        #    The timestamps are not accurate, but they're useful anyway
        # - the next words: segment transcript
        if now is None:
            now = time.time() - start
        if o[0] is not None:
            print("%1.4f %1.0f %1.0f %s" % (now * 1000, o[0] * 1000, o[1] * 1000, o[2]), file=sys.stderr, flush=True)
            print("%1.4f %1.0f %1.0f %s" % (now * 1000, o[0] * 1000, o[1] * 1000, o[2]), flush=True)
        else:
            print(o, file=sys.stderr, flush=True)


def main():
    print("Live ASR")
    asr_object = Live()
    asr_object.start()

    try:
        while True:
            output = asr_object.get_last_text()
            print(output)

    except KeyboardInterrupt:
        asr_object.stop()

        output = asr_object.online_processor.finish()
        asr_object.output_transcript(output, 0, 0)

        exit()


if __name__ == '__main__':
    main()
