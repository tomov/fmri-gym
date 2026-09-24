import threading
import av
from fractions import Fraction
import numpy as np
import pathlib
import queue
import time
import psutil

class EpisodeRecorder():

    def __init__(
            self,
            path:pathlib.Path,
            frame_size:tuple[int, int],
            video_codec='libx265',
            video_stream_pix_fmt = "yuv444p",
            video_stream_options = {"x265-params": "lossless=1:preset=ultrafast", "preset": "ultrafast"},
            audio_layout:str | None='stereo',
            audio_sample_rate:int=44100,
            audio_codec='flac',
            audio_sample_format='s16',
    ) -> None:
        self.container = av.open(path, mode="w")
        self.time_base = Fraction(1, 65535)
        self.last_pts = -1000
        self.video_stream = self.container.add_stream(video_codec, time_base=self.time_base)
        self.video_stream.width = frame_size[1]
        self.video_stream.height = frame_size[0]
        self.video_stream.options = video_stream_options
        self.video_stream.pix_fmt = video_stream_pix_fmt

        if audio_layout is not None:
            self.audio_stream = self.container.add_stream(audio_codec, rate=audio_sample_rate)
            self.audio_stream.layout = audio_layout
            self.audio_stream.codec_context.format = audio_sample_format


        self.steps = queue.Queue()
        self.lock = threading.Lock()
        self._stop_event = threading.Event()


    def _run(self)-> None:
        thread_id = threading.get_native_id()
        p = psutil.Process(thread_id)
        p.nice(19)
        while True:
            if self.steps.empty():
                if self._stop_event.is_set():
                    break
                time.sleep(.01)
            else:
                with self.lock:
                    timestamp, frame, audio = self.steps.get()
                    self._step(timestamp, frame, audio)
        for packet in self.video_stream.encode():
            self.container.mux(packet)
        if hasattr(self, 'audio_stream'):
            for packet in self.audio_stream.encode():
                self.container.mux(packet)
        self.container.close()

    def step(self, timestamp, frame, audio) -> None:
        audio = audio.pcm if audio is not None else None
        with self.lock:
            self.steps.put((timestamp, frame, audio))

    def _step(self, timestamp, frame, audio) -> None:
        v_frame = av.VideoFrame.from_ndarray(frame, channel_last=True)
        pts = int(timestamp / self.time_base)
        pts = max(pts, self.last_pts + 1)
        v_frame.pts = pts
        self.last_pts = pts
        for packet in self.video_stream.encode(v_frame):
            self.container.mux(packet)

        if hasattr(self, 'audio_stream') and audio is not None:
            a_frame = av.AudioFrame(
                samples=audio.shape[0],
                format=self.audio_stream.codec_context.format,
                layout=self.audio_stream.layout,
            )
            a_frame.planes[0].update(audio.tobytes())
            #a_frame.planes[1].update(audio[:,1].tobytes())
            a_frame.pts = pts
            for packet in self.audio_stream.encode(a_frame):
                self.container.mux(packet)


    def start(self) -> None:
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self._stop_event.set()

    def join(self) -> None:
        self.thread.join()
