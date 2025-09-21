# audio_streamer.py

import tempfile
import numpy as np
import soundfile
import sounddevice as sd
import queue
import threading
import os


class AudioStreamer:
    """
    A class to handle continuous audio streaming and playback.
    """
    
    def __init__(self, sample_rate=48000):
        self.audio_queue = queue.Queue()
        self.audio_thread = None
        self.playing = False
        self.sample_rate = sample_rate
        self.audio_stream = None
        self.audio_data_buffer = []  # Continuous buffer for audio samples
        self.buffer_lock = threading.Lock()  # Thread safety for buffer
        
    def start(self):
        """Start the audio streaming system."""
        if self.audio_thread is None or not self.audio_thread.is_alive():
            self.playing = True
            self.audio_thread = threading.Thread(target=self._audio_processing_worker, daemon=True)
            self.audio_thread.start()
            self._start_streaming_output()
            
    def stop(self):
        """Stop the audio streaming system."""
        self.playing = False
        if self.audio_stream:
            self.audio_stream.stop()
            self.audio_stream.close()
        if self.audio_thread and self.audio_thread.is_alive():
            self.audio_thread.join(timeout=1.0)
    
    def _start_streaming_output(self):
        """Start a continuous streaming audio output."""
        def audio_callback(outdata, frames, time, status):
            """Callback for continuous audio output."""
            if status:
                print(f"Audio status: {status}")
            
            with self.buffer_lock:
                if len(self.audio_data_buffer) >= frames:
                    # We have enough data
                    chunk = self.audio_data_buffer[:frames]
                    self.audio_data_buffer = self.audio_data_buffer[frames:]
                    outdata[:] = np.array(chunk).reshape(-1, 1)
                elif len(self.audio_data_buffer) > 0:
                    # We have some data but not enough
                    chunk = self.audio_data_buffer[:]
                    self.audio_data_buffer = []
                    outdata[:len(chunk)] = np.array(chunk).reshape(-1, 1)
                    outdata[len(chunk):] = 0  # Pad with silence
                else:
                    # No data available, output silence
                    outdata.fill(0)
        
        self.audio_stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=1,
            callback=audio_callback,
            blocksize=1024,
            dtype=np.float32
        )
        self.audio_stream.start()
        print(f"Started streaming audio output at {self.sample_rate}Hz")
            
    def _audio_processing_worker(self):
        """Worker thread that processes audio chunks and feeds the continuous stream."""
        while self.playing:
            try:
                # Get audio chunk from queue with timeout
                audio_data = self.audio_queue.get(timeout=0.1)
                if audio_data is None:  # Sentinel value to stop
                    break
                    
                # Try to decode the audio data using soundfile
                # The audio chunks are likely encoded audio files (WAV, etc.)
                try:
                    # Write to temporary file and read back with soundfile
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                        temp_file.write(audio_data)
                        temp_file.flush()
                        
                    # Read the audio file using soundfile to get proper format
                    audio_array, actual_sample_rate = soundfile.read(temp_file.name)
                    
                    # Clean up temp file
                    os.unlink(temp_file.name)
                    
                    # Add audio data to the continuous buffer
                    with self.buffer_lock:
                        self.audio_data_buffer.extend(audio_array.tolist())
                    print(f"Added audio chunk: {len(audio_array)} samples at {actual_sample_rate}Hz to buffer")
                    
                except Exception as decode_error:
                    print(f"Failed to decode audio chunk: {decode_error}")
                    # Fallback: try as raw PCM data
                    try:
                        audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                        with self.buffer_lock:
                            self.audio_data_buffer.extend(audio_array.tolist())
                        print(f"Added raw PCM chunk: {len(audio_array)} samples to buffer")
                    except Exception as pcm_error:
                        print(f"Failed to process audio chunk: {pcm_error}")
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in audio processing worker: {e}")
                continue
                
    def add_audio_chunk(self, audio_data):
        """Add an audio chunk to the playback queue."""
        try:
            self.audio_queue.put(audio_data, block=False)
        except queue.Full:
            print("Audio queue is full, dropping chunk")
