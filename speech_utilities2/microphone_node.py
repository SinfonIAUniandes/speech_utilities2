import rclpy
from rclpy.node import Node
import numpy as np
import threading
import time
import os

# Import the executor and callback group classes
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup

# Message and service types
from naoqi_bridge_msgs.msg import AudioBuffer
from speech_msgs2.srv import SpeechToText, RecordAudio, SetTranscriptionMode
from speech_msgs2.msg import Transcription
from naoqi_utilities_msgs.msg import LedParameters
from naoqi_utilities_msgs.srv import SetVolume, GetVolume

from std_srvs.srv import SetBool

from .logic.audio.audio_processing import save_recording
from .logic.audio.voice_activity_detection import detect_speech
from .logic.audio.speech_2_text import transcribe

# Realtime STT imports
import struct
from RealtimeSTT import AudioToTextRecorder


class MicrophoneNode(Node):
    """
    Capture robot audio, detect speech, and transcribe on request.

    Subscribes to an audio topic, runs voice activity detection continuously,
    and offers services to transcribe, record to a file, and stream
    continuous transcriptions.
    """

    def __init__(self):
        super().__init__("microphone_node")

        # --- Parameters ---
        self.declare_parameter("audio_topic", "/mic")
        self.declare_parameter("sample_rate", 16000)
        self.declare_parameter("vad_threshold", 0.5)

        # Declared so existing launch files may still pass it. Nothing in
        # this node reads it; the persona it used to select is handled by
        # conversation_node.
        self.declare_parameter("robot_name", "robot")

        audio_topic = (
            self.get_parameter("audio_topic").get_parameter_value().string_value
        )
        self.sample_rate = (
            self.get_parameter("sample_rate").get_parameter_value().integer_value
        )
        self.vad_threshold = (
            self.get_parameter("vad_threshold").get_parameter_value().double_value
        )

        # --- State ---
        self.is_listening = False  # Whether incoming audio is buffered
        self.person_speaking = False  # Current voice detection state
        self.last_speaking_instance = 0  # Consecutive silent chunks
        self.audio_buffer = []  # Samples captured while recording
        self.service_lock = (
            threading.Lock()
        )  # Serialises the recording services

        # Realtime STT
        self.is_transcribing = False
        self.recorder = AudioToTextRecorder(
            use_microphone=False,
            spinner=False,
            language="en",
            model="base.en",
            silero_sensitivity=0.17,
            silero_deactivity_detection=True,
            post_speech_silence_duration=1,
        )
        self.process_text_thread = threading.Thread(target=self.recorder_to_text)
        self.process_text_thread.start()

        # --- Callback Groups ---
        # Create a reentrant callback group for the services. This allows service
        # callbacks to run in parallel with the subscription callback, preventing blocking.
        self.service_callback_group = ReentrantCallbackGroup()

        self.leds_publisher = self.create_publisher(LedParameters, "/set_leds", 10)
        self.toggle_blinking_client = self.create_client(
            SetBool, "/naoqi_miscellaneous_node/toggle_blinking"
        )
        self.set_volume_client = self.create_client(
            SetVolume, "/naoqi_speech_node/set_volume"
        )
        self.get_volume_client = self.create_client(
            GetVolume, "/naoqi_speech_node/get_volume"
        )

        # --- Suscriptores ---
        self.audio_subscription = self.create_subscription(
            AudioBuffer, audio_topic, self.audio_callback, 10
        )
        self.get_logger().info(f"Subscribed to audio topic: '{audio_topic}'")

        # --- Servicios ---
        self.speech_to_text_service = self.create_service(
            SpeechToText,
            "~/speech_to_text",
            self.speech_to_text_callback,
            callback_group=self.service_callback_group,
        )
        self.get_logger().info("Service '~/speech_to_text' ready.")

        self.record_audio_service = self.create_service(
            RecordAudio,
            "~/record_audio",
            self.record_audio_callback,
            callback_group=self.service_callback_group,
        )
        self.get_logger().info("Service '~/record_audio' ready.")

        self.set_transcription_mode_service = self.create_service(
            SetTranscriptionMode,
            "~/set_transcription_mode",
            self.set_transcription_mode_callback
        )
        self.get_logger().info("Service '~/set_transcription_mode' ready.")

        # --- Publicadores ---
        self.transcription_publisher = self.create_publisher(
            Transcription, '~/transcription', 10
        )
        self.get_logger().info("Topico de transcripciones creado: ~/transcription")

    def audio_callback(self, msg: AudioBuffer):
        """
        Handle one incoming audio message.

        Runs voice detection, buffers the samples while recording, and feeds
        the streaming recogniser when that mode is on.
        """
        # Assumes signed 16-bit samples, the usual microphone format.
        audio_data = np.frombuffer(msg.data, dtype=np.int16)
        # 1. Voice activity detection
        self.last_speaking_instance = detect_speech(
            audio_buffer=msg.data,
            sample_rate=self.sample_rate,
            last_speaking_instance=self.last_speaking_instance,
        )
        if self.last_speaking_instance == 0:
            self.person_speaking = True
        else:
            self.person_speaking = False

        # 2. Recording
        if self.is_listening:
            self.audio_buffer.extend(audio_data.tolist())

        # 3. Realtime transcription
        if self.is_transcribing:
            # Check if the incoming data is a tuple of signed 16-bit PCM values
            if isinstance(msg.data, tuple):
                try:
                    # Pack the tuple of signed 16-bit PCM values into a bytearray
                    msg.data = bytearray(struct.pack(f"{len(msg.data)}h", *msg.data))
                except struct.error as e:
                    print(f"Error packing PCM data into bytearray: {e}")
            self.recorder.feed_audio(chunk=msg.data)

    def process_text(self, text):
        print(text + " ")
        msg = Transcription()
        msg.text = text
        self.transcription_publisher.publish(msg)

    def recorder_to_text(self):
        while True:
            if self.is_transcribing:
                self.recorder.text(self.process_text)

    def switch_model_blocking(self,current_recorder, new_model, new_language):
        # stop & free current recorder (important to release GPU mem)
        try:
            current_recorder.shutdown()
        except Exception:
            # ignore if already shut down
            pass

        # instantiate a new recorder (this will load the new model and can take time)
        if new_language=="en":
            new_model = "base.en"
        new_rec = AudioToTextRecorder(
            use_microphone=False,
            spinner=False,
            language=new_language,
            model=new_model,
            silero_sensitivity=0.17,
            silero_deactivity_detection=True,
            post_speech_silence_duration=1,
        )
        return new_rec

    def autocut_waiting(self, duration):
        # Timeout if the person talking is not recognized or it takes too long
        max_timeout = duration
        t1 = time.time()
        while not self.person_speaking and time.time() - t1 < 5:
            time.sleep(0.1)
        self.get_logger().info("Person is speaking")
        while (
            self.person_speaking or self.last_speaking_instance < 30
        ) and time.time() - t1 < max_timeout:
            time.sleep(0.1)
            t1 = time.time()
        if time.time() - t1 >= max_timeout:
            self.get_logger().warn("Timeout reached.")
        else:
            self.get_logger().info("Person finished talking")

    def set_transcription_mode_callback(self, request: SetTranscriptionMode.Request, response: SetTranscriptionMode.Response):
        response.success = False
        response.message = "Failure"
        self.is_transcribing = False
        self.recorder = self.switch_model_blocking(self.recorder, new_model="base", new_language=request.language)
        print(f"switched to {request.language} — ready")
        if request.state:
            self.is_transcribing = True
        response.success = True
        response.message = f"switched to {request.language} — ready"
        return response

    def record_audio_callback(
        self, request: RecordAudio.Request, response: RecordAudio.Response
    ):
        """
        Record audio and save it to a file on this machine.
        """
        if not self.service_lock.acquire(blocking=False):
            self.get_logger().warn(
                "Service already in use; ignoring this request."
            )
            response.success = False
            return response

        try:
            self.get_logger().info(
                f"Recording requested for '{request.file_name}.wav'."
            )
            self.audio_buffer = []
            self.is_listening = True

            if request.duration == 0:
                self.autocut_waiting(duration=20)
            else:
                record_duration = float(request.duration)
                # 2 seconds to account for delay
                time.sleep(record_duration+2)

            self.is_listening = False
            file_path = "/tmp/"
            save_recording(
                self.audio_buffer,
                saving_path=file_path,
                file_name=f"{request.file_name}.wav",
                sample_rate=self.sample_rate,
            )
            self.get_logger().info(
                f"Audio saved to {file_path}speech_recordings/{request.file_name}"
            )

            response.success = True

        except Exception as e:
            self.get_logger().error(f"record_audio failed: {e}")
            response.success = False
        finally:
            self.is_listening = False
            self.audio_buffer = []
            self.service_lock.release()

        return response

    def speech_to_text_callback(
        self, request: SpeechToText.Request, response: SpeechToText.Response
    ):
        """
        Record one utterance and return its transcription.
        """
        if not self.service_lock.acquire(blocking=False):
            self.get_logger().warn(
                "Service already in use; ignoring this request."
            )
            response.transcription = "ERROR: Service busy"
            return response

        try:
            self.get_logger().info("Speech-to-text requested.")
            self.audio_buffer = []
            self.is_listening = True

            self.toggle_blinking_client.call(SetBool.Request(data=False))
            current_volume = int(
                self.get_volume_client.call(GetVolume.Request()).volume
            )
            self.set_volume_client.call(SetVolume.Request(volume=0))

            self.set_leds_color(0, 255, 255)
            time.sleep(1)

            if request.autocut:
                self.autocut_waiting(duration=request.timeout)
            else:
                record_duration = float(request.timeout)
                time.sleep(record_duration + 2)

            self.is_listening = False
            file_path = "/tmp/"
            save_recording(
                self.audio_buffer,
                saving_path=file_path,
                file_name="speech2text.wav",
                sample_rate=self.sample_rate,
            )
            self.get_logger().info(
                f"Audio saved to {file_path}speech_recordings/speech2text.wav"
            )

            self.is_listening = False
            self.set_leds_color(255, 255, 255)

            self.set_volume_client.call(SetVolume.Request(volume=current_volume))
            self.toggle_blinking_client.call(SetBool.Request(data=True))

            transcription = transcribe(f"{file_path}speech_recordings/speech2text.wav")
            self.audio_buffer = []
            self.get_logger().info(f"Transcription: '{transcription}'")

            response.transcription = transcription

        except Exception as e:
            self.get_logger().error(f"speech_to_text failed: {e}")
            response.transcription = "ERROR: " + str(e)
        finally:
            self.is_listening = False
            self.audio_buffer = []
            self.service_lock.release()

        return response

    def set_leds_color(self, r, g, b):
        """
        Function for setting the colors of the eyes of the robot.
        Args:
        r,g,b numbers
            r for red
            g for green
            b for blue
        """
        ledsMessage = LedParameters()
        ledsMessage.name = "FaceLeds"
        ledsMessage.red = int(r)
        ledsMessage.green = int(g)
        ledsMessage.blue = int(b)
        ledsMessage.duration = float(0)
        self.leds_publisher.publish(ledsMessage)
        time.sleep(0.2)


def main(args=None):
    rclpy.init(args=args)
    microphone_node = MicrophoneNode()

    # Use a MultiThreadedExecutor to allow callbacks to run on different threads.
    # This is crucial for preventing the long-running services from blocking
    # the high-frequency audio subscription callback.
    executor = MultiThreadedExecutor()
    executor.add_node(microphone_node)
    try:
        # Spin the executor to process callbacks from all nodes.
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        # Make sure to shutdown the executor and clean up resources.
        executor.shutdown()
        microphone_node.destroy_node()
        if rclpy.ok():  # Only call shutdown if rclpy is still "ok"
            rclpy.shutdown()
        os._exit(0)


if __name__ == "__main__":
    main()
