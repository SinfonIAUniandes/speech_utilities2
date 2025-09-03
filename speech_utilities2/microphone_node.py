import rclpy
from rclpy.node import Node
import numpy as np
import threading
import time
import os

# Import the executor and callback group classes
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup

# Importar los tipos de mensajes y servicios necesarios
from naoqi_bridge_msgs.msg import AudioBuffer
from speech_msgs2.srv import SpeechToText, RecordAudio, SetTranscriptionMode
from speech_msgs2.msg import Transcription
from naoqi_utilities_msgs.msg import LedParameters
from naoqi_utilities_msgs.srv import SetVolume, GetVolume

from std_srvs.srv import SetBool

from .logic.audio_processing import save_recording
from .logic.voice_activity_detection import detect_speech
from .logic.speech_2_text import transcribe

# Realtime STT imports
import struct
from RealtimeSTT import AudioToTextRecorder


class MicrophoneNode(Node):
    """
    Nodo encargado de procesar el audio del micrófono.

    - Se suscribe a un tópico de audio (/audio/raw).
    - Realiza Detección de Actividad de Voz (VAD) continuamente.
    - Ofrece un servicio (~/speech_to_text) para grabar y transcribir audio.
    """

    def __init__(self, robot_name: str = None):
        super().__init__("microphone_node")

        # --- Parámetros ---
        self.declare_parameter("audio_topic", "/mic")
        self.declare_parameter("sample_rate", 16000)
        self.declare_parameter("vad_threshold", 0.5)

        # El nombre del robot
        self.declare_parameter("robot_name", "robot")

        if robot_name is not None:
            # si se pasa por constructor, también actualizamos el parámetro local
            self.robot_name = str(robot_name)
            # opcional: establecer el parámetro en el servidor de parámetros
            try:
                self.set_parameters(
                    [
                        rclpy.parameter.Parameter(
                            "robot_name", rclpy.Parameter.Type.STRING, self.robot_name
                        )
                    ]
                )
            except Exception:
                # no crítico si falla (dependiendo de la versión de rclpy)
                pass
        else:
            self.robot_name = (
                self.get_parameter("robot_name").get_parameter_value().string_value
            )

        audio_topic = (
            self.get_parameter("audio_topic").get_parameter_value().string_value
        )
        self.sample_rate = (
            self.get_parameter("sample_rate").get_parameter_value().integer_value
        )
        self.vad_threshold = (
            self.get_parameter("vad_threshold").get_parameter_value().double_value
        )

        # --- Variables de estado ---
        self.is_listening = False  # Controla si se debe guardar el audio en el buffer
        self.person_speaking = False  # Estado actual del VAD
        self.last_speaking_instance = 0  # Para detectar fin de locución
        self.audio_buffer = []  # Buffer para almacenar el audio durante la grabación
        self.service_lock = (
            threading.Lock()
        )  # Para evitar llamadas concurrentes al servicio

        # Realtime STT
        self.is_transcribing = False
        self.recorder = AudioToTextRecorder(
            use_microphone=False,
            spinner=False,
            language="en",
            model="base.en",
            silero_sensitivity=0.2,
            silero_deactivity_detection=True,
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
        self.get_logger().info(f"Suscrito al tópico de audio: '{audio_topic}'")

        # --- Servicios ---
        self.speech_to_text_service = self.create_service(
            SpeechToText,
            "~/speech_to_text",
            self.speech_to_text_callback,
            callback_group=self.service_callback_group,
        )
        self.get_logger().info("Servicio '~/speech_to_text' listo.")

        self.record_audio_service = self.create_service(
            RecordAudio,
            "~/record_audio",
            self.record_audio_callback,
            callback_group=self.service_callback_group,
        )
        self.get_logger().info("Servicio '~/record_audio' listo.")

        self.set_transcription_mode_service = self.create_service(
            SetTranscriptionMode,
            "~/set_transcription_mode",
            self.set_transcription_mode_callback
        )
        self.get_logger().info("Servicio '~/set_transcription_mode' listo.")

        # --- Publicadores ---
        self.transcription_publisher = self.create_publisher(
            Transcription, '~/transcription', 10
        )
        self.get_logger().info("Topico de transcripciones creado: ~/transcription")

    def audio_callback(self, msg: AudioBuffer):
        """
        Callback que se ejecuta cada vez que llega un mensaje de audio.
        Realiza VAD y, si está activo, guarda el audio en un buffer.
        """
        # Convertir el buffer de bytes a un array de numpy
        # Asumimos formato int16, común en micrófonos
        audio_data = np.frombuffer(msg.data, dtype=np.int16)
        # 1. Lógica de Detección de Actividad de Voz (VAD)
        self.last_speaking_instance = detect_speech(
            audio_buffer=msg.data,
            sample_rate=self.sample_rate,
            last_speaking_instance=self.last_speaking_instance,
        )
        if self.last_speaking_instance == 0:
            self.person_speaking = True
        else:
            self.person_speaking = False

        # 2. Lógica de grabación
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
            silero_sensitivity=0.2,
            silero_deactivity_detection=True
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
        Maneja las solicitudes del servicio para grabar audio y guardarlo en un archivo.
        """
        if not self.service_lock.acquire(blocking=False):
            self.get_logger().warn(
                "El servicio ya está en uso. Se ignora la nueva solicitud."
            )
            response.success = False
            return response

        try:
            self.get_logger().info(
                f"Solicitud de grabación recibida para '{request.file_name}.wav'."
            )
            self.audio_buffer = []
            self.is_listening = True

            if request.duration == 0:
                self.autocut_waiting(duration=20)
            else:
                record_duration = float(request.duration)
                time.sleep(record_duration)

            self.is_listening = False
            file_path = "/tmp/"
            save_recording(
                self.audio_buffer,
                saving_path=file_path,
                file_name=f"{request.file_name}.wav",
                sample_rate=self.sample_rate,
            )
            self.get_logger().info(
                f"Audio guardado en: {file_path}speech_recordings/{request.file_name}"
            )

            response.success = True

        except Exception as e:
            self.get_logger().error(f"Error durante el servicio record_audio: {e}")
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
        Maneja las solicitudes del servicio para transcribir audio.
        """
        if not self.service_lock.acquire(blocking=False):
            self.get_logger().warn(
                "El servicio ya está en uso. Se ignora la nueva solicitud."
            )
            response.transcription = "ERROR: Service busy"
            return response

        try:
            self.get_logger().info("Solicitud de Speech-to-Text recibida.")
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
                time.sleep(record_duration)

            self.is_listening = False
            file_path = "/tmp/"
            save_recording(
                self.audio_buffer,
                saving_path=file_path,
                file_name="speech2text.wav",
                sample_rate=self.sample_rate,
            )
            self.get_logger().info(
                f"Audio guardado en: {file_path}speech_recordings/speech2text.wav"
            )

            self.is_listening = False
            self.set_leds_color(255, 255, 255)

            self.set_volume_client.call(SetVolume.Request(volume=current_volume))
            self.toggle_blinking_client.call(SetBool.Request(data=True))

            transcription = transcribe(f"{file_path}speech_recordings/speech2text.wav")
            self.audio_buffer = []
            self.get_logger().info(f"Transcripción: '{transcription}'")

            response.transcription = transcription

        except Exception as e:
            self.get_logger().error(f"Error durante el servicio speech_to_text: {e}")
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
