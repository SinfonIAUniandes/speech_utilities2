import rclpy
from rclpy.node import Node
# Import the executor and callback group classes
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
import numpy as np
import threading
import time

# Importar los tipos de mensajes y servicios necesarios
from naoqi_bridge_msgs.msg import AudioBuffer
from speech_msgs2.srv import SpeechToText, RecordAudio

from .logic.audio_processing import save_recording
# Importar los módulos de lógica (aún no implementados)
# from .logic import audio_processing
# from .logic import transcription_clients

class MicrophoneNode(Node):
    """
    Nodo encargado de procesar el audio del micrófono.

    - Se suscribe a un tópico de audio (/audio/raw).
    - Realiza Detección de Actividad de Voz (VAD) continuamente.
    - Ofrece un servicio (~/speech_to_text) para grabar y transcribir audio.
    """

    def __init__(self, robot_name: str = None):
        super().__init__('microphone_node')

        # --- Parámetros ---
        self.declare_parameter('audio_topic', '/mic')
        self.declare_parameter('sample_rate', 16000)
        self.declare_parameter('vad_threshold', 0.5)
        self.declare_parameter('robot_name', 'robot')

        if robot_name is not None:
            self.robot_name = str(robot_name)
            try:
                self.set_parameters([rclpy.parameter.Parameter('robot_name', rclpy.Parameter.Type.STRING, self.robot_name)])
            except Exception:
                pass
        else:
            self.robot_name = self.get_parameter('robot_name').get_parameter_value().string_value
        
        audio_topic = self.get_parameter('audio_topic').get_parameter_value().string_value
        self.sample_rate = self.get_parameter('sample_rate').get_parameter_value().integer_value
        self.vad_threshold = self.get_parameter('vad_threshold').get_parameter_value().double_value

        # --- Variables de estado ---
        self.is_listening = False
        self.is_person_speaking = False
        self.last_speech_timestamp = self.get_clock().now()
        self.audio_buffer = []
        self.service_lock = threading.Lock()

        # --- Lógica de Negocio (Platzhalter) ---
        self.get_logger().info("Modelos de VAD y transcripción (simulados) cargados.")

        # --- Callback Groups ---
        # Create a reentrant callback group for the services. This allows service
        # callbacks to run in parallel with the subscription callback, preventing blocking.
        self.service_callback_group = ReentrantCallbackGroup()

        # --- Suscriptores ---
        # The subscription will use the default (mutually exclusive) callback group.
        self.audio_subscription = self.create_subscription(
            AudioBuffer,
            audio_topic,
            self.audio_callback,
            10)
        self.get_logger().info(f"Suscrito al tópico de audio: '{audio_topic}'")

        # --- Servicios ---
        # Assign the new callback group to the services.
        self.speech_to_text_service = self.create_service(
            SpeechToText,
            '~/speech_to_text',
            self.speech_to_text_callback,
            callback_group=self.service_callback_group)
        self.get_logger().info("Servicio '~/speech_to_text' listo.")

        self.record_audio_service = self.create_service(
            RecordAudio,
            '~/record_audio',
            self.record_audio_callback,
            callback_group=self.service_callback_group)
        self.get_logger().info("Servicio '~/record_audio' listo.")

    def audio_callback(self, msg: AudioBuffer):
        """
        Callback que se ejecuta cada vez que llega un mensaje de audio.
        Realiza VAD y, si está activo, guarda el audio en un buffer.
        """
        audio_data = np.frombuffer(msg.data, dtype=np.int16)
        
        # 1. VAD Logic (Placeholder)
        confidence = 0.0 
        # print(f"Audio callback running. Buffer size: {len(audio_data)}") # Uncomment for debugging
        if confidence > self.vad_threshold:
            self.is_person_speaking = True
            self.last_speech_timestamp = self.get_clock().now()
        else:
            self.is_person_speaking = False

        # 2. Recording Logic
        if self.is_listening:
            self.audio_buffer.extend(audio_data.tolist())

    def record_audio_callback(self, request: RecordAudio.Request, response: RecordAudio.Response):
        """
        Maneja las solicitudes del servicio para grabar audio y guardarlo en un archivo.
        """
        if not self.service_lock.acquire(blocking=False):
            self.get_logger().warn("El servicio ya está en uso. Se ignora la nueva solicitud.")
            response.success = False
            return response

        try:
            self.get_logger().info(f"Solicitud de grabación recibida para '{request.file_name}.wav'.")
            self.audio_buffer = []
            self.is_listening = True

            record_duration = float(request.duration if request.duration > 0 else 5.0)
            self.get_logger().info(f"Grabando por {record_duration} segundos... (Audio callback seguirá corriendo en segundo plano)")

            # Because this service runs in its own thread, time.sleep() will NOT
            # block the audio_callback. The buffer will be filled in the background.
            time.sleep(record_duration)
            
            self.get_logger().info("Grabación finalizada.")
            self.is_listening = False
            
            file_path = "/tmp/"
            save_recording(self.audio_buffer, saving_path=file_path, file_name=f"{request.file_name}.wav", sample_rate=self.sample_rate)
            self.get_logger().info(f"Audio guardado en: {file_path}speech_recordings/{request.file_name}")

            response.success = True

        except Exception as e:
            self.get_logger().error(f"Error durante el servicio record_audio: {e}")
            response.success = False
        finally:
            self.is_listening = False
            self.audio_buffer = []
            self.service_lock.release()

        return response

    def speech_to_text_callback(self, request: SpeechToText.Request, response: SpeechToText.Response):
        """
        Maneja las solicitudes del servicio para transcribir audio.
        """
        if not self.service_lock.acquire(blocking=False):
            self.get_logger().warn("El servicio ya está en uso. Se ignora la nueva solicitud.")
            response.transcription = "ERROR: Service busy"
            return response

        try:
            self.get_logger().info("Solicitud de Speech-to-Text recibida.")
            self.audio_buffer = []
            self.is_listening = True

            self.get_logger().info("Grabando por 3 segundos...")
            # Wait for 3 seconds. The audio_callback will continue running and filling the buffer.
            time.sleep(3.0) 
            self.get_logger().info("Grabación finalizada.")

            self.is_listening = False
            
            file_path = "/tmp/temp_record.wav" # Placeholder
            self.get_logger().info(f"Audio guardado en: {file_path}")

            transcription = "Esta es una transcripción de prueba." # Placeholder
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
        rclpy.shutdown()

if __name__ == '__main__':
    main()