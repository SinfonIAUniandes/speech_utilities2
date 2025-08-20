import rclpy
from rclpy.node import Node
import numpy as np
import threading
import os

# Importar los tipos de mensajes y servicios necesarios
from audio_msgs.msg import AudioData
from speech_utilities2.srv import SpeechToText, RecordAudio

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

    def __init__(self):
        super().__init__('microphone_node')

        # --- Parámetros ---
        self.declare_parameter('audio_topic', '/audio/raw')
        self.declare_parameter('sample_rate', 16000)
        self.declare_parameter('vad_threshold', 0.5)
        
        audio_topic = self.get_parameter('audio_topic').get_parameter_value().string_value
        self.sample_rate = self.get_parameter('sample_rate').get_parameter_value().integer_value
        self.vad_threshold = self.get_parameter('vad_threshold').get_parameter_value().double_value

        # --- Variables de estado ---
        self.is_listening = False  # Controla si se debe guardar el audio en el buffer
        self.is_person_speaking = False # Estado actual del VAD
        self.last_speech_timestamp = self.get_clock().now() # Para detectar fin de locución
        self.audio_buffer = [] # Buffer para almacenar el audio durante la grabación
        self.service_lock = threading.Lock() # Para evitar llamadas concurrentes al servicio

        # --- Lógica de Negocio (Platzhalter) ---
        # self.vad_model = audio_processing.load_vad_model()
        # self.transcription_model = transcription_clients.load_model()
        self.get_logger().info("Modelos de VAD y transcripción (simulados) cargados.")

        # --- Suscriptores ---
        self.audio_subscription = self.create_subscription(
            AudioData,
            audio_topic,
            self.audio_callback,
            10)
        self.get_logger().info(f"Suscrito al tópico de audio: '{audio_topic}'")

        # --- Servicios ---
        self.speech_to_text_service = self.create_service(
            SpeechToText,
            '~/speech_to_text',
            self.speech_to_text_callback)
        self.get_logger().info("Servicio '~/speech_to_text' listo.")

        self.record_audio_service = self.create_service(
            RecordAudio,
            '~/record_audio',
            self.record_audio_callback)
        self.get_logger().info("Servicio '~/record_audio' listo.")

    def audio_callback(self, msg: AudioData):
        """
        Callback que se ejecuta cada vez que llega un mensaje de audio.
        Realiza VAD y, si está activo, guarda el audio en un buffer.
        """
        # Convertir el buffer de bytes a un array de numpy
        # Asumimos formato int16, común en micrófonos
        audio_data = np.frombuffer(msg.data, dtype=np.int16)

        # 1. Lógica de Detección de Actividad de Voz (VAD)
        # TODO: Llamar a la función de VAD real desde audio_processing.py
        # confidence = audio_processing.get_vad_confidence(audio_data, self.sample_rate, self.vad_model)
        confidence = 0.0 # Platzhalter
        
        if confidence > self.vad_threshold:
            self.is_person_speaking = True
            self.last_speech_timestamp = self.get_clock().now()
        else:
            self.is_person_speaking = False

        # 2. Lógica de grabación
        if self.is_listening:
            self.audio_buffer.extend(audio_data.tolist())

    def record_audio_callback(self, request: RecordAudio.Request, response: RecordAudio.Response):
        """
        Maneja las solicitudes del servicio para grabar audio y guardarlo en un archivo.
        """
        if not self.service_lock.acquire(blocking=False):
            self.get_logger().warn("El servicio ya está en uso. Se ignora la nueva solicitud.")
            response.success = False
            response.file_path = "ERROR: Service busy"
            return response

        try:
            self.get_logger().info(f"Solicitud de grabación recibida para '{request.file_name}.wav'.")
            self.audio_buffer = []
            self.is_listening = True

            # TODO: Implementar lógica de grabación basada en VAD si request.duration es 0.
            # Por ahora, simulamos una grabación de duración fija.
            record_duration = float(request.duration if request.duration > 0 else 5.0)
            self.get_logger().info(f"Grabando por {record_duration} segundos...")
            # Esta es una forma simple de esperar, una implementación real usaría VAD.
            rclpy.spin_once(self, timeout_sec=record_duration)
            self.get_logger().info("Grabación finalizada.")

            self.is_listening = False

            # TODO: 1. Crear una carpeta de grabaciones si no existe.
            # recordings_dir = "path/to/your/recordings"
            # os.makedirs(recordings_dir, exist_ok=True)

            # TODO: 2. Guardar el audio grabado usando una función de audio_processing.py
            # file_path = audio_processing.save_recording(
            #     self.audio_buffer, recordings_dir, request.file_name, self.sample_rate
            # )
            file_path = f"/tmp/{request.file_name}.wav"  # Platzhalter
            self.get_logger().info(f"Audio guardado (simulado) en: {file_path}")

            response.success = True
            response.file_path = file_path

        except Exception as e:
            self.get_logger().error(f"Error durante el servicio record_audio: {e}")
            response.success = False
            response.file_path = "ERROR: " + str(e)
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
            response.text = "ERROR: Service busy"
            return response

        try:
            self.get_logger().info("Solicitud de Speech-to-Text recibida.")
            self.audio_buffer = []
            self.is_listening = True

            # TODO: Lógica de grabación (esperar a que el usuario hable y termine)
            # Por ahora, simulamos una grabación de 3 segundos.
            self.get_logger().info("Grabando...")
            rclpy.spin_once(self, timeout_sec=3.0) 
            self.get_logger().info("Grabación finalizada.")

            self.is_listening = False
            
            # TODO: 1. Guardar el audio grabado usando una función de audio_processing.py
            # file_path = audio_processing.save_recording(self.audio_buffer, "temp_record.wav", self.sample_rate)
            file_path = "/tmp/temp_record.wav" # Platzhalter
            self.get_logger().info(f"Audio guardado en: {file_path}")

            # TODO: 2. Transcribir el audio usando una función de transcription_clients.py
            # transcription = transcription_clients.transcribe(file_path, self.transcription_model, lang=request.lang)
            transcription = "Esta es una transcripción de prueba." # Platzhalter
            self.get_logger().info(f"Transcripción: '{transcription}'")

            response.text = transcription

        except Exception as e:
            self.get_logger().error(f"Error durante el servicio speech_to_text: {e}")
            response.text = "ERROR: " + str(e)
        finally:
            self.is_listening = False
            self.audio_buffer = []
            self.service_lock.release()

        return response


def main(args=None):
    rclpy.init(args=args)
    microphone_node = MicrophoneNode()
    try:
        rclpy.spin(microphone_node)
    except KeyboardInterrupt:
        pass
    finally:
        microphone_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()