import rclpy
from rclpy.node import Node
from typing import Dict, Any

# Importar los tipos de servicios necesarios
from speech_msgs2.srv import SetLLMSettings, LLMResponse
from std_srvs.srv import Trigger
from dotenv import load_dotenv

from .logic.llm.llm_handler import LLMHandler
load_dotenv()


class ConversationNode(Node):
    """
    Nodo encargado de gestionar la conversación con un Modelo de Lenguaje Grande (LLM).

    - Ofrece un servicio para configurar los parámetros del LLM.
    - Ofrece un servicio para enviar un prompt y recibir una respuesta.
    - Ofrece un servicio para limpiar el historial de la conversación.
    """

    def __init__(self, robot_name: str = None):
        super().__init__("conversation_node")

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

        # --- Declarar parámetros configurables del LLM ---
        self.declare_parameter("model_name", "gpt-4o-azure")
        self.declare_parameter("temperature", 0.7)
        self.declare_parameter("max_tokens", 256)
        self.declare_parameter("context", "")

        # --- Configuración inicial por defecto ---
        initial_settings = {
            "model_name": self.get_parameter("model_name").get_parameter_value().string_value or "gpt-4o-azure",
            "temperature": self.get_parameter("temperature").get_parameter_value().double_value if self.get_parameter("temperature").get_parameter_value().double_value != 0.0 else 0.7,
            "max_tokens": self.get_parameter("max_tokens").get_parameter_value().integer_value or 256,
            "context": self.get_parameter("context").get_parameter_value().string_value or ""
        }

        # --- Instanciar el manejador de LLM ---
        try:
            self.llm_handler = LLMHandler(initial_settings)
        except ImportError as e:
            self.get_logger().fatal(f"No se pudo inicializar LLMHandler. Asegúrate de que las dependencias de LangChain están instaladas: {e}")
            # Apagar el nodo si el componente principal no puede cargarse.
            # Esto evita que los servicios queden colgados sin funcionalidad.
            self.destroy_node()
            return
        except Exception as e:
            self.get_logger().fatal(f"Error inesperado al inicializar LLMHandler: {e}")
            self.destroy_node()
            return

        # --- Servicios ---
        self.set_settings_service = self.create_service(
            SetLLMSettings,
            "~/set_llm_settings",
            self.set_llm_settings_callback
        )
        self.get_logger().info("Servicio '~/set_llm_settings' listo.")

        self.llm_response_service = self.create_service(
            LLMResponse,
            "~/llm_response",
            self.llm_response_callback
        )
        self.get_logger().info("Servicio '~/llm_response' listo.")

        self.clear_history_service = self.create_service(
            Trigger,
            "~/clear_llm_history",
            self.clear_llm_history_callback
        )
        self.get_logger().info("Servicio '~/clear_llm_history' listo.")

    def set_llm_settings_callback(self, request: SetLLMSettings.Request, response: SetLLMSettings.Response):
        """
        Callback para configurar los parámetros del LLM a través del LLMHandler.
        """
        self.get_logger().info(f"Solicitud para actualizar la configuración del LLM recibida: {request}")
        try:
            new_settings: Dict[str, Any] = {
                "model_name": request.model_name+request.model_provider,
                "temperature": float(request.temperature),
                "max_tokens": int(request.max_tokens),
                "context": request.context
            }
            self.llm_handler.update_settings(new_settings)
            response.success = True
            response.message = "Configuración del LLM actualizada correctamente."
            self.get_logger().info(response.message)
        except (ValueError, TypeError) as e:
            response.success = False
            response.message = f"Error en los tipos de datos de la configuración: {e}"
            self.get_logger().error(response.message)
        except Exception as e:
            response.success = False
            response.message = f"Error al actualizar la configuración del LLM: {e}"
            self.get_logger().error(response.message)
        
        return response

    def llm_response_callback(self, request: LLMResponse.Request, response: LLMResponse.Response):
        """
        Callback para enviar un prompt al LLM y obtener una respuesta.
        """
        self.get_logger().info(f"Prompt recibido: '{request.prompt}'")
        try:
            answer = self.llm_handler.get_response(request.prompt)
            response.answer = answer
            self.get_logger().info(f"Respuesta generada: '{answer}'")
        except Exception as e:
            error_message = f"Error al obtener respuesta del LLM: {e}"
            self.get_logger().error(error_message)
            response.answer = error_message

        return response

    def clear_llm_history_callback(self, request: Trigger.Request, response: Trigger.Response):
        """
        Callback para limpiar el historial de la conversación del LLM.
        """
        self.get_logger().info("Solicitud para limpiar el historial de la conversación recibida.")
        try:
            self.llm_handler.clear_history()
            response.success = True
            response.message = "Historial de la conversación limpiado."
            self.get_logger().info(response.message)
        except Exception as e:
            response.success = False
            response.message = f"Error al limpiar el historial: {e}"
            self.get_logger().error(response.message)

        return response


def main(args=None):
    rclpy.init(args=args)
    conversation_node = ConversationNode()
    try:
        rclpy.spin(conversation_node)
    except KeyboardInterrupt:
        pass
    finally:
        conversation_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()