import rclpy
from rclpy.node import Node

# Importar los tipos de servicios necesarios
from speech_msgs2.srv import SetLLMSettings, LLMResponse
from std_srvs.srv import Trigger

class ConversationNode(Node):
    """
    Nodo encargado de gestionar la conversación con un Modelo de Lenguaje Grande (LLM).

    - Ofrece un servicio para configurar los parámetros del LLM.
    - Ofrece un servicio para enviar un prompt y recibir una respuesta.
    - Ofrece un servicio para limpiar el historial de la conversación.
    """

    def __init__(self, robot_name: str = None):
        super().__init__("conversation_node")

        # --- Parámetros del LLM ---
        self.declare_parameter("model_provider", "default_provider")
        self.declare_parameter("model_name", "default_model")
        self.declare_parameter("temperature", 0.7)
        self.declare_parameter("max_tokens", 500)

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

        # --- Variables de estado ---
        self.model_provider = self.get_parameter("model_provider").get_parameter_value().string_value
        self.model_name = self.get_parameter("model_name").get_parameter_value().string_value
        self.temperature = self.get_parameter("temperature").get_parameter_value().double_value
        self.max_tokens = self.get_parameter("max_tokens").get_parameter_value().integer_value
        self.context = "" # Contexto o mensaje de sistema para el LLM
        self.conversation_history = []

        self.get_logger().info(f"Proveedor LLM inicial: {self.model_provider}")
        self.get_logger().info(f"Modelo LLM inicial: {self.model_name}")

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
        Callback para configurar los parámetros del LLM.
        """
        self.get_logger().info("Solicitud para actualizar la configuración del LLM recibida.")
        # --- IMPLEMENTACIÓN AQUÍ ---
        # Lógica para actualizar los parámetros y reiniciar el cliente del LLM si es necesario.
        # Por ejemplo:
        # self.model_provider = request.model_provider
        # self.model_name = request.model_name
        # self.temperature = float(request.temperature)
        # self.max_tokens = int(request.max_tokens)
        # self.context = request.context
        # self.get_logger().info("Configuración del LLM actualizada.")
        # self.clear_llm_history_callback(None, None) # Opcional: limpiar historial al cambiar contexto
        # response.success = True
        # response.message = "Configuración actualizada correctamente."
        pass
        return response

    def llm_response_callback(self, request: LLMResponse.Request, response: LLMResponse.Response):
        """
        Callback para enviar un prompt al LLM y obtener una respuesta.
        """
        self.get_logger().info(f"Prompt recibido: '{request.prompt}'")
        # --- IMPLEMENTACIÓN AQUÍ ---
        # Lógica para enviar el prompt al LLM, gestionar el historial y obtener la respuesta.
        # Por ejemplo:
        # self.conversation_history.append({"role": "user", "content": request.prompt})
        # llm_answer = self.llm_client.get_response(self.conversation_history)
        # self.conversation_history.append({"role": "assistant", "content": llm_answer})
        # response.answer = llm_answer
        pass
        return response

    def clear_llm_history_callback(self, request: Trigger.Request, response: Trigger.Response):
        """
        Callback para limpiar el historial de la conversación del LLM.
        """
        self.get_logger().info("Solicitud para limpiar el historial de la conversación recibida.")
        # --- IMPLEMENTACIÓN AQUÍ ---
        # Lógica para borrar el historial de mensajes.
        # Por ejemplo:
        # self.conversation_history = []
        # if self.context:
        #     self.conversation_history.append({"role": "system", "content": self.context})
        # response.success = True
        # response.message = "Historial de la conversación limpiado."
        pass
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