import logging
from pathlib import Path

import rclpy
from dotenv import load_dotenv
from rcl_interfaces.msg import SetParametersResult
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node

from speech_msgs2.srv import DeepAgentResponse, SetDeepAgentResponse

# antes de importar speech_agent (lee SPEECH_MODEL, LANGFUSE_*, SINFONIA_* al importarse). resolve(): con
# --symlink-install el módulo corre desde build/, que enlaza a src/, donde está el .env del paquete
load_dotenv(Path(__file__).resolve().parents[1] / ".env")
from .logic.speech_agent import llm, prompts, router  # noqa: E402


class SpeechAgentNode(Node):
    """
    Nodo /speech/agent_node: un turno de conversación por llamada (router + prompts + memoria de speech_agent).

    - Servicio '~/response' (DeepAgentResponse): prompt -> respuesta hablada.
    - Servicio '~/set_settings' (SetDeepAgentResponse): cambia modelo / max_tokens en caliente.
    - Parámetro 'evento': evento activo (ros2 param set /speech/agent_node evento <slug>).
    - SPEECH_MEMORIA (.env): quién escribe la memoria larga, "extractor" (default) o "agente".
    """

    def __init__(self):
        super().__init__("agent_node", namespace="speech")
        self.declare_parameter("evento", prompts.evento() or "")  # default: $SINFONIA_EVENTO
        self.add_on_set_parameters_callback(self.on_params)

        c = prompts.catalog()  # carga Langfuse/snapshot ya, no en el primer turno
        self.get_logger().info(f"prompts desde {c.origen}: "
                               + ", ".join(f"{n}@v{p.version}" for n, p in sorted(c.pieces.items())))
        self.get_logger().info(f"modelo={llm.MODEL} en {llm.BASE_URL}, evento={prompts.evento()}, memoria={router.MEMORIA}")

        # turnos en serie en su propio grupo: un LLM lento no bloquea set_settings ni los parámetros
        self.create_service(DeepAgentResponse, "~/response", self.response_callback,
                            callback_group=MutuallyExclusiveCallbackGroup())
        self.create_service(SetDeepAgentResponse, "~/set_settings", self.set_settings_callback)
        self.get_logger().info("Servicios '~/response' y '~/set_settings' listos.")

    def on_params(self, params):
        for p in params:
            if p.name == "evento":
                prompts.set_evento(p.value)
                self.get_logger().info(f"evento={prompts.evento()}")
        return SetParametersResult(successful=True)

    def response_callback(self, request: DeepAgentResponse.Request, response: DeepAgentResponse.Response):
        user_id = request.user_id or "anonimo"
        self.get_logger().info(f"Prompt recibido ({user_id}): '{request.prompt}'")
        try:
            r, response.answer = router.responder(request.prompt, user_id, request.thread_id)
            response.perfil, response.success = r.perfil.id, True
            self.get_logger().info(f"[{r.perfil.id} {r.fuente} {r.confianza:.2f}] '{response.answer}'")
        except Exception as e:
            response.success, response.message = False, f"{type(e).__name__}: {e}"
            self.get_logger().error(response.message)
        return response

    def set_settings_callback(self, request: SetDeepAgentResponse.Request, response: SetDeepAgentResponse.Response):
        router.configurar(request.model_name, request.max_tokens)
        response.success = True
        response.message = f"modelo={llm.MODEL} max_tokens={router.MAX_TOKENS}"
        self.get_logger().info(response.message)
        return response


def main(args=None):
    logging.basicConfig(level=logging.INFO)  # logs de speech_agent (perfil, huella, memoria)
    rclpy.init(args=args)
    node = SpeechAgentNode()
    try:
        rclpy.spin(node, executor=MultiThreadedExecutor())
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
