"""
ROS 2 node exposing a language model as three services.

A thin adapter: it reads parameters, renders the robot's system prompt from
the package 'data' directory and hands both to :class:`LLMHandler`, which
owns the client and the conversation.

Services: '~/llm_response', '~/set_llm_settings', '~/clear_llm_history'.
"""

import os
from typing import Any, Dict

from ament_index_python.packages import get_package_share_directory
from dotenv import load_dotenv
from rcl_interfaces.msg import SetParametersResult
import rclpy
from rclpy.node import Node
from speech_msgs2.srv import LLMResponse, SetLLMSettings
from std_srvs.srv import Trigger

from .logic.llm.llm_handler import LLMError, LLMHandler
from .logic.llm.prompt_library import DEFAULT_LANGUAGE, fallback_answer, render_system_prompt
from .logic.log_bridge import install as install_log_bridge

load_dotenv()

#: Parameters that feed the system prompt. Changing one re-renders it and
#: starts a fresh conversation.
PROMPT_PARAMETERS = ('robot_name', 'language', 'context')

#: Parameters passed straight to the model client.
MODEL_PARAMETERS = ('model_name', 'temperature', 'max_tokens')


class ConversationNode(Node):
    """Serve language-model answers in the robot's configured persona."""

    def __init__(self, robot_name: str = None):
        """
        Declare parameters, render the system prompt and start the services.

        Raises if the robot has no persona file or the model cannot be prepared.
        Starting without a persona would leave the model answering as a generic
        assistant.
        """
        super().__init__('conversation_node')

        # Modules under logic/ use standard logging; route it into ROS.
        install_log_bridge(self.get_logger())

        self.declare_parameter('robot_name', 'nova')
        self.declare_parameter('model_name', 'gemma3:1b')
        self.declare_parameter('temperature', 0.7)
        self.declare_parameter('max_tokens', 256)
        # Situational text appended to the persona, e.g. "you are at a demo".
        self.declare_parameter('context', '')
        # Language of the persona and of the robot's replies: 'es' or 'en'.
        self.declare_parameter('language', DEFAULT_LANGUAGE)
        # Empty means the package's installed 'data' directory.
        self.declare_parameter('data_dir', '')

        if robot_name is not None:
            self.set_parameters([rclpy.parameter.Parameter(
                'robot_name', rclpy.Parameter.Type.STRING, str(robot_name))])

        self.robot_name = self._text('robot_name')
        self.language = self._text('language') or DEFAULT_LANGUAGE
        self.extra_context = self._text('context')
        self.data_dir = self._text('data_dir') or os.path.join(
            get_package_share_directory('speech_utilities2'), 'data')

        system_prompt = self._render_prompt(self.robot_name, self.language, self.extra_context)
        self.llm_handler = LLMHandler({
            'model_name': self._text('model_name'),
            'temperature': self.get_parameter('temperature').get_parameter_value().double_value,
            'max_tokens': self.get_parameter('max_tokens').get_parameter_value().integer_value,
            'context': system_prompt,
            'fallback_answer': fallback_answer(self.language),
        })

        self.add_on_set_parameters_callback(self._on_parameters_set)

        self.set_settings_service = self.create_service(
            SetLLMSettings, '~/set_llm_settings', self.set_llm_settings_callback)
        self.llm_response_service = self.create_service(
            LLMResponse, '~/llm_response', self.llm_response_callback)
        self.clear_history_service = self.create_service(
            Trigger, '~/clear_llm_history', self.clear_llm_history_callback)
        self.get_logger().info(f"Ready as '{self.robot_name}' in '{self.language}'.")

    def _text(self, name: str) -> str:
        """Return a string parameter's current value."""
        return self.get_parameter(name).get_parameter_value().string_value

    def _render_prompt(self, robot_name: str, language: str, context: str) -> str:
        """Render the system prompt and log its size."""
        prompt = render_system_prompt(
            self.data_dir, robot_name=robot_name, language=language, extra_context=context)
        self.get_logger().info(
            f"System prompt for '{robot_name}' ({language}): "
            f'{len(prompt)} characters from {self.data_dir}.')
        return prompt

    def _on_parameters_set(self, parameters) -> SetParametersResult:
        """
        Apply a live parameter change to the model or the system prompt.

        Changing the robot, language or context re-renders the prompt and
        clears the history: earlier turns in the previous language keep
        pulling the model back to it. A change that cannot be applied is
        rejected and the node carries on unchanged.
        """
        live = PROMPT_PARAMETERS + MODEL_PARAMETERS
        pending = {p.name: p.value for p in parameters if p.name in live}
        if not pending:
            return SetParametersResult(successful=True)

        robot_name = pending.get('robot_name', self.robot_name)
        language = pending.get('language', self.language)
        context = pending.get('context', self.extra_context)
        settings = {name: pending[name] for name in MODEL_PARAMETERS if name in pending}
        try:
            if any(name in pending for name in PROMPT_PARAMETERS):
                settings['context'] = self._render_prompt(robot_name, language, context)
                settings['fallback_answer'] = fallback_answer(language)
            self.llm_handler.update_settings(settings)
        except Exception as error:
            self.get_logger().error(f'Rejected parameter change: {error}')
            return SetParametersResult(successful=False, reason=str(error))

        self.robot_name = robot_name
        self.language = language
        self.extra_context = context
        self.get_logger().info(f'Applied: {", ".join(sorted(pending))}.')
        return SetParametersResult(successful=True)

    def set_llm_settings_callback(self, request, response):
        """
        Apply new model settings.

        An empty 'context' means leave unchanged, so a caller that omits the field
        does not wipe the robot's persona.
        """
        self.get_logger().info(
            f'LLM settings requested: model={request.model_name!r} '
            f'provider={request.model_provider!r} temperature={request.temperature} '
            f'max_tokens={request.max_tokens} context={len(request.context)} characters.')
        settings: Dict[str, Any] = {
            # The provider is a suffix on the model name, e.g. 'gpt-4o' + '-azure'.
            'model_name': request.model_name + request.model_provider,
            'temperature': float(request.temperature),
            'max_tokens': int(request.max_tokens),
        }
        if request.context.strip():
            settings['context'] = request.context
        try:
            self.llm_handler.update_settings(settings)
            response.success = True
            response.message = 'LLM settings updated.'
        except (LLMError, ValueError, TypeError) as error:
            response.success = False
            response.message = f'Could not apply the LLM settings: {error}'
            self.get_logger().error(response.message)
        return response

    def llm_response_callback(self, request, response):
        """
        Answer one prompt.

        This service has no success field, so a failure can only be reported as the
        fallback sentence until one is added.
        """
        self.get_logger().info(f"Prompt received: '{request.prompt}'")
        try:
            response.answer = self.llm_handler.get_response(request.prompt)
            self.get_logger().info(f"Answer: '{response.answer}'")
        except LLMError as error:
            response.answer = fallback_answer(self.language)
            self.get_logger().error(f'LLM request failed: {error}')
        return response

    def clear_llm_history_callback(self, request, response):
        """Forget the conversation, keeping the system prompt."""
        try:
            self.llm_handler.clear_history()
            response.success = True
            response.message = 'Conversation history cleared.'
        except Exception as error:
            response.success = False
            response.message = f'Could not clear the history: {error}'
            self.get_logger().error(response.message)
        return response


def main(args=None):
    """Run the node until interrupted, returning non-zero if it cannot start."""
    rclpy.init(args=args)
    try:
        node = ConversationNode()
    except Exception as error:
        rclpy.logging.get_logger('conversation_node').fatal(
            f'conversation_node could not start: {error}')
        if rclpy.ok():
            rclpy.shutdown()
        return 1

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == '__main__':
    main()
