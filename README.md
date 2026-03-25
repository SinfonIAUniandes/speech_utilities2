# Speech Utilities 2

**Speech Utilities 2** is a **ROS 2** speech toolkit package used in the **SinfonIA** ecosystem. It provides reusable audio and conversation components for robotic applications, including:

- Microphone audio capture from ROS topics
- Voice Activity Detection (VAD)
- Speech-to-Text (Whisper and Realtime STT)
- LLM-based conversational responses (Azure OpenAI, OpenAI, Ollama)

## Installation & Setup

Use this package inside a ROS 2 workspace.

### 1. Clone the Repository

Clone into the `src` folder of your ROS 2 workspace:

```bash
cd ~/ros2_ws/src
git clone https://github.com/SinfonIAUniandes/speech_utilities2
```

### 2. Install Python Dependencies

From the package root:

```bash
cd ~/ros2_ws/src/speech_utilities2
pip install -r requirements.txt
```

### 3. Install ROS Dependencies

Install ROS package dependencies with rosdep:

```bash
cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -r -y
```

### 4. Build the Workspace

```bash
cd ~/ros2_ws
colcon build --packages-select speech_utilities2
source install/setup.bash
```

---

## Required ROS Interfaces

This package imports custom messages/services from external interface packages. Make sure they are available in your workspace before building:

- `speech_msgs2`
- `naoqi_bridge_msgs`
- `naoqi_utilities_msgs`

If these packages are missing, `colcon build` or runtime imports will fail.

---

## Package Architecture

The package provides two main ROS 2 nodes.

### 1. `microphone_node`

Main responsibilities:

- Subscribes to microphone audio frames
- Performs VAD continuously
- Records audio on demand
- Runs Speech-to-Text services
- Publishes realtime transcriptions when enabled

Key ROS interfaces:

- **Subscriber**: `/mic` (`naoqi_bridge_msgs/msg/AudioBuffer`) by default
- **Services**:
	- `~/speech_to_text` (`speech_msgs2/srv/SpeechToText`)
	- `~/record_audio` (`speech_msgs2/srv/RecordAudio`)
	- `~/set_transcription_mode` (`speech_msgs2/srv/SetTranscriptionMode`)
- **Publisher**:
	- `~/transcription` (`speech_msgs2/msg/Transcription`)

Auxiliary robot integration endpoints used internally:

- `/set_leds`
- `/naoqi_miscellaneous_node/toggle_blinking`
- `/naoqi_speech_node/set_volume`
- `/naoqi_speech_node/get_volume`

Default parameters:

- `audio_topic`: `/mic`
- `sample_rate`: `16000`
- `vad_threshold`: `0.5`
- `robot_name`: `robot`

### 2. `conversation_node`

Main responsibilities:

- Creates and manages an LLM client
- Handles chat prompts with conversation memory
- Supports runtime LLM reconfiguration

Key ROS interfaces:

- **Services**:
	- `~/set_llm_settings` (`speech_msgs2/srv/SetLLMSettings`)
	- `~/llm_response` (`speech_msgs2/srv/LLMResponse`)
	- `~/clear_llm_history` (`std_srvs/srv/Trigger`)

Default parameters:

- `model_name`: `gpt-4o-azure`
- `temperature`: `0.7`
- `max_tokens`: `256`
- `context`: `""`
- `robot_name`: `robot`

---

## LLM Providers and Environment Variables

`conversation_node` uses `LLMHandler` with a model registry. Current built-in model keys include:

- `gpt-4o-azure`
- `gpt-4.1-azure`
- `gpt-4o-mini`
- `llama3.1-local`

Set environment variables (for example in `.env`):

```bash
# Azure OpenAI
AZURE_OPENAI_API_KEY=your_key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-02-01

# OpenAI (for non-Azure models)
OPENAI_API_KEY=your_key

# Ollama (optional, default shown)
OLLAMA_BASE_URL=http://localhost:11434
```

The package loads `.env` automatically via `python-dotenv`.

---

## Speech-to-Text Backend

The audio pipeline uses:

- **Whisper local models** (`openai-whisper`, `torch`) for service-based transcription
- **RealtimeSTT** for streaming transcriptions on `~/transcription`
- **Silero VAD** via `torch.hub` for speech activity detection

Whisper models are downloaded/cached under the package share directory data folder when first used.

Recorded audio files are saved in:

- `/tmp/speech_recordings/`

---

## Execution

After building and sourcing the workspace:

### Run microphone node

```bash
ros2 run speech_utilities2 microphone_node
```

### Run conversation node

```bash
ros2 run speech_utilities2 conversation_node
```

You can run both in separate terminals.
