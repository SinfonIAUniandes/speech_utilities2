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
git clone https://github.com/ros-naoqi/naoqi_bridge_msgs2.git
git clone https://github.com/SinfonIAUniandes/speech_msgs2.git
git clone https://github.com/SinfonIAUniandes/naoqi_utilities_msgs.git
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
colcon build
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

- `model_name`: `gpt-4o-azure` (or gemma3:270m for local usage)
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
- `gemma3:270m`

Set environment variables (for example in `.env`):

```bash
# Azure OpenAI
AZURE_OPENAI_API_KEY=your_key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-02-01

# OpenAI (for non-Azure models)
OPENAI_API_KEY=your_key

# Ollama (Local LLM)
# If ROS 2 runs in Docker and Ollama on the Host, DO NOT use localhost. 
# See the Local LLM Integration section below for how to get the actual URL.
OLLAMA_BASE_URL=http://<HOST_IP>:11434
```

The package loads `.env` automatically via `python-dotenv`.

---

## Local LLM Integration (Ollama + Docker)

When running the ROS 2 workspace inside a Docker container and the Ollama engine on the Ubuntu Host (to optimize hardware resources), network adjustments are required.

### 1. Host Setup (Ubuntu)

Install Ollama and the model on your host machine:

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama run gemma3:270m
```

By default, Ollama blocks external network requests. To allow Docker to reach it, edit the service:

```bash
sudo systemctl edit ollama.service
```

Add the following at the top of the file:

```ini
[Service]
Environment="OLLAMA_HOST=0.0.0.0"
```

Restart the service on the host:

```bash
sudo systemctl daemon-reload
sudo systemctl restart ollama
```

### 2. Container Setup (Docker)

Inside the container, `localhost` points to the container itself, not the host. To find the correct Host IP to set in your `OLLAMA_BASE_URL`, check the default nameserver:

```bash
cat /etc/resolv.conf
```

Copy the IP next to `nameserver` (usually `172.17.0.1` or `172.18.0.1`) and use it in your `.env` file or export it in your `~/.bashrc`.

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

## Test LLM Service (Example)

```bash
ros2 service call /conversation_node/llm_response speech_msgs2/srv/LLMResponse "{prompt: 'Hola, responde en una frase quién eres'}"
```

