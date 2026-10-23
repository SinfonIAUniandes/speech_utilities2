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

### 3. `speech_agent_node`

Runs one conversation turn per call using the `speech_agent` library in `speech_utilities2/logic/speech_agent/`
(analyzer/router, Langfuse prompts with an offline snapshot, per-user memory in SQLite). Design specs: `docs/speech_agent/`.

Key ROS interfaces (node `/speech/agent_node`):

- **Services**:
	- `~/response` (`speech_msgs2/srv/DeepAgentResponse`): `prompt`, `user_id` (`""` = anonymous, nothing saved), `thread_id` (`""` = `user_id`) -> `success`, `answer`, `perfil`, `message`
	- `~/set_settings` (`speech_msgs2/srv/SetDeepAgentResponse`): `model_name` (`""` = keep), `max_tokens` (`0` = keep)
- **Parameters**:
	- `evento` (default `$SINFONIA_EVENTO`): active event slug, changeable at runtime

Memory: short-term = LangGraph checkpointer (history of the `thread_id`, one per conversation session; the model sees
the last `SPEECH_HISTORIA` messages, default 20). Long-term = `/memories/perfil.md`, one file per `user_id`
(CompositeBackend + StoreBackend with a per-person namespace), injected into the system prompt on every turn.
`SPEECH_MEMORIA` in `.env` picks who writes it:

| `SPEECH_MEMORIA` | Who writes `perfil.md` | Minimum model |
|---|---|---|
| `extractor` (default) | `remember()` in the background (structured extraction, sensitive-data filter, 2 KB cap) | `llama3.2:3b` |
| `agente` | a separate memory agent in the background (short prompt, only `read_file`/`write_file`/`edit_file` on the CompositeBackend; can also correct or replace lines); sensitive lines and size are cleaned after each turn | `qwen3:4b-instruct-2507-q4_K_M` |

Environment (`.env` at the package root, found when built with `--symlink-install`): `OLLAMA_BASE_URL` (with or without `/v1`),
`SPEECH_MODEL` (default `llama3.2:3b`), `SPEECH_MEMORIA`, `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY`/`LANGFUSE_BASE_URL` (optional; without
them prompts come from the snapshot and nothing is traced), `SINFONIA_EVENTO`, `SINFONIA_ROBOT`, `SPEECH_DATA_DIR` (memory; in Docker put
it inside the mounted workspace, e.g. `/home/devuser/sinfonia/.speech_data`).

Ollama for this node: `ollama/docker-compose.yml`.

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

## Test speech_agent_node (sinfonia container)

ROS 2 and the Python dependencies live only inside the `sinfonia-jazzy-dev` container: running `colcon` or `pytest`
on the host fails (`Unknown command: colcon`, `No module named pytest`).

### 1. Model and `.env` (host)

```bash
docker exec -it ollama-ollama-1 ollama pull qwen3:4b-instruct-2507-q4_K_M   # needed for SPEECH_MEMORIA=agente
```

In `speech_utilities2/.env` (read when the node starts):

```bash
SPEECH_MODEL=qwen3:4b-instruct-2507-q4_K_M
SPEECH_MEMORIA=agente                             # or extractor (works with llama3.2:3b)
SPEECH_DATA_DIR=/home/devuser/sinfonia/.speech_data  # memory survives recreating the container
```

### 2. Enter the container

```bash
docker start sinfonia-jazzy-dev        # only if it is stopped
docker exec -it sinfonia-jazzy-dev bash
```

Only for a newly created container (pip packages are lost when it is recreated):

```bash
pip install --break-system-packages --user deepagents langchain-openai langfuse langgraph-checkpoint-sqlite python-dotenv pytest
```

### 3. Build and run (terminal 1)

```bash
cd ~/sinfonia
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install --packages-select speech_msgs2 speech_utilities2
source install/setup.bash
ros2 run speech_utilities2 speech_agent_node
```

The startup log must show `modelo=qwen3:4b-instruct-2507-q4_K_M ... memoria=agente`; if it shows `llama3.2:3b` the `.env`
is missing those lines. Build with `--symlink-install`: the node finds `.env` through the `build/` -> `src/` symlink.
With it, code changes only need a node restart; rebuild only after changing `setup.py`, `package.xml` or the `.srv` files.

### 4. Talk to it (terminal 2: another `docker exec -it sinfonia-jazzy-dev bash`)

```bash
source ~/sinfonia/install/setup.bash
R=/speech/agent_node/response; T=speech_msgs2/srv/DeepAgentResponse

ros2 service call $R $T "{prompt: 'hola, me llamo Ana y me encanta el café', user_id: ana, thread_id: s1}"
ros2 service call $R $T "{prompt: 'jaja ok, qué frío hace hoy', user_id: ana, thread_id: s1}"
ros2 service call $R $T "{prompt: 'ya no me gusta el café, ahora prefiero el té', user_id: ana, thread_id: s1}"
ros2 service call $R $T "{prompt: '¿qué sabes de mí?', user_id: ana, thread_id: s2}"   # new session: long-term memory only

# router: expect perfil general / accion / delicado
ros2 service call $R $T "{prompt: 'llévame al laboratorio'}"
ros2 service call $R $T "{prompt: 'me siento muy mal, creo que me voy a desmayar'}"
```

- `user_id`: one long-term memory (`perfil.md`) per person; empty = anonymous, nothing is saved.
- `thread_id`: one conversation session (short-term memory); empty = `user_id`.
- Terminal 1 logs, per turn, the chosen perfil and `memoria ana (agente|extractor): ...` with what was saved. Memory is
  written in the background, so leave a few seconds between calls.

Runtime config:

```bash
ros2 param set /speech/agent_node evento ejemplo
ros2 service call /speech/agent_node/set_settings speech_msgs2/srv/SetDeepAgentResponse "{max_tokens: 80}"
```

### 5. Library tests (no ROS needed)

```bash
cd ~/sinfonia/src/speech_utilities2
python3 -m pytest test/speech_agent -q        # no network
SPEECH_LIVE=1 SPEECH_MODEL=qwen3:4b-instruct-2507-q4_K_M python3 -m pytest test/speech_agent/test_memory_live.py -q   # against Ollama, both memory modes
```

`pytest` does not read `.env`, so pass `SPEECH_LIVE` / `SPEECH_MODEL` on the command line. The `agente` live test is skipped
with `llama3.2:3b`.

### Prompts

Regenerate the offline prompt snapshot from Langfuse (writes into the source tree): run
`python3 -m speech_utilities2.logic.speech_agent.prompts snapshot` from the package root.
Seed Langfuse with the initial prompts: `python3 scripts/seed_langfuse_prompts.py` (never overwrites existing prompts;
edit changed ones, such as `speech/response-base`, in the Langfuse UI).
