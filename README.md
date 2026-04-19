# YouTube-AI-Agent-Tool-Calling-with-LangChain-GPT-4o
An autonomous AI agent that takes a YouTube URL or search query and dynamically chains tools — transcript fetching, metadata extraction, video search, and thumbnail retrieval — to generate intelligent summaries. Built using LangChain, GPT-4o-mini, and Python with both fixed-sequence and recursive agentic pipelines.

# Tool Calling Agent with LangChain
> An AI agent that dynamically calls custom tools to interact with YouTube — extracting video IDs, fetching transcripts, searching videos, retrieving metadata, and generating summaries using GPT-4o-mini.

---

## What is Tool Calling?

Tool calling allows a language model to go beyond text generation — it can **decide when to call external functions**, pass the right arguments, and use the results to answer user queries intelligently.

```
User Query → LLM decides which tool to use → Tool executes → LLM processes result → Final Answer
```

Instead of the developer hardcoding every step, the LLM **reasons about which tools to use and in what order** — making it a true AI agent.

---

## Demo

**Query:** *"Summarize this YouTube video: https://www.youtube.com/watch?v=T-D1OfcDW1M in english"*

```
Step 1 → LLM calls extract_video_id("https://...") → returns "T-D1OfcDW1M"
Step 2 → LLM calls fetch_transcript("T-D1OfcDW1M", "en") → returns full transcript
Step 3 → LLM generates summary from transcript → Final Answer
```

All steps are handled automatically by the chain — no manual orchestration needed.

---

## Tools Built

| Tool | Purpose |
|------|---------|
| `extract_video_id` | Extracts 11-character video ID from any YouTube URL format |
| `fetch_transcript` | Fetches full transcript/captions from a video by ID |
| `search_youtube` | Searches YouTube and returns titles, IDs, and URLs |
| `get_full_metadata` | Extracts title, views, duration, channel, likes, comments, chapters |
| `get_thumbnails` | Retrieves all available thumbnail images and resolutions |

Each tool is built using LangChain's `@tool` decorator, which auto-generates a JSON schema the LLM uses to understand when and how to call it.

---

## Architecture

### Fixed Chain (2-step)
```
Query
  ↓
[LLM] → calls extract_video_id
  ↓
[Tool] → returns video ID
  ↓
[LLM] → calls fetch_transcript
  ↓
[Tool] → returns transcript
  ↓
[LLM] → generates final summary
```

### Recursive Chain (dynamic)
```
Query
  ↓
[LLM] → decides which tool to call
  ↓
[Tool executes] → result added to conversation
  ↓
[LLM] → any more tool calls needed?
   ├── YES → loop back and call next tool
   └── NO  → return final answer
```

The recursive chain handles **any number of tool calls** dynamically — ideal for complex queries like *"Get top 3 trending videos with metadata and thumbnails."*

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square)
![LangChain](https://img.shields.io/badge/LangChain-0.3.21-green?style=flat-square)
![OpenAI](https://img.shields.io/badge/GPT--4o--mini-OpenAI-black?style=flat-square)
![yt-dlp](https://img.shields.io/badge/yt--dlp-YouTube-red?style=flat-square)

- **LangChain** — Tool binding, chain composition, `RunnablePassthrough`, `RunnableLambda`
- **GPT-4o-mini** — Core reasoning and tool selection
- **pytube** — YouTube search and video metadata
- **youtube-transcript-api** — Transcript/caption fetching
- **yt-dlp** — Full metadata and thumbnail extraction
- **Python** — Core language

---

## Project Structure

```
Tool-Calling-Agent/
│
├── tool_calling_agent.ipynb   # Full implementation notebook
├── requirements.txt           # All dependencies
├── .env.example               # API key template
└── README.md
```

---

## Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/Saqib00712/IBM_GenerativeAI_Engineering-With-LLMS.git
cd IBM_GenerativeAI_Engineering-With-LLMS
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up API keys
```bash
cp .env.example .env
```
Edit `.env` and add your key:
```
OPENAI_API_KEY=your_openai_api_key_here
```

### 4. Run the notebook
```bash
jupyter notebook tool_calling_agent.ipynb
```

---

## Key Concepts Covered

- **@tool decorator** — converting Python functions into LangChain-compatible tools with auto-generated JSON schema
- **llm.bind_tools()** — giving the LLM awareness of available tools and how to call them
- **ToolMessage** — passing tool execution results back into the conversation with `tool_call_id`
- **Manual tool calling** — understanding the full step-by-step flow (invoke → extract tool call → execute → feed result back)
- **Automated chain** — using `RunnablePassthrough` and `RunnableLambda` to build a fixed 2-step pipeline
- **Recursive chain** — dynamic tool calling loop that runs until the LLM has no more tool calls to make
- **Tool mapping dictionary** — looking up and executing tools by name at runtime

---

## How the Tool Calling Loop Works

```python
# LLM decides to call a tool
response = llm_with_tools.invoke(messages)

# Extract the tool call
tool_call = response.tool_calls[0]
# → {"name": "extract_video_id", "args": {"url": "https://..."}, "id": "abc123"}

# Execute the tool
result = tool_mapping[tool_call["name"]].invoke(tool_call["args"])
# → "T-D1OfcDW1M"

# Feed result back to LLM as ToolMessage
messages.append(ToolMessage(content=result, tool_call_id=tool_call["id"]))

# LLM continues reasoning with the new information
next_response = llm_with_tools.invoke(messages)
```

The `tool_call_id` links each tool response back to the specific request — essential for multi-step reasoning.

---

## Example Queries

```python
# Summarize a video
{"query": "Summarize this YouTube video: https://www.youtube.com/watch?v=T-D1OfcDW1M in english"}

# Search and get metadata
{"query": "Get top 3 YouTube videos in India and their metadata"}

# Trending videos with thumbnails
{"query": "Show top 3 US trending videos with metadata and thumbnails"}
```

---

## Related Certifications

Built as part of the IBM **Generative AI Engineering with Transformers & LLMs** and **Building AI Agents and Agentic Workflows** Specializations on Coursera.

[![IBM Badge](https://img.shields.io/badge/IBM-Generative%20AI%20Engineering-blue?style=flat-square)](https://www.credly.com/users/muhammad-saqib.361f9b8c)

---

## Author

**Muhammad Saqib**
- GitHub: [@Saqib00712](https://github.com/Saqib00712)
- LinkedIn: [muhammad-saqib](https://www.linkedin.com/in/muhammad-saqib-68b9b3374/)
- Email: saqibkhosa649@gmail.com
- Credly: [15x IBM Certified](https://www.credly.com/users/muhammad-saqib.361f9b8c)
