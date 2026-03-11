# External AI Setup and Usage Guide

This guide explains how to use ACE-Step with external AI models for caption generation, planning, and lyric generation.

## What External AI Does

External AI lets ACE-Step use a remote or local text model instead of the built-in 5Hz LM for text-heavy tasks such as:

- creating a sample from a short idea
- expanding a rough caption into a structured song description
- generating lyrics from a caption
- planning caption, metadata, and lyrics together

The music generation model is still ACE-Step. The external model only helps with the text side.

## Supported Providers

ACE-Step currently supports:

- Z.ai
- OpenAI-compatible endpoints
- Anthropic Claude
- Ollama

## Model Suitability and Caveats

External AI support is for instruction-following text models only.

ACE-Step expects a model that can read a prompt and return useful text for:

- planning
- caption generation
- lyric generation
- metadata formatting

The following are not supported as external text-task models:

- embedding models
- speech-to-text or transcription models
- text-to-speech models
- image or vision-only models
- rerankers
- moderation models
- diffusion or audio generation endpoints pretending to be chat endpoints
- any provider endpoint that cannot follow the selected chat protocol

If you plug a non-LM or non-chat model into this pipeline, ACE-Step may:

- fail to fetch models cleanly
- return invalid JSON
- generate unusable captions or lyrics
- time out
- appear to save correctly but fail when a text task runs

In short: use proper chat or messages language models only.

### Do You Need a Thinking or Reasoning Model?

Not strictly. ACE-Step can work with normal instruction-following chat models too.

However, stronger reasoning-style models usually perform better for:

- **Create Sample**
- **Enhance Caption**
- full planning tasks
- structured metadata generation

These tasks ask the model to keep track of several constraints at once, so higher-quality reasoning or planning models tend to be more reliable.

For example:

- a stronger structured model like GLM 4.7 can do well on planning and caption work
- a good local instruct model in Ollama can also work well
- weaker or poorly aligned models may ignore format rules, invent bad metadata, or produce low-quality lyrics

So the rule is:

- reasoning model: recommended for best results
- normal instruction-following chat model: supported
- non-language-model endpoint: not supported

## Quick Start

1. Open **Settings** in the Gradio UI.
2. Open the **External LLM** accordion within **Settings**.
3. Pick a **Provider**.
4. Check or edit the **Protocol**, **Base URL**, and **Model**.
5. Enter your **API key** if the provider requires one.
6. Optional: enter a **Store Passphrase** to save the API key in encrypted form.
7. Click **Fetch Models** if you want ACE-Step to read the model list from the endpoint.
8. Click **Save External LLM Settings**.
9. Go back to **Service Configuration** and choose the saved external entry in **5Hz LM Model Path**.
10. Use **Create Sample**, **Enhance Caption**, **Enhance Lyrics**, or **Generate Lyrics** as normal.

## How Selection Works

When an external model is selected in **Service Configuration**:

- ACE-Step uses the external model for text tasks
- the local 5Hz LM is not required for those text tasks
- **Initialize 5Hz LM** is automatically turned off

Current limitation: **Create Sample** still requires the local 5Hz LM to be initialized. Today both the Gradio click handler check and the `create_sample()` function enforce `llm_handler.llm_initialized`, so external models do not yet cover **Create Sample** until that routing is implemented.

When you switch back to a local 5Hz LM model:

- external settings stay saved
- external mode is turned off
- local text generation resumes

This makes it easy to swap between local and external text models without re-entering your provider details each time.

## Saving Provider Settings

ACE-Step saves non-secret provider settings so you can switch back later more easily.

These usually include:

- provider
- protocol
- base URL
- selected model

API keys are stored separately in encrypted form if you choose to save them.

## Provider Notes

### Ollama

Use Ollama when you want a fully local external text model.

Typical setup:

- Provider: `Ollama`
- Protocol: `openai_chat`
- Base URL: usually `http://127.0.0.1:11434/v1/chat/completions`
- API key: not required in normal local setups

Use **Fetch Models** after Ollama is running so ACE-Step can populate the model list.

### Z.ai / GLM

Z.ai works well for structured caption and planning tasks, but there is one important restriction:

> The special Z.ai coding endpoint is not a general-purpose shortcut.
> It only works if your account has access to the coding-plan style quota for that endpoint.

If you see quota or `1113` errors on a coding endpoint:

- your account may not have coding-plan quota for that endpoint
- you may need to use the normal Z.ai chat endpoint instead
- or use another provider/model that your account can fund

In short: the coding endpoint is account-plan dependent, not a universal "better GLM endpoint".

### OpenAI

OpenAI works through the normal API platform.

> A ChatGPT Plus account does not automatically give you API billing or API quota.

If ACE-Step reports OpenAI quota errors:

- your API project may not have billing enabled
- your API key may belong to a different project or org than you expect
- ChatGPT subscription access and API billing are separate products

This means a ChatGPT Plus subscription can still show `insufficient_quota` in ACE-Step if the API project itself is not funded.

### Claude

Claude uses the Anthropic messages API path. Make sure the provider, protocol, endpoint, and model belong to the same Anthropic account setup.

## Which Features Use External AI?

These UI actions can use the selected external model:

- **Create Sample** - not supported by external models until routing is implemented
- **Random Narrative Caption**
- **Enhance Caption**
- **Enhance Lyrics**
- **Generate Lyrics**
- metadata / planning tasks when thinking or CoT-style helpers are enabled

## New Text Features in Plain English

These tools are meant to work together.

### Create Sample

Start here when you only have a simple idea.

Example:

`dreamy synthpop with female vocals`

ACE-Step asks the text model to turn that into:

- a caption
- lyrics or instrumental marker
- BPM
- key
- time signature
- duration suggestion

### Random Narrative Caption

Use this when you want inspiration. It creates a fresh song concept from a random seed, then fills the caption box with something more detailed than a few genre words.

### Enhance Caption

Use this when your caption is too short, rough, or list-like.

Example:

`Tropical funk, female vocals`

becomes a fuller "standard caption" that usually includes:

- who is singing
- the singer's delivery or mood
- core instrumentation
- the song arc from intro to chorus/drop to outro
- the mix or energy progression

This is the best tool when your idea is good but under-described.

### Generate Lyrics

Use this when you already have a caption and want singable lyrics that fit the idea.

ACE-Step uses the current:

- caption
- vocal language
- BPM
- duration
- key
- time signature

This helps the generated lyrics better match the planned song.

### Enhance Lyrics

Use this when you already have lyrics, but want them cleaned up, structured better, or made more musical.

## Recommended Workflow

For a non-technical user, the easiest flow is:

1. write a short idea
2. click **Create Sample** or **Random Narrative Caption**
3. click **Enhance Caption** if the caption is still too simple
4. click **Generate Lyrics** if you want vocals
5. review the result and edit anything you want by hand
6. generate the music

## Troubleshooting

### The model list is empty

- make sure the endpoint is running
- click **Fetch Models**
- confirm the **Base URL** is correct
- check whether the provider requires an API key

### I saved the provider, but it does not seem active

Saving settings is not the final step. You still need to select the saved external entry in **Service Configuration** under **5Hz LM Model Path**.

### OpenAI says no quota

This usually means API billing is missing on the OpenAI platform project. A ChatGPT subscription alone is not enough.

### Z.ai coding endpoint gives quota errors

Your account may not have access to that coding-plan endpoint. Try the standard endpoint or another funded provider.

## See Also

- [Gradio Guide](GRADIO_GUIDE.md)
- [Installation Guide](INSTALL.md)
