# External LM Provider Setup

**Language / 语言 / 言語:** [English](EXTERNAL_LM_SETUP.md)

---

This guide explains how to configure ACE-Step to use external language-model providers for
external LM workflows in the Gradio UI.

## Table of Contents

- [Why Use External Providers](#why-use-external-providers)
- [What This Feature Does](#what-this-feature-does)
- [Open the External LM Panel](#open-the-external-lm-panel)
- [Provider Setup](#provider-setup)
- [Saved Settings](#saved-settings)
- [Known Caveats](#known-caveats)
- [Troubleshooting](#troubleshooting)

---

## Why Use External Providers

External providers can be a practical upgrade when you want a little more flexibility or a
little less pressure on your local machine.

Common reasons to use one instead of the built-in 5Hz LM include:

- **More capable text generation** for captions, lyrics, and related planning tasks
- **Lower local VRAM pressure** by moving LM work off the ACE-Step machine
- **Access to uncensored or differently tuned models** when your workflow needs them
- **Better use of other hardware you already own**, such as another GPU-equipped PC on your
  network handling the LM side of the workload

## What This Feature Does

ACE-Step can use an external language model instead of the built-in 5Hz LM for supported
external LM workflows in the UI.

Supported provider profiles currently include:

- **Z.ai**
- **OpenAI**
- **Claude**
- **Ollama**

## Open the External LM Panel

In the Gradio UI:

1. Open **Settings**
2. Open the **External LM** accordion
3. Choose a provider
4. Set the protocol, model, and endpoint
5. Enter an API key when required
6. Click **Save External LM Settings**

After saving, the configured external model is surfaced in the main LM picker.

## Provider Setup

### Z.ai

Typical setup:

- **Provider:** `zai`
- **Protocol:** `openai_chat`
- **Model:** your Z.ai model name
- **Base URL:** standard chat endpoint, or the coding endpoint if your account requires it

Notes:

- Some Z.ai models and plans need the **coding endpoint** rather than the standard chat
  endpoint.
- If a model fails with quota, credits, or access-style errors on the standard endpoint, try
  the coding endpoint for that provider.

### OpenAI

Typical setup:

- **Provider:** `openai`
- **Protocol:** `openai_chat`
- **Model:** an API model available to your OpenAI API account
- **Base URL:** the provider default unless you have a custom-compatible endpoint

Important:

- A **ChatGPT Plus** subscription does **not** include OpenAI API usage.
- ACE-Step needs a real **OpenAI API key with billing/API access**, not just a ChatGPT web
  subscription.

Testing note:

- OpenAI provider support is available, but testing coverage has been more limited than Z.ai
  and Ollama.

### Claude

Typical setup:

- **Provider:** `claude`
- **Protocol:** `anthropic_messages`
- **Model:** your Claude model name
- **Base URL:** the provider default unless your deployment uses a different endpoint

Testing note:

- Claude provider support is available, but real-world testing has been more limited than Z.ai
  and Ollama.

### Ollama

Typical setup:

- **Provider:** `ollama`
- **Protocol:** `openai_chat`
- **Model:** your Ollama model tag
- **Base URL:** your Ollama server URL

Examples:

- `http://127.0.0.1:11434/v1`
- `http://192.168.1.124:11434/v1`

Notes:

- ACE-Step supports both local and remote Ollama servers.

## Saved Settings

ACE-Step now stores provider settings **per provider**.

That means:

- Switching to Ollama does not overwrite your saved Z.ai settings
- Switching back restores the last saved configuration for that provider
- First run is safe even when no external-provider settings have been saved yet

## Known Caveats

### Coding endpoints may be required

Some provider/model combinations, especially on Z.ai, may require a **coding** endpoint rather
than the normal chat endpoint for planning-style requests.

If you see errors about:

- no credits
- missing access
- unsupported subscription plan
- model access denied

try the provider's coding endpoint or coding-capable model.

### OpenAI web plans are not API plans

ChatGPT Plus access does not automatically grant OpenAI API access.

If OpenAI setup appears correct but requests fail immediately, confirm that:

- the API key is valid
- API billing/access is enabled on the OpenAI account
- the selected model is available to that API account

### Claude and OpenAI testing is more limited

Compared with Z.ai and Ollama, Claude and OpenAI provider flows have had less real-world CER
and smoke-test coverage so far.

They should work, but if you are choosing a provider for the most battle-tested path today,
Z.ai and Ollama have seen more direct validation in ACE-Step.

### Prompt/response terminal logging is optional

ACE-Step includes an **LLM Enhancement Debug** toggle for printing enhancement prompts and raw
responses to the terminal.

Use it when diagnosing:

- bad JSON output
- provider formatting issues
- endpoint mismatches
- model-specific quirks

## Troubleshooting

### The model list is empty

Try:

1. Confirm the provider base URL is correct
2. Confirm the protocol matches the provider
3. Click **Get Models** to force a fresh fetch
4. Re-save settings if you changed endpoint or credentials

### The request works for one provider but not another

Check:

- provider-specific endpoint requirements
- whether the model actually exists on that account/server
- whether the account has API access for that model

### The provider returns access or credits errors

This usually means one of:

- wrong account tier
- wrong endpoint
- model not included in your plan
- API access missing even though a web subscription exists

---

For general UI usage, see the [Gradio Guide](GRADIO_GUIDE.md).
