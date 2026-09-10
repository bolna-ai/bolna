"use client";

import { useState } from "react";
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { Select } from "./ui/select";

const STEP_LABELS = ["Details", "Brain (LLM)", "Ears (ASR)", "Voice (TTS)"];

const LLM_MODELS = [
  { value: "gpt-4o", label: "GPT-4o (OpenAI)" },
  { value: "gpt-4o-mini", label: "GPT-4o Mini (OpenAI)" },
  { value: "gpt-3.5-turbo", label: "GPT-3.5 Turbo (OpenAI)" },
  { value: "claude-3-5-sonnet-20241022", label: "Claude 3.5 Sonnet (Anthropic)" },
  { value: "gemini/gemini-1.5-pro", label: "Gemini 1.5 Pro (Google)" },
];

const LLM_PROVIDERS = [
  { value: "openai", label: "OpenAI" },
  { value: "anthropic", label: "Anthropic" },
  { value: "google", label: "Google" },
  { value: "groq", label: "Groq" },
  { value: "litellm", label: "LiteLLM (Custom)" },
];

const ASR_PROVIDERS = [
  { value: "deepgram", label: "Deepgram" },
  { value: "assemblyai", label: "AssemblyAI" },
  { value: "azure", label: "Azure Speech" },
  { value: "google", label: "Google Speech" },
];

const DEEPGRAM_MODELS = [
  { value: "nova-2", label: "Nova-2 (Best)" },
  { value: "nova-2-phonecall", label: "Nova-2 Phone Call" },
  { value: "enhanced", label: "Enhanced" },
  { value: "base", label: "Base" },
];

const TTS_PROVIDERS = [
  { value: "elevenlabs", label: "ElevenLabs" },
  { value: "openai", label: "OpenAI TTS" },
  { value: "deepgram", label: "Deepgram Aura" },
  { value: "cartesia", label: "Cartesia" },
  { value: "polly", label: "Amazon Polly" },
];

const ELEVENLABS_VOICES = [
  { value: "21m00Tcm4TlvDq8ikWAM", label: "Rachel (Female, calm)" },
  { value: "AZnzlk1XvdvUeBnXmlld", label: "Domi (Female, energetic)" },
  { value: "EXAVITQu4vr4xnSDxMaL", label: "Bella (Female, soft)" },
  { value: "ErXwobaYiN019PkySvjV", label: "Antoni (Male, crisp)" },
  { value: "MF3mGyEYCl7XYWbV9V6O", label: "Elli (Female, young)" },
  { value: "TxGEqnHWrfWFTfGW9XjX", label: "Josh (Male, deep)" },
  { value: "VR6AewLTigWG4xSOukaG", label: "Arnold (Male, crisp)" },
  { value: "pNInz6obpgDQGcFmaJgB", label: "Adam (Male, deep)" },
];

const OPENAI_VOICES = [
  { value: "alloy", label: "Alloy (Neutral)" },
  { value: "echo", label: "Echo (Male)" },
  { value: "fable", label: "Fable (British)" },
  { value: "onyx", label: "Onyx (Male, deep)" },
  { value: "nova", label: "Nova (Female)" },
  { value: "shimmer", label: "Shimmer (Female, soft)" },
];

interface AgentFormData {
  name: string;
  description: string;
  system_prompt: string;
  welcome_message: string;
  llm_model: string;
  llm_provider: string;
  llm_max_tokens: string;
  llm_temperature: string;
  asr_provider: string;
  asr_model: string;
  asr_language: string;
  tts_provider: string;
  tts_voice: string;
  tts_voice_id: string;
  tts_model: string;
}

const DEFAULT_FORM: AgentFormData = {
  name: "",
  description: "",
  system_prompt:
    "You are a helpful AI voice assistant. Keep your responses concise and natural for voice conversations. Avoid using markdown, bullet points, or special characters.",
  welcome_message: "Hi! How can I help you today?",
  llm_model: "gpt-4o",
  llm_provider: "openai",
  llm_max_tokens: "150",
  llm_temperature: "0.1",
  asr_provider: "deepgram",
  asr_model: "nova-2",
  asr_language: "en",
  tts_provider: "elevenlabs",
  tts_voice: "Rachel",
  tts_voice_id: "21m00Tcm4TlvDq8ikWAM",
  tts_model: "eleven_turbo_v2",
};

function buildBolnaConfig(form: AgentFormData) {
  const synthConfig = buildSynthesizerConfig(form);

  return {
    agent_name: form.name,
    agent_type: "other",
    agent_welcome_message: form.welcome_message,
    tasks: [
      {
        task_type: "conversation",
        tools_config: {
          transcriber: {
            provider: form.asr_provider,
            model: form.asr_model,
            language: form.asr_language || null,
            stream: true,
            endpointing: 500,
          },
          synthesizer: synthConfig,
          llm_agent: {
            agent_flow_type: "streaming",
            agent_type: "simple_llm_agent",
            llm_config: {
              model: form.llm_model,
              provider: form.llm_provider,
              max_tokens: parseInt(form.llm_max_tokens, 10),
              temperature: parseFloat(form.llm_temperature),
              family: form.llm_provider,
            },
          },
          input: { provider: "twilio", format: "wav" },
          output: { provider: "twilio", format: "wav" },
        },
        task_config: {},
      },
    ],
  };
}

function buildSynthesizerConfig(form: AgentFormData) {
  const base = {
    stream: true,
    buffer_size: 40,
    audio_format: "pcm",
    caching: true,
  };

  if (form.tts_provider === "elevenlabs") {
    return {
      ...base,
      provider: "elevenlabs",
      provider_config: {
        voice: form.tts_voice,
        voice_id: form.tts_voice_id,
        model: form.tts_model || "eleven_turbo_v2",
        temperature: 0.5,
        similarity_boost: 0.75,
        speed: 1.0,
        style: 0.0,
      },
    };
  }

  if (form.tts_provider === "openai") {
    return {
      ...base,
      provider: "openai",
      provider_config: {
        voice: form.tts_voice,
        model: "tts-1",
      },
    };
  }

  if (form.tts_provider === "deepgram") {
    return {
      ...base,
      provider: "deepgram",
      provider_config: {
        voice_id: form.tts_voice_id || "aura-asteria-en",
        voice: form.tts_voice || "aura-asteria-en",
        model: "aura-asteria-en",
      },
    };
  }

  return {
    ...base,
    provider: form.tts_provider,
    provider_config: {
      voice: form.tts_voice,
      voice_id: form.tts_voice_id,
      model: form.tts_model,
    },
  };
}

function buildPrompts(form: AgentFormData) {
  return {
    task_1: {
      system_prompt: form.system_prompt,
    },
  };
}

interface AgentBuilderProps {
  initialData?: Partial<AgentFormData>;
  onSubmit: (bolnaConfig: unknown, prompts: unknown, name: string, description: string) => Promise<void>;
  onCancel?: () => void;
  submitLabel?: string;
  isLoading?: boolean;
}

export function AgentBuilder({
  initialData,
  onSubmit,
  onCancel,
  submitLabel = "Create Agent",
  isLoading = false,
}: AgentBuilderProps) {
  const [step, setStep] = useState(0);
  const [form, setForm] = useState<AgentFormData>({
    ...DEFAULT_FORM,
    ...initialData,
  });
  const [errors, setErrors] = useState<Partial<AgentFormData>>({});

  const update = (key: keyof AgentFormData, value: string) => {
    setForm((prev) => ({ ...prev, [key]: value }));
    if (errors[key]) setErrors((prev) => ({ ...prev, [key]: undefined }));
  };

  const validateStep = (s: number): boolean => {
    const errs: Partial<AgentFormData> = {};
    if (s === 0) {
      if (!form.name.trim()) errs.name = "Agent name is required";
      if (!form.welcome_message.trim()) errs.welcome_message = "Welcome message is required";
    }
    if (s === 1) {
      if (!form.system_prompt.trim()) errs.system_prompt = "System prompt is required";
    }
    setErrors(errs);
    return Object.keys(errs).length === 0;
  };

  const next = () => {
    if (validateStep(step)) setStep((s) => Math.min(s + 1, STEP_LABELS.length - 1));
  };

  const back = () => setStep((s) => Math.max(s - 1, 0));

  const handleSubmit = async () => {
    for (let i = 0; i <= step; i++) {
      if (!validateStep(i)) {
        setStep(i);
        return;
      }
    }
    const bolnaConfig = buildBolnaConfig(form);
    const prompts = buildPrompts(form);
    await onSubmit(bolnaConfig, prompts, form.name, form.description);
  };

  const ttsVoices =
    form.tts_provider === "elevenlabs"
      ? ELEVENLABS_VOICES
      : form.tts_provider === "openai"
      ? OPENAI_VOICES
      : [{ value: "default", label: "Default" }];

  return (
    <div className="flex flex-col gap-6">
      {/* Stepper */}
      <nav aria-label="Progress">
        <ol className="flex items-center gap-2">
          {STEP_LABELS.map((label, i) => (
            <li key={label} className="flex items-center gap-2">
              <button
                type="button"
                onClick={() => i < step && setStep(i)}
                className={`flex h-8 w-8 items-center justify-center rounded-full text-sm font-medium transition-colors ${
                  i === step
                    ? "bg-brand-600 text-white"
                    : i < step
                    ? "bg-brand-100 text-brand-700 cursor-pointer hover:bg-brand-200"
                    : "bg-gray-100 text-gray-400 cursor-default"
                }`}
              >
                {i < step ? "✓" : i + 1}
              </button>
              <span
                className={`text-sm font-medium ${
                  i === step ? "text-brand-700" : "text-gray-500"
                }`}
              >
                {label}
              </span>
              {i < STEP_LABELS.length - 1 && (
                <div className="mx-1 h-px w-8 bg-gray-200" />
              )}
            </li>
          ))}
        </ol>
      </nav>

      {/* Step 0: Agent Details */}
      {step === 0 && (
        <div className="flex flex-col gap-4">
          <Input
            label="Agent Name *"
            placeholder="e.g. Customer Support Bot"
            value={form.name}
            onChange={(e) => update("name", e.target.value)}
            error={errors.name}
          />
          <div className="flex flex-col gap-1">
            <label className="text-sm font-medium text-gray-700">
              Description
            </label>
            <textarea
              className="block w-full rounded-lg border border-gray-300 px-3 py-2 text-sm shadow-sm placeholder-gray-400 focus:border-brand-500 focus:outline-none focus:ring-1 focus:ring-brand-500"
              rows={2}
              placeholder="What does this agent do?"
              value={form.description}
              onChange={(e) => update("description", e.target.value)}
            />
          </div>
          <Input
            label="Welcome Message *"
            placeholder="Hi! How can I help you today?"
            value={form.welcome_message}
            onChange={(e) => update("welcome_message", e.target.value)}
            error={errors.welcome_message}
            hint="The first thing your agent says when a call connects."
          />
        </div>
      )}

      {/* Step 1: Brain (LLM) */}
      {step === 1 && (
        <div className="flex flex-col gap-4">
          <div className="flex flex-col gap-1">
            <label className="text-sm font-medium text-gray-700">
              System Prompt *
            </label>
            <textarea
              className={`block w-full rounded-lg border px-3 py-2 text-sm shadow-sm placeholder-gray-400 focus:border-brand-500 focus:outline-none focus:ring-1 focus:ring-brand-500 ${
                errors.system_prompt ? "border-red-500" : "border-gray-300"
              }`}
              rows={6}
              value={form.system_prompt}
              onChange={(e) => update("system_prompt", e.target.value)}
              placeholder="You are a helpful AI voice assistant..."
            />
            {errors.system_prompt && (
              <p className="text-xs text-red-600">{errors.system_prompt}</p>
            )}
            <p className="text-xs text-gray-500">
              Keep responses concise for voice. Avoid markdown or bullet points.
            </p>
          </div>
          <div className="grid grid-cols-2 gap-4">
            <Select
              label="LLM Provider"
              options={LLM_PROVIDERS}
              value={form.llm_provider}
              onChange={(e) => update("llm_provider", e.target.value)}
            />
            <Select
              label="Model"
              options={LLM_MODELS}
              value={form.llm_model}
              onChange={(e) => update("llm_model", e.target.value)}
            />
          </div>
          <div className="grid grid-cols-2 gap-4">
            <Input
              label="Max Tokens"
              type="number"
              min="50"
              max="2000"
              value={form.llm_max_tokens}
              onChange={(e) => update("llm_max_tokens", e.target.value)}
              hint="Keep low (100–200) for natural voice responses."
            />
            <Input
              label="Temperature"
              type="number"
              min="0"
              max="2"
              step="0.1"
              value={form.llm_temperature}
              onChange={(e) => update("llm_temperature", e.target.value)}
              hint="0.1 for focused, 0.8 for creative responses."
            />
          </div>
        </div>
      )}

      {/* Step 2: Ears (ASR) */}
      {step === 2 && (
        <div className="flex flex-col gap-4">
          <Select
            label="Transcription Provider"
            options={ASR_PROVIDERS}
            value={form.asr_provider}
            onChange={(e) => update("asr_provider", e.target.value)}
          />
          {form.asr_provider === "deepgram" && (
            <Select
              label="Deepgram Model"
              options={DEEPGRAM_MODELS}
              value={form.asr_model}
              onChange={(e) => update("asr_model", e.target.value)}
            />
          )}
          <Input
            label="Language"
            placeholder="en"
            value={form.asr_language}
            onChange={(e) => update("asr_language", e.target.value)}
            hint="BCP-47 language code (e.g. en, es, fr, de)"
          />
          <div className="rounded-lg border border-blue-100 bg-blue-50 p-3 text-sm text-blue-800">
            <strong>Tip:</strong> Deepgram nova-2-phonecall is optimized for phone audio quality.
          </div>
        </div>
      )}

      {/* Step 3: Voice (TTS) */}
      {step === 3 && (
        <div className="flex flex-col gap-4">
          <Select
            label="Voice Provider"
            options={TTS_PROVIDERS}
            value={form.tts_provider}
            onChange={(e) => {
              update("tts_provider", e.target.value);
              if (e.target.value === "elevenlabs") {
                update("tts_voice", "Rachel");
                update("tts_voice_id", "21m00Tcm4TlvDq8ikWAM");
                update("tts_model", "eleven_turbo_v2");
              } else if (e.target.value === "openai") {
                update("tts_voice", "nova");
                update("tts_voice_id", "nova");
                update("tts_model", "tts-1");
              }
            }}
          />
          <Select
            label="Voice"
            options={ttsVoices}
            value={form.tts_voice_id || form.tts_voice}
            onChange={(e) => {
              const opt = ttsVoices.find((v) => v.value === e.target.value);
              update("tts_voice_id", e.target.value);
              update("tts_voice", opt?.label.split(" ")[0] ?? e.target.value);
            }}
          />
          {form.tts_provider === "elevenlabs" && (
            <>
              <Input
                label="ElevenLabs Voice ID"
                value={form.tts_voice_id}
                onChange={(e) => update("tts_voice_id", e.target.value)}
                hint="Custom voice ID from your ElevenLabs library (optional)."
              />
              <Select
                label="ElevenLabs Model"
                options={[
                  { value: "eleven_turbo_v2", label: "Turbo v2 (Fastest)" },
                  { value: "eleven_turbo_v2_5", label: "Turbo v2.5 (Multilingual)" },
                  { value: "eleven_multilingual_v2", label: "Multilingual v2 (Best quality)" },
                  { value: "eleven_monolingual_v1", label: "Monolingual v1 (English only)" },
                ]}
                value={form.tts_model}
                onChange={(e) => update("tts_model", e.target.value)}
              />
            </>
          )}
        </div>
      )}

      {/* Navigation */}
      <div className="flex items-center justify-between border-t border-gray-200 pt-4">
        <div>
          {onCancel && step === 0 && (
            <Button variant="ghost" onClick={onCancel}>
              Cancel
            </Button>
          )}
          {step > 0 && (
            <Button variant="secondary" onClick={back}>
              ← Back
            </Button>
          )}
        </div>
        <div>
          {step < STEP_LABELS.length - 1 ? (
            <Button onClick={next}>Next →</Button>
          ) : (
            <Button onClick={handleSubmit} loading={isLoading}>
              {submitLabel}
            </Button>
          )}
        </div>
      </div>
    </div>
  );
}
