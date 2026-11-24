import type { FastifyPluginAsync } from "fastify";
import { z } from "zod";
import { getSupabaseAdmin, type Database } from "../lib/supabase.js";
import { syncAgentToBolna, deleteAgentFromBolna } from "../lib/bolna.js";

type AgentRow = Database["public"]["Tables"]["agents"]["Row"];
type OrgRow = Database["public"]["Tables"]["organizations"]["Row"];

const TranscriberSchema = z.object({
  provider: z.string().default("deepgram"),
  model: z.string().default("nova-2"),
  language: z.string().nullable().optional(),
  stream: z.boolean().default(true),
  endpointing: z.number().int().default(500),
  keywords: z.string().nullable().optional(),
});

const SynthesizerSchema = z.object({
  provider: z.string(),
  provider_config: z.record(z.unknown()),
  stream: z.boolean().default(true),
  buffer_size: z.number().int().default(40),
  audio_format: z.string().default("pcm"),
  caching: z.boolean().default(true),
});

const LlmSchema = z.object({
  model: z.string().default("gpt-3.5-turbo"),
  provider: z.string().default("openai"),
  max_tokens: z.number().int().default(100),
  temperature: z.number().default(0.1),
  family: z.string().default("openai"),
});

const TaskToolsConfigSchema = z.object({
  transcriber: TranscriberSchema,
  synthesizer: SynthesizerSchema,
  llm_agent: z
    .object({
      agent_flow_type: z.string().default("streaming"),
      agent_type: z.string().default("simple_llm_agent"),
      llm_config: LlmSchema,
    })
    .optional(),
  input: z
    .object({ provider: z.string().default("twilio"), format: z.string().default("wav") })
    .optional(),
  output: z
    .object({ provider: z.string().default("twilio"), format: z.string().default("wav") })
    .optional(),
});

const TaskSchema = z.object({
  task_type: z.string().default("conversation"),
  tools_config: TaskToolsConfigSchema,
  task_config: z.record(z.unknown()).optional(),
});

const BolnaConfigSchema = z.object({
  agent_name: z.string(),
  agent_type: z.string().default("other"),
  agent_welcome_message: z.string().optional(),
  tasks: z.array(TaskSchema),
});

const PromptsSchema = z.record(z.record(z.string()));

const AgentCreateSchema = z.object({
  name: z.string().min(1).max(100),
  description: z.string().max(500).optional(),
  bolna_config: BolnaConfigSchema,
  prompts: PromptsSchema.default({}),
});

const AgentUpdateSchema = AgentCreateSchema.partial();

const agentsRoutes: FastifyPluginAsync = async (fastify) => {
  fastify.addHook("onRequest", fastify.authenticate);

  fastify.get("/", async (request, reply) => {
    const supabase = getSupabaseAdmin();
    const { data, error } = await supabase
      .from("agents")
      .select("id, name, description, status, bolna_agent_id, created_at, updated_at")
      .eq("organization_id", request.organizationId)
      .order("created_at", { ascending: false });

    if (error) {
      request.log.error(error, "Failed to fetch agents");
      return reply.status(500).send({ error: "Failed to fetch agents" });
    }
    return { agents: data as Partial<AgentRow>[] };
  });

  fastify.get<{ Params: { id: string } }>("/:id", async (request, reply) => {
    const supabase = getSupabaseAdmin();
    const { data, error } = await supabase
      .from("agents")
      .select("*")
      .eq("id", request.params.id)
      .eq("organization_id", request.organizationId)
      .single();

    if (error || !data) {
      return reply.status(404).send({ error: "Agent not found" });
    }
    return { agent: data as AgentRow };
  });

  fastify.post("/", async (request, reply) => {
    const parsed = AgentCreateSchema.safeParse(request.body);
    if (!parsed.success) {
      return reply.status(400).send({ error: parsed.error.flatten() });
    }

    const supabase = getSupabaseAdmin();
    const { data: orgData, error: orgError } = await supabase
      .from("organizations")
      .select("bolna_engine_url")
      .eq("id", request.organizationId)
      .single();

    const org = orgData as Pick<OrgRow, "bolna_engine_url"> | null;

    if (orgError || !org) {
      return reply.status(500).send({ error: "Failed to fetch organization config" });
    }

    let bolnaAgentId: string | undefined;
    try {
      bolnaAgentId = await syncAgentToBolna(
        org.bolna_engine_url,
        JSON.parse(JSON.stringify(parsed.data.bolna_config)),
        JSON.parse(JSON.stringify(parsed.data.prompts))
      );
    } catch (err) {
      request.log.warn(err, "Bolna sync failed — saving agent without engine registration");
    }

    const { data: agent, error } = await supabase
      .from("agents")
      .insert({
        organization_id: request.organizationId,
        created_by: request.userId,
        name: parsed.data.name,
        description: parsed.data.description ?? null,
        bolna_config: JSON.parse(JSON.stringify(parsed.data.bolna_config)),
        prompts: JSON.parse(JSON.stringify(parsed.data.prompts)),
        bolna_agent_id: bolnaAgentId ?? null,
        status: "active",
      })
      .select()
      .single();

    if (error || !agent) {
      request.log.error(error, "Failed to create agent");
      return reply.status(500).send({ error: "Failed to create agent" });
    }

    return reply.status(201).send({ agent: agent as AgentRow });
  });

  fastify.put<{ Params: { id: string } }>("/:id", async (request, reply) => {
    const parsed = AgentUpdateSchema.safeParse(request.body);
    if (!parsed.success) {
      return reply.status(400).send({ error: parsed.error.flatten() });
    }

    const supabase = getSupabaseAdmin();
    const { data: existingData, error: fetchError } = await supabase
      .from("agents")
      .select("*")
      .eq("id", request.params.id)
      .eq("organization_id", request.organizationId)
      .single();

    const existing = existingData as AgentRow | null;

    if (fetchError || !existing) {
      return reply.status(404).send({ error: "Agent not found" });
    }

    const updates: Database["public"]["Tables"]["agents"]["Update"] = {};
    if (parsed.data.name !== undefined) updates.name = parsed.data.name;
    if (parsed.data.description !== undefined) updates.description = parsed.data.description;
    if (parsed.data.bolna_config !== undefined)
      updates.bolna_config = JSON.parse(JSON.stringify(parsed.data.bolna_config));
    if (parsed.data.prompts !== undefined)
      updates.prompts = JSON.parse(JSON.stringify(parsed.data.prompts));

    if (parsed.data.bolna_config !== undefined || parsed.data.prompts !== undefined) {
      const { data: orgData } = await supabase
        .from("organizations")
        .select("bolna_engine_url")
        .eq("id", request.organizationId)
        .single();

      const org = orgData as Pick<OrgRow, "bolna_engine_url"> | null;

      if (org) {
        try {
          const newBolnaId = await syncAgentToBolna(
            org.bolna_engine_url,
            JSON.parse(JSON.stringify(parsed.data.bolna_config ?? existing.bolna_config)),
            JSON.parse(JSON.stringify(parsed.data.prompts ?? existing.prompts)),
            existing.bolna_agent_id
          );
          updates.bolna_agent_id = newBolnaId;
        } catch (err) {
          request.log.warn(err, "Bolna sync failed during update");
        }
      }
    }

    const { data: agent, error } = await supabase
      .from("agents")
      .update(updates)
      .eq("id", request.params.id)
      .eq("organization_id", request.organizationId)
      .select()
      .single();

    if (error || !agent) {
      request.log.error(error, "Failed to update agent");
      return reply.status(500).send({ error: "Failed to update agent" });
    }

    return { agent: agent as AgentRow };
  });

  fastify.delete<{ Params: { id: string } }>("/:id", async (request, reply) => {
    const supabase = getSupabaseAdmin();
    const { data: existingData, error: fetchError } = await supabase
      .from("agents")
      .select("bolna_agent_id, organization_id")
      .eq("id", request.params.id)
      .eq("organization_id", request.organizationId)
      .single();

    const existing = existingData as Pick<AgentRow, "bolna_agent_id" | "organization_id"> | null;

    if (fetchError || !existing) {
      return reply.status(404).send({ error: "Agent not found" });
    }

    if (existing.bolna_agent_id) {
      const { data: orgData } = await supabase
        .from("organizations")
        .select("bolna_engine_url")
        .eq("id", request.organizationId)
        .single();

      const org = orgData as Pick<OrgRow, "bolna_engine_url"> | null;

      if (org) {
        try {
          await deleteAgentFromBolna(org.bolna_engine_url, existing.bolna_agent_id);
        } catch (err) {
          request.log.warn(err, "Bolna engine delete failed — proceeding with DB delete");
        }
      }
    }

    const { error } = await supabase
      .from("agents")
      .delete()
      .eq("id", request.params.id)
      .eq("organization_id", request.organizationId);

    if (error) {
      request.log.error(error, "Failed to delete agent");
      return reply.status(500).send({ error: "Failed to delete agent" });
    }

    return reply.status(204).send();
  });
};

export default agentsRoutes;
