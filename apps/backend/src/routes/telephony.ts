import type { FastifyPluginAsync } from "fastify";
import twilio from "twilio";
import { z } from "zod";
import { getSupabaseAdmin, type Database } from "../lib/supabase.js";

type AgentRow = Database["public"]["Tables"]["agents"]["Row"];
type OrgRow = Database["public"]["Tables"]["organizations"]["Row"];
type CallRow = Database["public"]["Tables"]["calls"]["Row"];

const OutboundCallSchema = z.object({
  agent_id: z.string().uuid(),
  to_number: z.string().min(7),
});

const CallStatusSchema = z.object({
  CallSid: z.string(),
  CallStatus: z.string(),
  Duration: z.string().optional(),
  RecordingUrl: z.string().optional(),
});

const telephonyRoutes: FastifyPluginAsync = async (fastify) => {
  fastify.post<{ Params: { agentId: string } }>(
    "/twilio/connect/:agentId",
    async (request, reply) => {
      const { agentId } = request.params;
      const supabase = getSupabaseAdmin();

      const { data: agentData, error } = await supabase
        .from("agents")
        .select("bolna_agent_id, organization_id")
        .eq("id", agentId)
        .single();

      const agent = agentData as Pick<AgentRow, "bolna_agent_id" | "organization_id"> | null;

      if (error || !agent) {
        return reply.status(404).send("Agent not found");
      }

      const bolnaAgentId = agent.bolna_agent_id ?? agentId;

      const { data: orgData } = await supabase
        .from("organizations")
        .select("bolna_engine_url")
        .eq("id", agent.organization_id)
        .single();

      const org = orgData as Pick<OrgRow, "bolna_engine_url"> | null;
      const bolnaEngineUrl =
        org?.bolna_engine_url ??
        process.env.BOLNA_ENGINE_URL ??
        "http://localhost:8000";
      const bolnaWsUrl = bolnaEngineUrl
        .replace(/^https:/, "wss:")
        .replace(/^http:/, "ws:");

      const VoiceResponse = twilio.twiml.VoiceResponse;
      const twiml = new VoiceResponse();
      const connect = twiml.connect();
      connect.stream({ url: `${bolnaWsUrl}/chat/v1/${bolnaAgentId}` });

      reply.header("Content-Type", "text/xml");
      return reply.send(twiml.toString());
    }
  );

  fastify.post<{ Params: { agentId: string } }>(
    "/twilio/status/:agentId",
    async (request, reply) => {
      const { agentId } = request.params;
      const parsed = CallStatusSchema.safeParse(request.body);
      if (!parsed.success) {
        return reply.status(400).send({ error: "Invalid callback payload" });
      }

      const supabase = getSupabaseAdmin();
      const { data: callData } = await supabase
        .from("calls")
        .select("id")
        .eq("agent_id", agentId)
        .eq("call_sid", parsed.data.CallSid)
        .single();

      const call = callData as Pick<CallRow, "id"> | null;

      if (call) {
        await supabase
          .from("calls")
          .update({
            status: parsed.data.CallStatus,
            duration_seconds: parsed.data.Duration
              ? parseInt(parsed.data.Duration, 10)
              : null,
            recording_url: parsed.data.RecordingUrl ?? null,
            ended_at: new Date().toISOString(),
          })
          .eq("id", call.id);
      }

      return reply.status(204).send();
    }
  );

  fastify.post(
    "/calls/outbound",
    { onRequest: [fastify.authenticate] },
    async (request, reply) => {
      const parsed = OutboundCallSchema.safeParse(request.body);
      if (!parsed.success) {
        return reply.status(400).send({ error: parsed.error.flatten() });
      }

      const supabase = getSupabaseAdmin();
      const { data: agentData, error: agentError } = await supabase
        .from("agents")
        .select("id, bolna_agent_id, status")
        .eq("id", parsed.data.agent_id)
        .eq("organization_id", request.organizationId)
        .single();

      const agent = agentData as Pick<AgentRow, "id" | "bolna_agent_id" | "status"> | null;

      if (agentError || !agent) {
        return reply.status(404).send({ error: "Agent not found" });
      }

      if (agent.status !== "active") {
        return reply.status(422).send({ error: "Agent is not active" });
      }

      const { data: orgData, error: orgError } = await supabase
        .from("organizations")
        .select(
          "twilio_account_sid, twilio_auth_token, twilio_phone_number, bolna_engine_url"
        )
        .eq("id", request.organizationId)
        .single();

      const org = orgData as Pick<
        OrgRow,
        | "twilio_account_sid"
        | "twilio_auth_token"
        | "twilio_phone_number"
        | "bolna_engine_url"
      > | null;

      if (orgError || !org?.twilio_account_sid) {
        return reply
          .status(422)
          .send({ error: "Twilio credentials not configured for this organization" });
      }

      const twilioClient = twilio(org.twilio_account_sid, org.twilio_auth_token!);
      const backendPublicUrl =
        process.env.BACKEND_PUBLIC_URL ??
        `http://localhost:${process.env.PORT ?? 3001}`;

      let callSid: string;
      try {
        const call = await twilioClient.calls.create({
          to: parsed.data.to_number,
          from: org.twilio_phone_number!,
          url: `${backendPublicUrl}/api/telephony/twilio/connect/${parsed.data.agent_id}`,
          statusCallback: `${backendPublicUrl}/api/telephony/twilio/status/${parsed.data.agent_id}`,
          statusCallbackMethod: "POST",
          record: true,
        });
        callSid = call.sid;
      } catch (err) {
        request.log.error(err, "Twilio call creation failed");
        return reply.status(502).send({ error: "Failed to initiate call via Twilio" });
      }

      const { data: callRecord, error: callInsertError } = await supabase
        .from("calls")
        .insert({
          organization_id: request.organizationId,
          agent_id: parsed.data.agent_id,
          call_sid: callSid,
          direction: "outbound",
          from_number: org.twilio_phone_number,
          to_number: parsed.data.to_number,
          status: "initiated",
          started_at: new Date().toISOString(),
        })
        .select()
        .single();

      if (callInsertError) {
        request.log.warn(callInsertError, "Failed to log call to database");
      }

      const callData2 = callRecord as CallRow | null;

      return reply.status(202).send({
        call_sid: callSid,
        call_id: callData2?.id ?? null,
        status: "initiated",
      });
    }
  );

  fastify.get(
    "/calls",
    { onRequest: [fastify.authenticate] },
    async (request, reply) => {
      const query = request.query as {
        agent_id?: string;
        limit?: string;
        offset?: string;
      };
      const limit = Math.min(parseInt(query.limit ?? "50", 10), 200);
      const offset = parseInt(query.offset ?? "0", 10);

      const supabase = getSupabaseAdmin();
      let q = supabase
        .from("calls")
        .select(
          "id, agent_id, call_sid, direction, from_number, to_number, status, duration_seconds, recording_url, started_at, ended_at, created_at",
          { count: "exact" }
        )
        .eq("organization_id", request.organizationId)
        .order("created_at", { ascending: false })
        .range(offset, offset + limit - 1);

      if (query.agent_id) {
        q = q.eq("agent_id", query.agent_id);
      }

      const { data, error, count } = await q;

      if (error) {
        request.log.error(error, "Failed to fetch calls");
        return reply.status(500).send({ error: "Failed to fetch calls" });
      }

      return { calls: data as Partial<CallRow>[], total: count };
    }
  );
};

export default telephonyRoutes;
