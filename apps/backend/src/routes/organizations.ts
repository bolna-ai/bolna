import type { FastifyPluginAsync } from "fastify";
import { createClient } from "@supabase/supabase-js";
import { z } from "zod";
import { getSupabaseAdmin, type Database } from "../lib/supabase.js";

type OrgRow = Database["public"]["Tables"]["organizations"]["Row"];
type ProfileRow = Database["public"]["Tables"]["profiles"]["Row"];

const OrgUpdateSchema = z.object({
  name: z.string().min(1).max(100).optional(),
  bolna_engine_url: z.string().url().optional(),
  twilio_account_sid: z.string().optional(),
  twilio_auth_token: z.string().optional(),
  twilio_phone_number: z.string().optional(),
  plivo_auth_id: z.string().optional(),
  plivo_auth_token: z.string().optional(),
  plivo_phone_number: z.string().optional(),
});

const OnboardingSchema = z.object({
  organization_name: z.string().min(1).max(100),
  slug: z
    .string()
    .min(2)
    .max(50)
    .regex(/^[a-z0-9-]+$/),
});

const organizationRoutes: FastifyPluginAsync = async (fastify) => {
  fastify.post("/onboard", async (request, reply) => {
    const authHeader = request.headers.authorization;
    if (!authHeader?.startsWith("Bearer ")) {
      return reply.status(401).send({ error: "Unauthorized" });
    }
    const token = authHeader.slice(7);

    const supabaseUser = createClient<Database>(
      process.env.SUPABASE_URL!,
      process.env.SUPABASE_ANON_KEY!,
      {
        global: { headers: { Authorization: `Bearer ${token}` } },
        auth: { autoRefreshToken: false, persistSession: false },
      }
    );

    const {
      data: { user },
      error: authError,
    } = await supabaseUser.auth.getUser(token);

    if (authError || !user) {
      return reply.status(401).send({ error: "Invalid token" });
    }

    const parsed = OnboardingSchema.safeParse(request.body);
    if (!parsed.success) {
      return reply.status(400).send({ error: parsed.error.flatten() });
    }

    const supabase = getSupabaseAdmin();

    const { data: existingProfileData } = await supabase
      .from("profiles")
      .select("organization_id")
      .eq("id", user.id)
      .single();

    const existingProfile = existingProfileData as Pick<ProfileRow, "organization_id"> | null;

    if (existingProfile?.organization_id) {
      return reply.status(409).send({ error: "User already belongs to an organization" });
    }

    const { data: orgData, error: orgError } = await supabase
      .from("organizations")
      .insert({
        name: parsed.data.organization_name,
        slug: parsed.data.slug,
        plan: "free",
        bolna_engine_url: process.env.BOLNA_ENGINE_URL ?? "http://localhost:8000",
      })
      .select()
      .single();

    const org = orgData as OrgRow | null;

    if (orgError || !org) {
      if (orgError?.code === "23505") {
        return reply.status(409).send({ error: "Organization slug already taken" });
      }
      request.log.error(orgError, "Failed to create organization");
      return reply.status(500).send({ error: "Failed to create organization" });
    }

    const { error: profileError } = await supabase
      .from("profiles")
      .update({ organization_id: org.id, role: "owner" })
      .eq("id", user.id);

    if (profileError) {
      request.log.error(profileError, "Failed to link user to organization");
      return reply.status(500).send({ error: "Failed to link user to organization" });
    }

    return reply.status(201).send({ organization: org });
  });

  fastify.get(
    "/me",
    { onRequest: [fastify.authenticate] },
    async (request, reply) => {
      const supabase = getSupabaseAdmin();
      const { data: orgData, error } = await supabase
        .from("organizations")
        .select(
          "id, name, slug, plan, bolna_engine_url, twilio_phone_number, plivo_phone_number, created_at"
        )
        .eq("id", request.organizationId)
        .single();

      const org = orgData as Partial<OrgRow> | null;

      if (error || !org) {
        return reply.status(404).send({ error: "Organization not found" });
      }

      return { organization: org };
    }
  );

  fastify.put(
    "/me",
    { onRequest: [fastify.authenticate] },
    async (request, reply) => {
      if (!["owner", "admin"].includes(request.userRole)) {
        return reply.status(403).send({ error: "Insufficient permissions" });
      }

      const parsed = OrgUpdateSchema.safeParse(request.body);
      if (!parsed.success) {
        return reply.status(400).send({ error: parsed.error.flatten() });
      }

      const supabase = getSupabaseAdmin();
      const { data: orgData, error } = await supabase
        .from("organizations")
        .update(parsed.data)
        .eq("id", request.organizationId)
        .select()
        .single();

      const org = orgData as OrgRow | null;

      if (error || !org) {
        request.log.error(error, "Failed to update organization");
        return reply.status(500).send({ error: "Failed to update organization" });
      }

      return { organization: org };
    }
  );
};

export default organizationRoutes;
