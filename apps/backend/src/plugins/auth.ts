import fp from "fastify-plugin";
import type { FastifyPluginAsync, FastifyRequest, FastifyReply } from "fastify";
import { createClient } from "@supabase/supabase-js";
import type { Database } from "../lib/supabase.js";

declare module "fastify" {
  interface FastifyRequest {
    userId: string;
    organizationId: string;
    userRole: string;
  }
  interface FastifyInstance {
    authenticate: (
      request: FastifyRequest,
      reply: FastifyReply
    ) => Promise<void>;
  }
}

const authPlugin: FastifyPluginAsync = fp(async (fastify) => {
  fastify.decorate(
    "authenticate",
    async (request: FastifyRequest, reply: FastifyReply) => {
      const authHeader = request.headers.authorization;
      if (!authHeader?.startsWith("Bearer ")) {
        return reply.status(401).send({ error: "Missing authorization header" });
      }

      const token = authHeader.slice(7);
      const supabase = createClient<Database>(
        process.env.SUPABASE_URL!,
        process.env.SUPABASE_ANON_KEY!,
        {
          global: { headers: { Authorization: `Bearer ${token}` } },
          auth: { autoRefreshToken: false, persistSession: false },
        }
      );

      const {
        data: { user },
        error,
      } = await supabase.auth.getUser(token);

      if (error || !user) {
        return reply.status(401).send({ error: "Invalid or expired token" });
      }

      const { data: profileData, error: profileError } = await supabase
        .from("profiles")
        .select("organization_id, role")
        .eq("id", user.id)
        .single();

      type ProfileRow = Database["public"]["Tables"]["profiles"]["Row"];
      const profile = profileData as ProfileRow | null;

      if (profileError || !profile?.organization_id) {
        return reply
          .status(403)
          .send({ error: "User has no organization. Please complete onboarding." });
      }

      request.userId = user.id;
      request.organizationId = profile.organization_id;
      request.userRole = profile.role;
    }
  );
});

export default authPlugin;
