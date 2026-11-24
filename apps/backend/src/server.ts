import Fastify from "fastify";
import cors from "@fastify/cors";
import helmet from "@fastify/helmet";
import rateLimit from "@fastify/rate-limit";

import authPlugin from "./plugins/auth.js";
import agentsRoutes from "./routes/agents.js";
import telephonyRoutes from "./routes/telephony.js";
import organizationRoutes from "./routes/organizations.js";

const PORT = parseInt(process.env.PORT ?? "3001", 10);
const HOST = process.env.HOST ?? "0.0.0.0";

async function buildServer() {
  const fastify = Fastify({
    logger: {
      level: process.env.LOG_LEVEL ?? "info",
      transport:
        process.env.NODE_ENV !== "production"
          ? { target: "pino-pretty", options: { colorize: true } }
          : undefined,
    },
  });

  await fastify.register(helmet, { contentSecurityPolicy: false });
  await fastify.register(cors, {
    origin: process.env.CORS_ORIGIN ?? "*",
    credentials: true,
    methods: ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
  });
  await fastify.register(rateLimit, {
    max: 100,
    timeWindow: "1 minute",
  });

  await fastify.register(authPlugin);

  await fastify.register(organizationRoutes, { prefix: "/api/organizations" });
  await fastify.register(agentsRoutes, { prefix: "/api/agents" });
  await fastify.register(telephonyRoutes, { prefix: "/api" });

  fastify.get("/health", async () => ({ status: "ok", timestamp: new Date().toISOString() }));

  return fastify;
}

async function start() {
  const fastify = await buildServer();
  try {
    await fastify.listen({ port: PORT, host: HOST });
  } catch (err) {
    fastify.log.error(err);
    process.exit(1);
  }
}

start();

export { buildServer };
