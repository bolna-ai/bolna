"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { AgentBuilder } from "@/components/agent-builder";
import { agentsApi } from "@/lib/api";
import { createBrowserClient } from "@/lib/supabase";
import toast from "react-hot-toast";

export default function NewAgentPage() {
  const router = useRouter();
  const [loading, setLoading] = useState(false);
  const supabase = createBrowserClient();

  const handleSubmit = async (
    bolnaConfig: unknown,
    prompts: unknown,
    name: string,
    description: string
  ) => {
    setLoading(true);
    try {
      const {
        data: { session },
      } = await supabase.auth.getSession();
      if (!session) throw new Error("Not authenticated");

      await agentsApi.create(session.access_token, {
        name,
        description,
        bolna_config: bolnaConfig as Record<string, unknown>,
        prompts: prompts as Record<string, Record<string, string>>,
      });

      toast.success("Agent created successfully!");
      router.push("/dashboard/agents");
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Failed to create agent");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="mx-auto max-w-2xl">
      <div className="page-header">
        <div>
          <h1 className="page-title">Create Agent</h1>
          <p className="mt-1 text-sm text-gray-500">
            Configure a new AI voice agent
          </p>
        </div>
      </div>

      <div className="card p-6">
        <AgentBuilder
          onSubmit={handleSubmit}
          onCancel={() => router.push("/dashboard/agents")}
          submitLabel="Create Agent"
          isLoading={loading}
        />
      </div>
    </div>
  );
}
