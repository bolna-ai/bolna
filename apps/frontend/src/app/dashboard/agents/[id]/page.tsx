"use client";

import { useEffect, useState } from "react";
import { useRouter, useParams } from "next/navigation";
import { AgentBuilder } from "@/components/agent-builder";
import { agentsApi, telephonyApi, type Agent } from "@/lib/api";
import { createBrowserClient } from "@/lib/supabase";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge, statusBadgeVariant } from "@/components/ui/badge";
import { Phone, Trash2 } from "lucide-react";
import toast from "react-hot-toast";

export default function AgentDetailPage() {
  const router = useRouter();
  const { id } = useParams<{ id: string }>();
  const [agent, setAgent] = useState<Agent | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [callLoading, setCallLoading] = useState(false);
  const [toNumber, setToNumber] = useState("");
  const [showCall, setShowCall] = useState(false);
  const supabase = createBrowserClient();

  useEffect(() => {
    async function load() {
      const {
        data: { session },
      } = await supabase.auth.getSession();
      if (!session) return;
      try {
        const res = await agentsApi.get(session.access_token, id);
        setAgent(res.agent);
      } catch {
        toast.error("Agent not found");
        router.push("/dashboard/agents");
      } finally {
        setLoading(false);
      }
    }
    load();
  }, [id, router, supabase.auth]);

  const handleUpdate = async (
    bolnaConfig: unknown,
    prompts: unknown,
    name: string,
    description: string
  ) => {
    setSaving(true);
    try {
      const {
        data: { session },
      } = await supabase.auth.getSession();
      if (!session) throw new Error("Not authenticated");

      const res = await agentsApi.update(session.access_token, id, {
        name,
        description,
        bolna_config: bolnaConfig as Record<string, unknown>,
        prompts: prompts as Record<string, Record<string, string>>,
      });
      setAgent(res.agent);
      toast.success("Agent updated successfully!");
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Update failed");
    } finally {
      setSaving(false);
    }
  };

  const handleDelete = async () => {
    if (!window.confirm("Delete this agent? This cannot be undone.")) return;
    try {
      const {
        data: { session },
      } = await supabase.auth.getSession();
      if (!session) return;
      await agentsApi.delete(session.access_token, id);
      toast.success("Agent deleted");
      router.push("/dashboard/agents");
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Delete failed");
    }
  };

  const handleCall = async () => {
    if (!toNumber.trim()) {
      toast.error("Enter a phone number");
      return;
    }
    setCallLoading(true);
    try {
      const {
        data: { session },
      } = await supabase.auth.getSession();
      if (!session) return;
      const res = await telephonyApi.initiateCall(
        session.access_token,
        id,
        toNumber
      );
      toast.success(`Call initiated! SID: ${res.call_sid}`);
      setShowCall(false);
      setToNumber("");
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Call failed");
    } finally {
      setCallLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="flex h-64 items-center justify-center">
        <div className="h-8 w-8 animate-spin rounded-full border-4 border-brand-600 border-t-transparent" />
      </div>
    );
  }

  if (!agent) return null;

  return (
    <div className="mx-auto max-w-2xl">
      <div className="page-header">
        <div>
          <div className="flex items-center gap-3">
            <h1 className="page-title">{agent.name}</h1>
            <Badge variant={statusBadgeVariant(agent.status)}>
              {agent.status}
            </Badge>
          </div>
          {agent.description && (
            <p className="mt-1 text-sm text-gray-500">{agent.description}</p>
          )}
        </div>
        <div className="flex items-center gap-2">
          <Button
            variant="secondary"
            size="sm"
            onClick={() => setShowCall(!showCall)}
          >
            <Phone className="h-4 w-4" />
            Test Call
          </Button>
          <Button variant="danger" size="sm" onClick={handleDelete}>
            <Trash2 className="h-4 w-4" />
          </Button>
        </div>
      </div>

      {showCall && (
        <div className="card mb-6 p-4">
          <h3 className="mb-3 text-sm font-medium text-gray-900">
            Initiate Test Call
          </h3>
          <div className="flex gap-2">
            <Input
              placeholder="+1 (555) 000-0000"
              value={toNumber}
              onChange={(e) => setToNumber(e.target.value)}
              className="flex-1"
            />
            <Button onClick={handleCall} loading={callLoading}>
              Call
            </Button>
            <Button variant="secondary" onClick={() => setShowCall(false)}>
              Cancel
            </Button>
          </div>
          <p className="mt-2 text-xs text-gray-500">
            Requires Twilio credentials to be configured in Settings.
          </p>
        </div>
      )}

      <div className="card p-6">
        <AgentBuilder
          onSubmit={handleUpdate}
          onCancel={() => router.push("/dashboard/agents")}
          submitLabel="Save Changes"
          isLoading={saving}
        />
      </div>
    </div>
  );
}
