import { createServerClient } from "@/lib/supabase";
import Link from "next/link";
import { Bot, Plus, Pencil, Phone } from "lucide-react";
import { Badge, statusBadgeVariant } from "@/components/ui/badge";

type AgentRow = {
  id: string;
  name: string;
  description: string | null;
  status: string;
  bolna_agent_id: string | null;
  created_at: string;
  updated_at: string;
};

type ProfileRow = { organization_id: string | null };

export default async function AgentsPage() {
  const supabase = createServerClient();
  const {
    data: { session },
  } = await supabase.auth.getSession();

  const { data: profileData } = await supabase
    .from("profiles")
    .select("organization_id")
    .eq("id", session!.user.id)
    .single();

  const profile = profileData as ProfileRow | null;

  const { data: agentsData } = await supabase
    .from("agents")
    .select("id, name, description, status, bolna_agent_id, created_at, updated_at")
    .eq("organization_id", profile?.organization_id ?? "")
    .order("created_at", { ascending: false });

  const agents = (agentsData ?? []) as AgentRow[];

  return (
    <div>
      <div className="page-header">
        <div>
          <h1 className="page-title">Agents</h1>
          <p className="mt-1 text-sm text-gray-500">
            Manage your AI voice agents
          </p>
        </div>
        <Link
          href="/dashboard/agents/new"
          className="inline-flex items-center gap-2 rounded-lg bg-brand-600 px-4 py-2 text-sm font-medium text-white hover:bg-brand-700"
        >
          <Plus className="h-4 w-4" />
          New Agent
        </Link>
      </div>

      {agents.length === 0 ? (
        <div className="card flex flex-col items-center justify-center py-16 text-center">
          <Bot className="mb-4 h-12 w-12 text-gray-300" />
          <h3 className="text-lg font-medium text-gray-900">No agents yet</h3>
          <p className="mt-2 text-sm text-gray-500">
            Create your first voice agent to get started.
          </p>
          <Link
            href="/dashboard/agents/new"
            className="mt-6 inline-flex items-center gap-2 rounded-lg bg-brand-600 px-4 py-2 text-sm font-medium text-white hover:bg-brand-700"
          >
            <Plus className="h-4 w-4" />
            Create Agent
          </Link>
        </div>
      ) : (
        <div className="table-container">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="table-header">Agent</th>
                <th className="table-header">Status</th>
                <th className="table-header">Engine ID</th>
                <th className="table-header">Updated</th>
                <th className="table-header">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200 bg-white">
              {agents.map((agent) => (
                <tr key={agent.id} className="hover:bg-gray-50">
                  <td className="table-cell">
                    <div className="flex items-center gap-3">
                      <div className="flex h-9 w-9 flex-shrink-0 items-center justify-center rounded-lg bg-brand-50">
                        <Bot className="h-4 w-4 text-brand-600" />
                      </div>
                      <div>
                        <p className="font-medium text-gray-900">{agent.name}</p>
                        {agent.description && (
                          <p className="text-xs text-gray-500 truncate max-w-xs">
                            {agent.description}
                          </p>
                        )}
                      </div>
                    </div>
                  </td>
                  <td className="table-cell">
                    <Badge variant={statusBadgeVariant(agent.status)}>
                      {agent.status}
                    </Badge>
                  </td>
                  <td className="table-cell">
                    <code className="rounded bg-gray-100 px-1.5 py-0.5 text-xs text-gray-600">
                      {agent.bolna_agent_id
                        ? agent.bolna_agent_id.slice(0, 8) + "…"
                        : "—"}
                    </code>
                  </td>
                  <td className="table-cell text-gray-500">
                    {new Date(agent.updated_at).toLocaleDateString()}
                  </td>
                  <td className="table-cell">
                    <div className="flex items-center gap-2">
                      <Link
                        href={`/dashboard/agents/${agent.id}`}
                        className="inline-flex items-center gap-1 rounded-md px-2 py-1 text-xs font-medium text-gray-600 hover:bg-gray-100"
                      >
                        <Pencil className="h-3 w-3" />
                        Edit
                      </Link>
                      <Link
                        href={`/dashboard/calls?agent_id=${agent.id}`}
                        className="inline-flex items-center gap-1 rounded-md px-2 py-1 text-xs font-medium text-gray-600 hover:bg-gray-100"
                      >
                        <Phone className="h-3 w-3" />
                        Calls
                      </Link>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
