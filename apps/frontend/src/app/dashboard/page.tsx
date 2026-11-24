import { createServerClient } from "@/lib/supabase";
import { Bot, Phone, TrendingUp, Activity } from "lucide-react";
import Link from "next/link";

type AgentRow = { id: string; status: string };
type CallRow = { id: string; status: string; duration_seconds: number | null };
type ProfileRow = { organization_id: string | null };

export default async function DashboardPage() {
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
  const orgId = profile?.organization_id;

  const [agentsRes, callsRes] = await Promise.all([
    orgId
      ? supabase
          .from("agents")
          .select("id, status", { count: "exact" })
          .eq("organization_id", orgId)
      : Promise.resolve({ data: [] as AgentRow[], count: 0, error: null }),
    orgId
      ? supabase
          .from("calls")
          .select("id, status, duration_seconds", { count: "exact" })
          .eq("organization_id", orgId)
          .gte(
            "created_at",
            new Date(Date.now() - 30 * 24 * 60 * 60 * 1000).toISOString()
          )
      : Promise.resolve({ data: [] as CallRow[], count: 0, error: null }),
  ]);

  const agentsData = (agentsRes.data ?? []) as AgentRow[];
  const callsData = (callsRes.data ?? []) as CallRow[];

  const totalAgents = agentsRes.count ?? 0;
  const activeAgents = agentsData.filter((a) => a.status === "active").length;
  const totalCalls = callsRes.count ?? 0;
  const completedCalls = callsData.filter((c) => c.status === "completed");
  const avgDuration =
    completedCalls.length > 0
      ? Math.round(
          completedCalls.reduce((s, c) => s + (c.duration_seconds ?? 0), 0) /
            completedCalls.length
        )
      : 0;

  const stats = [
    {
      label: "Total Agents",
      value: totalAgents,
      sub: `${activeAgents} active`,
      icon: Bot,
      color: "bg-blue-50 text-blue-600",
      href: "/dashboard/agents",
    },
    {
      label: "Calls (30d)",
      value: totalCalls,
      sub: `${completedCalls.length} completed`,
      icon: Phone,
      color: "bg-green-50 text-green-600",
      href: "/dashboard/calls",
    },
    {
      label: "Avg Duration",
      value: avgDuration > 0 ? `${avgDuration}s` : "—",
      sub: "per completed call",
      icon: TrendingUp,
      color: "bg-purple-50 text-purple-600",
      href: "/dashboard/calls",
    },
    {
      label: "Success Rate",
      value:
        totalCalls > 0
          ? `${Math.round((completedCalls.length / totalCalls) * 100)}%`
          : "—",
      sub: "last 30 days",
      icon: Activity,
      color: "bg-orange-50 text-orange-600",
      href: "/dashboard/calls",
    },
  ];

  return (
    <div>
      <div className="page-header">
        <h1 className="page-title">Overview</h1>
        <Link
          href="/dashboard/agents/new"
          className="inline-flex items-center gap-2 rounded-lg bg-brand-600 px-4 py-2 text-sm font-medium text-white hover:bg-brand-700"
        >
          <Bot className="h-4 w-4" />
          New Agent
        </Link>
      </div>

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {stats.map(({ label, value, sub, icon: Icon, color, href }) => (
          <Link
            key={label}
            href={href}
            className="card p-6 transition-shadow hover:shadow-md"
          >
            <div className="flex items-start justify-between">
              <div>
                <p className="text-sm font-medium text-gray-500">{label}</p>
                <p className="mt-1 text-3xl font-bold text-gray-900">{value}</p>
                <p className="mt-1 text-xs text-gray-500">{sub}</p>
              </div>
              <div className={`rounded-lg p-2 ${color}`}>
                <Icon className="h-5 w-5" />
              </div>
            </div>
          </Link>
        ))}
      </div>

      {totalAgents === 0 && (
        <div className="mt-12 flex flex-col items-center justify-center text-center">
          <div className="mb-4 flex h-16 w-16 items-center justify-center rounded-full bg-brand-50">
            <Bot className="h-8 w-8 text-brand-600" />
          </div>
          <h2 className="text-lg font-semibold text-gray-900">
            Create your first voice agent
          </h2>
          <p className="mt-2 max-w-sm text-sm text-gray-500">
            Configure an AI voice agent with your LLM of choice, connect it to
            a phone number, and start handling calls automatically.
          </p>
          <Link
            href="/dashboard/agents/new"
            className="mt-6 inline-flex items-center gap-2 rounded-lg bg-brand-600 px-5 py-2.5 text-sm font-medium text-white hover:bg-brand-700"
          >
            <Bot className="h-4 w-4" />
            Build your first agent
          </Link>
        </div>
      )}
    </div>
  );
}
