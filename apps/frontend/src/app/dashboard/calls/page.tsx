"use client";

import { useEffect, useState } from "react";
import { useSearchParams } from "next/navigation";
import { telephonyApi, type Call } from "@/lib/api";
import { createBrowserClient } from "@/lib/supabase";
import { Badge, statusBadgeVariant } from "@/components/ui/badge";
import { Phone, PhoneIncoming, PhoneOutgoing } from "lucide-react";
import toast from "react-hot-toast";
import { Suspense } from "react";

function formatDuration(seconds: number | null) {
  if (!seconds) return "—";
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return m > 0 ? `${m}m ${s}s` : `${s}s`;
}

function CallsTable() {
  const searchParams = useSearchParams();
  const agentId = searchParams.get("agent_id") ?? undefined;
  const [calls, setCalls] = useState<Call[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const supabase = createBrowserClient();

  useEffect(() => {
    async function load() {
      const {
        data: { session },
      } = await supabase.auth.getSession();
      if (!session) return;
      try {
        const res = await telephonyApi.listCalls(session.access_token, agentId);
        setCalls(res.calls);
        setTotal(res.total);
      } catch (err) {
        toast.error(err instanceof Error ? err.message : "Failed to load calls");
      } finally {
        setLoading(false);
      }
    }
    load();
  }, [agentId, supabase.auth]);

  if (loading) {
    return (
      <div className="flex h-64 items-center justify-center">
        <div className="h-8 w-8 animate-spin rounded-full border-4 border-brand-600 border-t-transparent" />
      </div>
    );
  }

  if (calls.length === 0) {
    return (
      <div className="card flex flex-col items-center justify-center py-16 text-center">
        <Phone className="mb-4 h-12 w-12 text-gray-300" />
        <h3 className="text-lg font-medium text-gray-900">No calls yet</h3>
        <p className="mt-2 text-sm text-gray-500">
          Calls will appear here once your agents start handling them.
        </p>
      </div>
    );
  }

  return (
    <div className="table-container">
      <div className="flex items-center justify-between border-b border-gray-200 bg-gray-50 px-6 py-3">
        <span className="text-sm font-medium text-gray-700">
          {total} total call{total !== 1 ? "s" : ""}
        </span>
      </div>
      <table className="min-w-full divide-y divide-gray-200">
        <thead className="bg-gray-50">
          <tr>
            <th className="table-header">Direction</th>
            <th className="table-header">From</th>
            <th className="table-header">To</th>
            <th className="table-header">Status</th>
            <th className="table-header">Duration</th>
            <th className="table-header">Date</th>
            <th className="table-header">Recording</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-gray-200 bg-white">
          {calls.map((call) => (
            <tr key={call.id} className="hover:bg-gray-50">
              <td className="table-cell">
                <div className="flex items-center gap-1.5">
                  {call.direction === "inbound" ? (
                    <PhoneIncoming className="h-4 w-4 text-green-500" />
                  ) : (
                    <PhoneOutgoing className="h-4 w-4 text-blue-500" />
                  )}
                  <span className="capitalize text-gray-700">
                    {call.direction}
                  </span>
                </div>
              </td>
              <td className="table-cell font-mono text-gray-600">
                {call.from_number ?? "—"}
              </td>
              <td className="table-cell font-mono text-gray-600">
                {call.to_number ?? "—"}
              </td>
              <td className="table-cell">
                <Badge variant={statusBadgeVariant(call.status)}>
                  {call.status}
                </Badge>
              </td>
              <td className="table-cell">
                {formatDuration(call.duration_seconds)}
              </td>
              <td className="table-cell text-gray-500">
                {new Date(call.created_at).toLocaleString()}
              </td>
              <td className="table-cell">
                {call.recording_url ? (
                  <a
                    href={call.recording_url}
                    target="_blank"
                    rel="noreferrer"
                    className="text-brand-600 hover:underline text-xs"
                  >
                    Listen
                  </a>
                ) : (
                  "—"
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function CallsPage() {
  return (
    <div>
      <div className="page-header">
        <div>
          <h1 className="page-title">Call Logs</h1>
          <p className="mt-1 text-sm text-gray-500">
            Recent telephony interactions
          </p>
        </div>
      </div>
      <Suspense
        fallback={
          <div className="flex h-64 items-center justify-center">
            <div className="h-8 w-8 animate-spin rounded-full border-4 border-brand-600 border-t-transparent" />
          </div>
        }
      >
        <CallsTable />
      </Suspense>
    </div>
  );
}
