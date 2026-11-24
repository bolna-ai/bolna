"use client";

import { useEffect, useState } from "react";
import { organizationsApi, type Organization } from "@/lib/api";
import { createBrowserClient } from "@/lib/supabase";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import toast from "react-hot-toast";

export default function SettingsPage() {
  const [org, setOrg] = useState<Organization | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const supabase = createBrowserClient();

  const [form, setForm] = useState({
    name: "",
    bolna_engine_url: "",
    twilio_account_sid: "",
    twilio_auth_token: "",
    twilio_phone_number: "",
    plivo_auth_id: "",
    plivo_auth_token: "",
    plivo_phone_number: "",
  });

  useEffect(() => {
    async function load() {
      const {
        data: { session },
      } = await supabase.auth.getSession();
      if (!session) return;
      try {
        const res = await organizationsApi.getMe(session.access_token);
        setOrg(res.organization);
        setForm((f) => ({
          ...f,
          name: res.organization.name,
          bolna_engine_url: res.organization.bolna_engine_url,
        }));
      } catch (err) {
        toast.error(err instanceof Error ? err.message : "Failed to load settings");
      } finally {
        setLoading(false);
      }
    }
    load();
  }, [supabase.auth]);

  const update = (key: keyof typeof form, value: string) => {
    setForm((prev) => ({ ...prev, [key]: value }));
  };

  const handleSave = async () => {
    setSaving(true);
    try {
      const {
        data: { session },
      } = await supabase.auth.getSession();
      if (!session) return;

      const payload: Partial<Organization & Record<string, string>> = {};
      if (form.name) payload.name = form.name;
      if (form.bolna_engine_url) payload.bolna_engine_url = form.bolna_engine_url;
      if (form.twilio_account_sid) payload.twilio_account_sid = form.twilio_account_sid;
      if (form.twilio_auth_token) payload.twilio_auth_token = form.twilio_auth_token;
      if (form.twilio_phone_number) payload.twilio_phone_number = form.twilio_phone_number;
      if (form.plivo_auth_id) payload.plivo_auth_id = form.plivo_auth_id;
      if (form.plivo_auth_token) payload.plivo_auth_token = form.plivo_auth_token;
      if (form.plivo_phone_number) payload.plivo_phone_number = form.plivo_phone_number;

      const res = await organizationsApi.update(session.access_token, payload);
      setOrg(res.organization);
      toast.success("Settings saved!");
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Failed to save settings");
    } finally {
      setSaving(false);
    }
  };

  if (loading) {
    return (
      <div className="flex h-64 items-center justify-center">
        <div className="h-8 w-8 animate-spin rounded-full border-4 border-brand-600 border-t-transparent" />
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-2xl">
      <div className="page-header">
        <div>
          <h1 className="page-title">Settings</h1>
          <p className="mt-1 text-sm text-gray-500">
            Configure your organization and integrations
          </p>
        </div>
      </div>

      <div className="space-y-6">
        {/* Organization */}
        <div className="card p-6">
          <h2 className="mb-4 text-base font-semibold text-gray-900">
            Organization
          </h2>
          <div className="space-y-4">
            <Input
              label="Organization Name"
              value={form.name}
              onChange={(e) => update("name", e.target.value)}
            />
            <Input
              label="Bolna Engine URL"
              value={form.bolna_engine_url}
              onChange={(e) => update("bolna_engine_url", e.target.value)}
              hint="URL of your deployed Bolna Python engine (e.g. https://bolna.yourdomain.com)"
              placeholder="https://bolna.yourdomain.com"
            />
            {org && (
              <div className="rounded-lg bg-gray-50 p-3 text-sm text-gray-600">
                <span className="font-medium">Plan:</span> {org.plan} &nbsp;|&nbsp;
                <span className="font-medium">Slug:</span> {org.slug}
              </div>
            )}
          </div>
        </div>

        {/* Twilio */}
        <div className="card p-6">
          <h2 className="mb-1 text-base font-semibold text-gray-900">
            Twilio Credentials
          </h2>
          <p className="mb-4 text-sm text-gray-500">
            Required for outbound calls. Values are stored encrypted.
          </p>
          <div className="space-y-4">
            <Input
              label="Account SID"
              value={form.twilio_account_sid}
              onChange={(e) => update("twilio_account_sid", e.target.value)}
              placeholder="ACxxxxxxxxxxxxxxxx"
            />
            <Input
              label="Auth Token"
              type="password"
              value={form.twilio_auth_token}
              onChange={(e) => update("twilio_auth_token", e.target.value)}
              placeholder="••••••••••••••••"
            />
            <Input
              label="Phone Number"
              value={form.twilio_phone_number}
              onChange={(e) => update("twilio_phone_number", e.target.value)}
              placeholder="+15551234567"
              hint="E.164 format, e.g. +15551234567"
            />
          </div>
        </div>

        {/* Plivo */}
        <div className="card p-6">
          <h2 className="mb-1 text-base font-semibold text-gray-900">
            Plivo Credentials
          </h2>
          <p className="mb-4 text-sm text-gray-500">
            Optional: configure Plivo as an alternative telephony provider.
          </p>
          <div className="space-y-4">
            <Input
              label="Auth ID"
              value={form.plivo_auth_id}
              onChange={(e) => update("plivo_auth_id", e.target.value)}
            />
            <Input
              label="Auth Token"
              type="password"
              value={form.plivo_auth_token}
              onChange={(e) => update("plivo_auth_token", e.target.value)}
            />
            <Input
              label="Phone Number"
              value={form.plivo_phone_number}
              onChange={(e) => update("plivo_phone_number", e.target.value)}
              placeholder="+15551234567"
            />
          </div>
        </div>

        <div className="flex justify-end">
          <Button onClick={handleSave} loading={saving}>
            Save Settings
          </Button>
        </div>
      </div>
    </div>
  );
}
