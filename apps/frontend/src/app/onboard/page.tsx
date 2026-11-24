"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { createBrowserClient } from "@/lib/supabase";
import { organizationsApi } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import toast from "react-hot-toast";

export default function OnboardPage() {
  const router = useRouter();
  const [name, setName] = useState("");
  const [slug, setSlug] = useState("");
  const [loading, setLoading] = useState(false);
  const [errors, setErrors] = useState<{ name?: string; slug?: string }>({});
  const supabase = createBrowserClient();

  const handleNameChange = (value: string) => {
    setName(value);
    const autoSlug = value
      .toLowerCase()
      .replace(/[^a-z0-9\s-]/g, "")
      .replace(/\s+/g, "-")
      .replace(/-+/g, "-")
      .slice(0, 50);
    setSlug(autoSlug);
    setErrors({});
  };

  const validate = (): boolean => {
    const errs: typeof errors = {};
    if (!name.trim()) errs.name = "Organization name is required";
    if (!slug.match(/^[a-z0-9-]{2,50}$/))
      errs.slug = "Slug must be 2–50 lowercase letters, numbers, and hyphens";
    setErrors(errs);
    return Object.keys(errs).length === 0;
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!validate()) return;
    setLoading(true);
    try {
      const {
        data: { session },
      } = await supabase.auth.getSession();
      if (!session) throw new Error("Not authenticated");

      await organizationsApi.onboard(session.access_token, {
        organization_name: name,
        slug,
      });

      toast.success("Welcome! Your workspace is ready.");
      router.push("/dashboard");
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Onboarding failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex min-h-screen items-center justify-center bg-gradient-to-br from-brand-50 to-gray-100 p-4">
      <div className="w-full max-w-md">
        <div className="mb-8 text-center">
          <div className="mb-4 flex justify-center">
            <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-brand-600 text-white text-xl font-bold">
              B
            </div>
          </div>
          <h1 className="text-2xl font-bold text-gray-900">Set up your workspace</h1>
          <p className="mt-1 text-sm text-gray-500">
            Create your organization to get started
          </p>
        </div>

        <div className="card p-8">
          <form onSubmit={handleSubmit} className="flex flex-col gap-4">
            <Input
              label="Organization Name"
              placeholder="Acme Corp"
              value={name}
              onChange={(e) => handleNameChange(e.target.value)}
              error={errors.name}
              required
            />
            <Input
              label="Workspace Slug"
              placeholder="acme-corp"
              value={slug}
              onChange={(e) => {
                setSlug(e.target.value);
                setErrors({});
              }}
              error={errors.slug}
              hint="Used in URLs and identifiers. Lowercase letters, numbers, hyphens."
            />
            <Button type="submit" loading={loading} className="mt-2 w-full">
              Create Workspace
            </Button>
          </form>
        </div>
      </div>
    </div>
  );
}
