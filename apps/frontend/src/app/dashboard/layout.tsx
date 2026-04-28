import { redirect } from "next/navigation";
import { createServerClient } from "@/lib/supabase";
import DashboardShell from "@/components/dashboard-shell";

type ProfilePartial = {
  organization_id: string | null;
  full_name: string | null;
  role: string;
};

export default async function DashboardLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const supabase = createServerClient();
  const {
    data: { session },
  } = await supabase.auth.getSession();

  if (!session) {
    redirect("/auth");
  }

  const { data: profileData } = await supabase
    .from("profiles")
    .select("organization_id, full_name, role")
    .eq("id", session.user.id)
    .single();

  const profile = profileData as ProfilePartial | null;

  if (!profile?.organization_id) {
    redirect("/onboard");
  }

  return (
    <DashboardShell
      user={{
        email: session.user.email ?? "",
        fullName: (profile as ProfilePartial).full_name ?? "",
      }}
    >
      {children}
    </DashboardShell>
  );
}
