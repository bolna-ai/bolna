import { clsx } from "clsx";

type BadgeVariant = "green" | "yellow" | "red" | "blue" | "gray";

const variantClasses: Record<BadgeVariant, string> = {
  green: "bg-green-100 text-green-800",
  yellow: "bg-yellow-100 text-yellow-800",
  red: "bg-red-100 text-red-800",
  blue: "bg-blue-100 text-blue-800",
  gray: "bg-gray-100 text-gray-700",
};

interface BadgeProps {
  children: React.ReactNode;
  variant?: BadgeVariant;
  className?: string;
}

export function Badge({ children, variant = "gray", className }: BadgeProps) {
  return (
    <span
      className={clsx(
        "inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium",
        variantClasses[variant],
        className
      )}
    >
      {children}
    </span>
  );
}

export function statusBadgeVariant(status: string): BadgeVariant {
  switch (status) {
    case "active":
    case "completed":
      return "green";
    case "draft":
    case "initiated":
    case "ringing":
      return "yellow";
    case "inactive":
    case "failed":
    case "busy":
    case "no-answer":
      return "red";
    case "in-progress":
      return "blue";
    default:
      return "gray";
  }
}
