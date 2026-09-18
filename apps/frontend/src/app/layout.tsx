import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Bolna Voice AI",
  description: "Build and deploy production-ready AI voice agents",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
