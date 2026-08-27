import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "AI Vehicle Manual Assistant",
  description: "Ask your vehicle manual anything. Get answers with page citations.",
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
