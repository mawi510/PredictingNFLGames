import type { Metadata } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import Link from "next/link";
import StatusBar from "@/components/StatusBar";
import { DataProvider } from "@/lib/data";
import "./globals.css";

const inter = Inter({
  variable: "--font-inter",
  subsets: ["latin"],
  weight: ["400", "500", "600"],
});

const mono = JetBrains_Mono({
  variable: "--font-jetbrains-mono",
  subsets: ["latin"],
  weight: ["400", "500"],
});

export const metadata: Metadata = {
  title: "ProMatchPredict: NFL spread model",
  description:
    "Every NFL spread, one model's number, and the record kept in public.",
};

const NAV = [
  { href: "/", label: "Slate" },
  { href: "/record/", label: "Record" },
  { href: "/teams/", label: "Teams" },
];

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={`${inter.variable} ${mono.variable} h-full antialiased`}>
      <body className="flex min-h-full flex-col bg-bg text-fg">
        <DataProvider>
          <header className="border-b border-line">
            <div className="mx-auto flex h-12 max-w-[1200px] items-center justify-between px-4 md:px-6">
              <Link href="/" className="text-[15px] font-semibold">
                ProMatchPredict
              </Link>
              <nav className="flex items-center gap-5 text-[13px] text-fg-2">
                {NAV.map((item) => (
                  <Link key={item.href} href={item.href} className="hover:text-fg">
                    {item.label}
                  </Link>
                ))}
              </nav>
            </div>
          </header>
          <StatusBar />
          <main className="mx-auto w-full max-w-[1200px] flex-1 px-4 pt-8 pb-12 md:px-6 md:pt-12">
            {children}
          </main>
          <footer className="border-t border-line">
            <div className="mx-auto max-w-[1200px] px-4 py-4 text-[13px] leading-6 text-fg-2 md:px-6">
              <p>Model output for entertainment, not betting advice.</p>
              <p>
                Validated close to breakeven against the closing line. The full
                record is on the{" "}
                <Link href="/record/" className="text-accent">
                  Track record
                </Link>{" "}
                page.
              </p>
            </div>
          </footer>
        </DataProvider>
      </body>
    </html>
  );
}
