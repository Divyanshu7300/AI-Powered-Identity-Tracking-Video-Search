import "./globals.css";
import Providers from "./providers";
import Navbar from "./components/Navbar";

export const metadata = {
  title: "AURA — AI-Powered Identity Tracking & Video Search",
  description: "AI-powered identity tracking, person re-identification, and video search.",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-[#f8fafc] text-slate-900 font-sans antialiased selection:bg-slate-900 selection:text-white flex flex-col overflow-x-hidden">
        <Providers>
          <Navbar />
          <main className="flex-1 w-full">
            {children}
          </main>
        </Providers>
      </body>
    </html>
  );
}
