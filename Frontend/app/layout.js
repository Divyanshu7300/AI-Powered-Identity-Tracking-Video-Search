import "./globals.css";

export const metadata = {
  title: "MOT Re-ID Dashboard",
  description: "Video search dashboard for person tracking, Re-ID, and evidence review.",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body className="min-h-full flex flex-col">{children}</body>
    </html>
  );
}
