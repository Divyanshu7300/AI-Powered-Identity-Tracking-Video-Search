/** @type {import('next').NextConfig} */
const nextConfig = {
  distDir: ".next-build",
  reactCompiler: false,
  experimental: {
    proxyClientMaxBodySize: "512mb",
    serverActions: {
      bodySizeLimit: "512mb",
    },
  },
};

export default nextConfig;
