import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        surface: {
          900: "#0a0d14",
          800: "#0f1320",
          700: "#161b2e",
          600: "#1e2540",
        },
        accent: {
          DEFAULT: "#6366f1",
          hover: "#818cf8",
        },
        danger: "#f43f5e",
      },
      fontFamily: {
        sans: ["var(--font-inter)", "system-ui", "sans-serif"],
        mono: ["var(--font-mono)", "monospace"],
      },
    },
  },
  plugins: [],
};

export default config;
