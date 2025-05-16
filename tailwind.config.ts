import type { Config } from "tailwindcss";

export default {
  darkMode: ["class"],
  content: ["./client/index.html", "./client/src/**/*.{js,jsx,ts,tsx}"],
  theme: {
    fontFamily: {
      sans: ['Roboto', 'sans-serif'],
      poppins: ['Poppins', 'sans-serif'],
    },
    extend: {
      borderRadius: {
        lg: "var(--radius)",
        md: "calc(var(--radius) - 2px)",
        sm: "calc(var(--radius) - 4px)",
      },
      colors: {
        background: "hsl(var(--background))",
        foreground: "hsl(var(--foreground))",
        card: {
          DEFAULT: "hsl(var(--card))",
          foreground: "hsl(var(--card-foreground))",
        },
        popover: {
          DEFAULT: "hsl(var(--popover))",
          foreground: "hsl(var(--popover-foreground))",
        },
        primary: {
          DEFAULT: "hsl(var(--primary))",
          foreground: "hsl(var(--primary-foreground))",
        },
        secondary: {
          DEFAULT: "hsl(var(--secondary))",
          foreground: "hsl(var(--secondary-foreground))",
        },
        muted: {
          DEFAULT: "hsl(var(--muted))",
          foreground: "hsl(var(--muted-foreground))",
        },
        accent: {
          DEFAULT: "hsl(var(--accent))",
          foreground: "hsl(var(--accent-foreground))",
        },
        destructive: {
          DEFAULT: "hsl(var(--destructive))",
          foreground: "hsl(var(--destructive-foreground))",
        },
        border: "hsl(var(--border))",
        input: "hsl(var(--input))",
        ring: "hsl(var(--ring))",
        chart: {
          "1": "hsl(var(--chart-1))",
          "2": "hsl(var(--chart-2))",
          "3": "hsl(var(--chart-3))",
          "4": "hsl(var(--chart-4))",
          "5": "hsl(var(--chart-5))",
        },
        sidebar: {
          DEFAULT: "hsl(var(--sidebar-background))",
          foreground: "hsl(var(--sidebar-foreground))",
          primary: "hsl(var(--sidebar-primary))",
          "primary-foreground": "hsl(var(--sidebar-primary-foreground))",
          accent: "hsl(var(--sidebar-accent))",
          "accent-foreground": "hsl(var(--sidebar-accent-foreground))",
          border: "hsl(var(--sidebar-border))",
          ring: "hsl(var(--sidebar-ring))",
        },
      },
      keyframes: {
        "accordion-down": {
          from: {
            height: "0",
          },
          to: {
            height: "var(--radix-accordion-content-height)",
          },
        },
        "accordion-up": {
          from: {
            height: "var(--radix-accordion-content-height)",
          },
          to: {
            height: "0",
          },
        },
        "scan": {
          "0%": { left: "-30%" },
          "100%": { left: "100%" }
        },
        "pulse-glow": {
          "0%, 100%": { 
            opacity: "0.6",
            boxShadow: "0 0 15px rgba(0, 200, 255, 0.3)"
          },
          "50%": { 
            opacity: "1.0",
            boxShadow: "0 0 25px rgba(0, 200, 255, 0.6)"
          }
        },
        "shine": {
          "100%": { left: "125%" }
        },
        "shine-slow": {
          "100%": { left: "150%" }
        },
        "twinkle": {
          "0%": { 
            opacity: "0.1",
            transform: "scale(1)" 
          },
          "50%": { 
            opacity: "0.7",
            transform: "scale(1.3)" 
          },
          "100%": { 
            opacity: "0.1",
            transform: "scale(1)" 
          }
        },
        "float": {
          "0%, 100%": { transform: "translateY(0)" },
          "50%": { transform: "translateY(-10px)" }
        },
        "rotate-3d": {
          "0%": { transform: "rotate3d(1, 1, 1, 0deg)" },
          "100%": { transform: "rotate3d(1, 1, 1, 360deg)" }
        },
        "pop": {
          "0%": { transform: "scale(0.95)", opacity: "0.5" },
          "50%": { transform: "scale(1.05)", opacity: "1" },
          "100%": { transform: "scale(1)", opacity: "1" }
        }
      },
      animation: {
        "accordion-down": "accordion-down 0.2s ease-out",
        "accordion-up": "accordion-up 0.2s ease-out",
        "scan": "scan 2s infinite",
        "pulse-glow": "pulse-glow 2s infinite",
        "shine": "shine 1.5s forwards",
        "shine-slow": "shine-slow 2.5s forwards",
        "twinkle": "twinkle 4s ease-in-out infinite",
        "float": "float 3s ease-in-out infinite",
        "rotate-3d": "rotate-3d 10s linear infinite",
        "pop": "pop 0.3s ease-out"
      },
    },
  },
  plugins: [require("tailwindcss-animate"), require("@tailwindcss/typography")],
} satisfies Config;
