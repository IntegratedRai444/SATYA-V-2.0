import type { Config } from "tailwindcss";

export default {
  darkMode: ["class"],
  content: ["./client/index.html", "./client/src/**/*.{js,jsx,ts,tsx}"],
  theme: {
    fontFamily: {
      sans: ['Roboto', 'sans-serif'],
      poppins: ['Poppins', 'sans-serif'],
      inter: ['Inter', 'sans-serif'],
    },
    extend: {
      borderRadius: {
        lg: "var(--radius)",
        md: "calc(var(--radius) - 2px)",
        sm: "calc(var(--radius) - 4px)",
      },
      colors: {
        // SatyaAI Brand Colors - Exact from design
        'satyaai': {
          'primary': '#00a8ff',
          'primary-dark': '#0088cc',
          'primary-light': '#33b8ff',
          'blue': '#00a8ff',
          'cyan': '#00d4ff',
        },
        // Background Colors - Dark theme from design
        'bg': {
          'primary': '#0a0a0a',
          'secondary': '#1a1a1a', 
          'tertiary': '#2a2a2a',
          'card': '#1e1e1e',
          'sidebar': '#141414',
        },
        // Text Colors
        'text': {
          'primary': '#ffffff',
          'secondary': '#b3b3b3',
          'muted': '#666666',
          'accent': '#00a8ff',
        },
        // Accent Colors
        'accent': {
          'cyan': '#00a8ff',
          'cyan-dark': '#0088cc',
          'cyan-light': '#33b8ff',
          'green': '#00ff88',
          'orange': '#ff8800', 
          'red': '#ff4444',
          'purple': '#8b5cf6',
        },
        // Border Colors
        'border': {
          'primary': '#333333',
          'secondary': '#444444',
          'accent': '#00a8ff',
        },
        // Legacy support for existing components
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
          DEFAULT: "#00a8ff",
          foreground: "#ffffff",
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
          DEFAULT: "#141414",
          foreground: "#ffffff",
          primary: "#00a8ff",
          "primary-foreground": "#ffffff",
          accent: "#00a8ff",
          "accent-foreground": "#ffffff",
          border: "#333333",
          ring: "#00a8ff",
        },
      },
      fontSize: {
        'hero': ['3.5rem', { lineHeight: '1.1', fontWeight: '700' }],
        'heading-1': ['2.5rem', { lineHeight: '1.2', fontWeight: '600' }],
        'heading-2': ['1.875rem', { lineHeight: '1.3', fontWeight: '600' }],
        'body-large': ['1.125rem', { lineHeight: '1.6', fontWeight: '400' }],
        'body': ['1rem', { lineHeight: '1.5', fontWeight: '400' }],
      },
      backgroundImage: {
        'gradient-primary': 'linear-gradient(135deg, #00a8ff 0%, #0088cc 100%)',
        'gradient-card': 'linear-gradient(135deg, #1e1e1e 0%, #2a2a2a 100%)',
        'gradient-hero': 'linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 50%, #2a2a2a 100%)',
        'gradient-cyan-purple': 'linear-gradient(135deg, #00a8ff 0%, #8b5cf6 100%)',
        'gradient-green-cyan': 'linear-gradient(135deg, #00ff88 0%, #00a8ff 100%)',
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
        },
        "fadeIn": {
          "0%": { opacity: "0", transform: "translateY(10px)" },
          "100%": { opacity: "1", transform: "translateY(0)" }
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
        "pop": "pop 0.3s ease-out",
        "fadeIn": "fadeIn 0.5s ease-out"
      },
    },
  },
  plugins: [require("tailwindcss-animate"), require("@tailwindcss/typography")],
} satisfies Config;
