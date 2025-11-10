/** @type {import('tailwindcss').Config} */
export default {
  darkMode: ["class"],
  content: [
    './pages/**/*.{ts,tsx}',
    './components/**/*.{ts,tsx}',
    './app/**/*.{ts,tsx}',
    './src/**/*.{ts,tsx}',
  ],
  prefix: "",
  theme: {
    container: {
      center: true,
      padding: "1rem",
      screens: {
        "2xl": "1400px",
      },
    },
    extend: {
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
        'accent-colors': {
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
        // Base colors
        border: "hsl(var(--border))",
        input: "hsl(var(--input))",
        ring: "hsl(var(--ring))",
        background: "hsl(var(--background))",
        foreground: "hsl(var(--foreground))",
        
        // Primary colors (purple theme)
        primary: {
          DEFAULT: "#7C3AED",
          foreground: "#FFFFFF",
          dark: "#6D28D9",
          light: "#8B5CF6",
        },
        
        // Secondary colors
        secondary: {
          DEFAULT: "#1F2937",
          foreground: "#F9FAFB",
        },
        
        // Accent colors
        accent: {
          DEFAULT: "#8B5CF6",
          foreground: "#FFFFFF",
        },
        
        // Background colors
        background: {
          DEFAULT: "#111827",
          secondary: "#1F2937",
          card: "#1F2937",
        },
        
        // Text colors
        foreground: {
          DEFAULT: "#F9FAFB",
          muted: "#9CA3AF",
        },
        
        // Status colors
        success: {
          DEFAULT: "#10B981",
          foreground: "#ECFDF5",
        },
        warning: {
          DEFAULT: "#F59E0B",
          foreground: "#FFFBEB",
        },
        error: {
          DEFAULT: "#EF4444",
          foreground: "#FEF2F2",
        },
        
        // Sidebar specific colors
        sidebar: {
          DEFAULT: "#111827",
          foreground: "#F9FAFB",
          primary: "#7C3AED",
          "primary-foreground": "#FFFFFF",
          accent: "#8B5CF6",
          "accent-foreground": "#FFFFFF",
          border: "#1F2937",
          ring: "#7C3AED",
          hover: "#1F2937",
          active: "#1F2937",
        },
        
        // Input fields
        input: {
          DEFAULT: "#1F2937",
          foreground: "#F9FAFB",
          placeholder: "#6B7280",
          border: "#374151",
        },
        
        // Chart colors
        chart: {
          "1": "#7C3AED",
          "2": "#8B5CF6",
          "3": "#A78BFA",
          "4": "#C4B5FD",
          "5": "#DDD6FE",
        },
        
        // Gradients
        gradient: {
          primary: "linear-gradient(135deg, #7C3AED 0%, #8B5CF6 100%)",
          secondary: "linear-gradient(135deg, #1F2937 0%, #111827 100%)",
        },
      },
      borderRadius: {
        lg: "var(--radius)",
        md: "calc(var(--radius) - 2px)",
        sm: "calc(var(--radius) - 4px)",
      },
      fontFamily: {
        sans: ["Roboto", "system-ui", "sans-serif"],
        poppins: ["Poppins", "system-ui", "sans-serif"],
        orbitron: ["Orbitron", "monospace"],
      },
      keyframes: {
        "accordion-down": {
          from: { height: "0" },
          to: { height: "var(--radix-accordion-content-height)" },
        },
        "accordion-up": {
          from: { height: "var(--radix-accordion-content-height)" },
          to: { height: "0" },
        },
        scan: {
          "0%": { transform: "translateX(-100%)" },
          "100%": { transform: "translateX(400%)" },
        },
        "pulse-glow": {
          "0%, 100%": { 
            opacity: "1",
            boxShadow: "0 0 5px hsl(var(--primary))"
          },
          "50%": { 
            opacity: "0.8",
            boxShadow: "0 0 20px hsl(var(--primary)), 0 0 30px hsl(var(--primary))"
          },
        },
        "gradient-x": {
          "0%, 100%": {
            "background-size": "200% 200%",
            "background-position": "left center"
          },
          "50%": {
            "background-size": "200% 200%",
            "background-position": "right center"
          }
        },
        shimmer: {
          "0%": { transform: "translateX(-100%)" },
          "100%": { transform: "translateX(100%)" },
        },
      },
      animation: {
        "accordion-down": "accordion-down 0.2s ease-out",
        "accordion-up": "accordion-up 0.2s ease-out",
        scan: "scan 2s linear infinite",
        "pulse-glow": "pulse-glow 2s ease-in-out infinite",
        "gradient-x": "gradient-x 3s ease infinite",
        shimmer: "shimmer 2s linear infinite",
      },
    },
  },
  plugins: [require("tailwindcss-animate")],
}