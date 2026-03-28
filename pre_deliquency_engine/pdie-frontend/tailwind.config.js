/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        barclays: {
          dark: '#0f172a',
          primary: '#00395D',
          accent: '#00A3E0',
          light: '#38bdf8',
        },
        risk: {
          critical: '#dc2626',
          high: '#f97316',
          medium: '#eab308',
          low: '#22c55e',
        }
      },
      fontFamily: {
        inter: ['Inter', 'system-ui', 'sans-serif'],
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
      }
    },
  },
  plugins: [],
}
