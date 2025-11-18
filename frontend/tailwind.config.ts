import type { Config } from 'tailwindcss';

const config: Config = {
  content: ['./index.html', './src/**/*.{ts,tsx,js,jsx}'],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        primary: '#2D6AD5',
        accent: '#F97316',
      },
      boxShadow: {
        card: '0 10px 35px rgba(15, 23, 42, 0.12)',
      },
    },
  },
  plugins: [],
};

export default config;
