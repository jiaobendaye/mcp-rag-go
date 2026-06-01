import { defineConfig } from 'vite';
import { svelte } from '@sveltejs/vite-plugin-svelte';
import { resolve } from 'node:path';

export default defineConfig(({ command }) => ({
  base: command === 'build' ? '/static/app/' : '/',
  plugins: [svelte()],
  build: {
    outDir: resolve(__dirname, '../static/app'),
    emptyOutDir: true,
    assetsDir: 'assets',
    manifest: false,
  },
}));
