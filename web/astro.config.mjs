// @ts-check
import { defineConfig } from 'astro/config';

import react from '@astrojs/react';

// https://astro.build/config
export default defineConfig({
  // GitHub Pages serves this repo from https://aaronaschimp.github.io/mcc-model/.
  // Change both values (site: 'https://<user>.github.io', base: '/') if a custom
  // domain is ever set up.
  site: 'https://aaronaschimp.github.io',
  base: '/mcc-model',
  integrations: [react()]
});
