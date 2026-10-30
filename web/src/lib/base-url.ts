const base = import.meta.env.BASE_URL;

/**
 * Absolute path of the deployed site root, always ending in a slash
 * (e.g. "/mcc-model/"). Astro's `import.meta.env.BASE_URL` omits the trailing
 * slash, so we normalize it once here.
 */
export const BASE_PATH = base.endsWith('/') ? base : `${base}/`;

/**
 * Resolve a site-relative path (e.g. "model/model.onnx") against the deploy
 * root, so links and assets keep working when the site is served from a
 * subpath like GitHub Pages project sites.
 */
export function siteUrl(path: string = ''): string {
	return `${BASE_PATH}${path.replace(/^\/+/, '')}`;
}
