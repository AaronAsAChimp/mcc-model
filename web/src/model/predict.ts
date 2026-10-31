import type { InferenceSession } from 'onnxruntime-web';
import wasmUrl from 'onnxruntime-web/ort-wasm-simd-threaded.wasm?url';
import { siteUrl } from '../lib/base-url';

const MAX_GAMES = 8;

interface PlayerFeatures {
	id: number;
	features: number[];
}

interface FeatureFile {
	generatedAt: string;
	featureCount: number;
	players: Record<string, PlayerFeatures>;
}

export interface PredictionInput {
	gameIds: number[];
	teams: { slug: string; players: string[] }[];
}

export interface TeamPrediction {
	slug: string;
	/** Predicted team coins, in the same units as an MCC scoreboard. */
	coins: number;
	place: number;
}

export interface Prediction {
	/** Date the player feature vectors were generated. */
	generatedAt: string;
	teams: TeamPrediction[];
}

let ortPromise: Promise<typeof import('onnxruntime-web/wasm')> | null = null;
let sessionPromise: Promise<InferenceSession> | null = null;
let featuresPromise: Promise<FeatureFile> | null = null;

function loadOrt() {
	ortPromise ??= import('onnxruntime-web/wasm');
	return ortPromise;
}

async function loadSession() {
	if (!sessionPromise) {
		sessionPromise = (async () => {
			const ort = await loadOrt();

			// A single thread avoids requiring SharedArrayBuffer, so the page does
			// not need cross-origin isolation headers.
			ort.env.wasm.numThreads = 1;
			ort.env.wasm.wasmPaths = { wasm: wasmUrl };

			return ort.InferenceSession.create(siteUrl('model/model.onnx'), {
				executionProviders: ['wasm']
			});
		})().catch(error => {
			sessionPromise = null;
			throw error;
		});
	}

	return sessionPromise;
}

async function loadFeatures() {
	if (!featuresPromise) {
		featuresPromise = fetch(siteUrl('model/player-features.json'))
			.then(response => {
				if (!response.ok) {
					throw new Error(`Could not load player features (${response.status}).`);
				}

				return response.json() as Promise<FeatureFile>;
			})
			.catch(error => {
				featuresPromise = null;
				throw error;
			});
	}

	return featuresPromise;
}

/**
 * Runs the team-placement model in the browser.
 *
 * The model needs each player's id and feature vector at prediction time. Those
 * are exported from the Python training pipeline into `player-features.json`;
 * the neural network itself runs here through ONNX Runtime Web.
 */
export async function predictEvent(input: PredictionInput): Promise<Prediction> {
	const [session, features, ort] = await Promise.all([
		loadSession(),
		loadFeatures(),
		loadOrt()
	]);

	const teamCount = input.teams.length;
	const maxPlayers = Math.max(40, ...input.teams.map(team => team.players.length));

	// if (teamCount === 0 || maxPlayers === 0) {
	// 	throw new Error('Add players to every team before predicting.');
	// }

	const teamFeatures = new Float32Array(teamCount * maxPlayers * features.featureCount);
	const playerIds = new BigInt64Array(teamCount * maxPlayers);
	const teamSizes = new BigInt64Array(teamCount);
	const gameIds = new BigInt64Array(teamCount * MAX_GAMES);

	// Id 0 is the model's learned unknown token, so the feature values for an
	// unknown player are placeholders that the model replaces internally.
	const zeroFeatures = Array.from({ length: features.featureCount }, () => 0);

	for (let teamIndex = 0; teamIndex < input.teams.length; teamIndex++) {
		const team = input.teams[teamIndex];
		const players: (string | null)[] = Array.from(team.players);
		const numPlayers = Math.max(4, players.length);

		for (let i = players.length; i < numPlayers; i++) {
			players[i] = null;
		}

		for (let i = 0; i < players.length; i++) {
			const name = players[i];
			const playerFeatures = name ? features.players[name] : null;
			const player = playerFeatures ?? { id: 0, features: zeroFeatures };
			const offset = (teamIndex * maxPlayers + i) * features.featureCount;

			teamFeatures.set(player.features, offset);
			playerIds[teamIndex * maxPlayers + i] = BigInt(player.id);
		}

		teamSizes[teamIndex] = BigInt(numPlayers);

		for (let gameIndex = 0; gameIndex < MAX_GAMES; gameIndex += 1) {
			gameIds[teamIndex * MAX_GAMES + gameIndex] = BigInt(input.gameIds[gameIndex] ?? 0);
		}
	}

	const output = await session.run({
		team_features: new ort.Tensor('float32', teamFeatures, [teamCount, maxPlayers, features.featureCount]),
		player_ids: new ort.Tensor('int64', playerIds, [teamCount, maxPlayers]),
		team_sizes: new ort.Tensor('int64', teamSizes, [teamCount]),
		game_ids: new ort.Tensor('int64', gameIds, [teamCount, MAX_GAMES])
	});

	const coins = output.coins?.data;

	if (!(coins instanceof Float32Array)) {
		throw new Error('The model returned an unexpected result.');
	}

	const teams = input.teams
		.map((team, teamIndex) => ({
			slug: team.slug,
			coins: coins[teamIndex]
		}))
		.sort((a, b) => b.coins - a.coins)
		.map((team, index) => ({ ...team, place: index + 1 }));

	return { generatedAt: features.generatedAt, teams };
}
