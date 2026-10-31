import { useMemo, useRef, useState } from 'react';
import type { CSSProperties } from 'react';
import GAMES from '../../../../data/games.json';
import { CURRENT_GAMES, GameSelect, type GameKey } from '../game-select/index.tsx';
import { PlayerSelect } from '../player-select/index.tsx';
import { predictEvent, type Prediction } from '../../model/predict';
import { PLAYERS } from '../../data/players';
import { TEAMS } from '../../data/teams';
import { TEAM_ICONS } from '../team-icons';
import styles from './styles.module.css';

const GAME_COUNT = 8;
const PLAYERS_PER_TEAM = 4;

type PlayerKey = string;

function shuffle<T>(items: readonly T[]): T[] {
	const result = [...items];

	// Fisher-Yates, walking backwards so each item is only swapped once.
	for (let index = result.length - 1; index > 0; index -= 1) {
		const swapIndex = Math.floor(Math.random() * (index + 1));
		[result[index], result[swapIndex]] = [result[swapIndex], result[index]];
	}

	return result;
}

export function FantasyEvent() {
	const [games, setGames] = useState<(GameKey | null)[]>(
		() => Array<GameKey | null>(GAME_COUNT).fill(null)
	);
	const [teamPlayers, setTeamPlayers] = useState<(PlayerKey | null)[][]>(
		() => TEAMS.map(() => Array<PlayerKey | null>(PLAYERS_PER_TEAM).fill(null))
	);
	const [prediction, setPrediction] = useState<Prediction | null>(null);
	const [isPredicting, setIsPredicting] = useState(false);
	const [predictionError, setPredictionError] = useState<string | null>(null);
	const predictionRequest = useRef(0);

	const usedPlayers = useMemo(
		() => new Set(teamPlayers.flat().filter((player): player is PlayerKey => player !== null)),
		[teamPlayers]
	);

	const usedGames = useMemo(
		() => new Set(games.filter((game): game is GameKey => game !== null)),
		[games]
	);

	const predictionsByTeam = useMemo(
		() => new Map(prediction?.teams.map(team => [team.slug, team]) ?? []),
		[prediction]
	);

	const selectedGameCount = games.filter(Boolean).length;
	const selectedPlayerCount = usedPlayers.size;
	const isComplete = selectedGameCount === GAME_COUNT;
		// && selectedPlayerCount === TEAMS.length * PLAYERS_PER_TEAM;

	function clearPrediction() {
		predictionRequest.current += 1;
		setPrediction(null);
		setPredictionError(null);
	}

	function updateGame(index: number, game: GameKey | null) {
		clearPrediction();
		setGames(current => current.map((value, i) => (i === index ? game : value)));
	}

	function updatePlayer(teamIndex: number, playerIndex: number, player: PlayerKey | null) {
		clearPrediction();
		setTeamPlayers(current => current.map((team, t) => {
			if (t !== teamIndex) {
				return team;
			}

			return team.map((value, p) => (p === playerIndex ? player : value));
		}));
	}

	function randomizeGames() {
		clearPrediction();
		setGames(shuffle(CURRENT_GAMES).slice(0, GAME_COUNT));
	}

	function randomizePlayers() {
		clearPrediction();
		const picked = shuffle(PLAYERS).slice(0, TEAMS.length * PLAYERS_PER_TEAM);

		setTeamPlayers(TEAMS.map((_, teamIndex) => {
			const start = teamIndex * PLAYERS_PER_TEAM;

			return picked.slice(start, start + PLAYERS_PER_TEAM).map(player => player.key);
		}));
	}

	async function predict() {
		if (!isComplete || isPredicting) {
			return;
		}

		const request = predictionRequest.current + 1;
		predictionRequest.current = request;
		setIsPredicting(true);
		setPredictionError(null);

		try {
			const result = await predictEvent({
				gameIds: games.map(game => (game ? GAMES[game].id : 0)),
				teams: TEAMS.map((team, index) => ({
					slug: team.slug,
					players: teamPlayers[index].filter((player): player is PlayerKey => player !== null)
				}))
			});

			// Ignore a result that arrived after the user changed the event.
			if (request === predictionRequest.current) {
				setPrediction(result);
			}
		} catch (error) {
			if (request === predictionRequest.current) {
				setPrediction(null);
				setPredictionError(error instanceof Error ? error.message : 'Prediction failed.');
			}
		} finally {
			setIsPredicting(false);
		}
	}

	return <div className={styles.fantasy}>
		<section className={styles.section}>
			<div className={styles.sectionHeader}>
				<div>
					<h2>Games</h2>
					<p className={styles.hint}>
						{selectedGameCount} of {GAME_COUNT} games selected
					</p>
				</div>
				<button type="button" className={styles.action} onClick={randomizeGames}>
					Randomize games
				</button>
			</div>
			<div className={styles.games}>
				{
					games.slice(0, 4).map((game, index) => {
						return <GameSelect
							key={index}
							value={game}
							usedValues={usedGames}
							label={`Game ${index + 1}`}
							onChange={next => updateGame(index, next)}
						/>
					})
				}
				<div className={styles['event-intermission']}>Intermission</div>
				{
					games.slice(4, 8).map((game, index) => {
						return <GameSelect
							key={index}
							value={game}
							usedValues={usedGames}
							label={`Game ${index + 5}`}
							onChange={next => updateGame(index, next)}
						/>
					})
				}
			</div>
		</section>

		<section className={styles.section}>
			<div className={styles.sectionHeader}>
				<div>
					<h2>Teams</h2>
					<p className={styles.hint}>
						{selectedPlayerCount} of {TEAMS.length * PLAYERS_PER_TEAM} players selected
					</p>
				</div>
				<div className={styles.actions}>
					<button type="button" className={styles.action} onClick={randomizePlayers}>
						Randomize players
					</button>
					<button
						type="button"
						className={styles.predictAction}
						onClick={predict}
						disabled={!isComplete || isPredicting}
						title={isComplete ? undefined : 'Select all 8 games and 40 players first.'}
					>
						{isPredicting ? 'Predicting…' : 'Predict event'}
					</button>
				</div>
			</div>
			{predictionError && <p className={styles.error} role="alert">{predictionError}</p>}
			<div className={styles.teams}>
				{
					TEAMS.map((team, teamIndex) => {
						const teamStyle = { '--team-color': team.color } as CSSProperties;

						return <div key={team.slug} className={styles.team} style={teamStyle}>
							<h3 className={styles['team-name']}>{team.name}</h3>
							<div className={styles['team-icon']}>
								{ TEAM_ICONS[team.slug] }
							</div>
							<div className={styles['team-players']}>
								{
									teamPlayers[teamIndex].map((player, playerIndex) => {
										return <PlayerSelect
											key={playerIndex}
											value={player}
											options={PLAYERS}
											usedValues={usedPlayers}
											label={`${team.name} player ${playerIndex + 1}`}
											onChange={next => updatePlayer(teamIndex, playerIndex, next)}
										/>
									})
								}
							</div>
						</div>
					})
				}
			</div>
			{
				prediction && <div className={styles.results} aria-live="polite">
					<div className={styles.resultsHeader}>
						<h3>Predicted standings</h3>
						<span className={styles.modelNote}>player data {prediction.generatedAt}</span>
					</div>
					<ol className={styles.leaderboard}>
						{
							prediction.teams.map((team, idx) => {
								const info = TEAMS.find(candidate => candidate.slug === team.slug);
								const teamColor = { '--team-color': info?.color } as CSSProperties;
								let className = styles.standing

								if (idx == 0 || idx == 1) {
									className += ' ' + styles.dodgebolt;
								}

								return <li key={team.slug} style={teamColor} className={className}>
									<span className={styles.rank}>{team.place}</span>
									<div className={styles.dot} aria-hidden="true">
										{ TEAM_ICONS[team.slug] }
									</div>
									<span className={styles.standingName}>{info?.name ?? team.slug}</span>
									<span className={styles.coins}>{Math.round(team.coins).toLocaleString()} coins</span>
								</li>
							})
						}
					</ol>
				</div>
			}
		</section>
	</div>
}
