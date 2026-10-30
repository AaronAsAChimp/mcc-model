import { useId, useRef } from 'react';
import GAMES from '../../../../data/games.json';
import styles from './styles.module.css';

export type GameKey = keyof typeof GAMES;
type GameStatus = 'current' | 'retired' | 'special';

interface GameEntry {
	name: string;
	status: string;
}

const GAME_ENTRIES = Object.entries(GAMES) as [GameKey, GameEntry][];

/** Games that are still played in current MCC events. */
export const CURRENT_GAMES = GAME_ENTRIES
	.filter(([, game]) => game.status === 'current')
	.map(([key]) => key);

// The status field in the JSON is a plain string, so group explicitly instead
// of relying on its order to keep the menu sections predictable.
const GAME_GROUPS = (['current', 'retired', 'special'] as const)
	.map(status => ({
		status,
		options: GAME_ENTRIES
			.filter(([, game]) => game.status === status)
			.map(([key, game]) => ({ key, name: game.name }))
	}))
	.filter(group => group.options.length > 0);

export interface GameSelectProps {
	value: GameKey | null;
	/** Games already picked in another slot. They stay visible but cannot be picked. */
	usedValues?: ReadonlySet<GameKey>;
	onChange?: (value: GameKey | null) => void;
	/** Accessible name for the trigger, for example "Game 3". */
	label?: string;
}

export function GameSelect({ value, usedValues, onChange, label }: GameSelectProps) {
	const popoverRef = useRef<HTMLDivElement | null>(null);
	const anchorName = `--game-select-${useId().replace(/[^a-zA-Z0-9_-]/g, '')}`;

	const displayValue = value ? GAMES[value].name : 'Select a game';

	function isUsed(key: GameKey) {
		return key !== value && (usedValues?.has(key) ?? false);
	}

	function toggle() {
		popoverRef.current?.togglePopover();
	}

	function select(key: GameKey | null) {
		onChange?.(key);

		if (popoverRef.current) {
			popoverRef.current.hidePopover();
		}
	}

	return <div className={styles.wrapper}>
		<button
			type="button"
			className={styles.trigger}
			onClick={toggle}
			style={{ anchorName }}
			aria-label={label ? `${label}: ${displayValue}` : undefined}
		>
			<span>{displayValue}</span>
			<span className={styles.chevron} aria-hidden="true">▾</span>
		</button>
		<div
			className={styles.menu}
			popover="auto"
			ref={popoverRef}
			style={{ positionAnchor: anchorName }}
		>
			<button type="button" className={styles.item} onClick={() => select(null)}>
				<em>Clear selection</em>
			</button>
			{
				GAME_GROUPS.map(group => {
					return <div key={group.status}>
						<div className={styles.groupHeader}>{group.status}</div>
						{group.options.map(option => {
							return <button
								key={option.key}
								type="button"
								className={styles.item}
								disabled={isUsed(option.key)}
								aria-current={value === option.key}
								onClick={() => select(option.key)}
							>
								{option.name}
							</button>
						})}
					</div>
				})
			}
		</div>
	</div>
}
