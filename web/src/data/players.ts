import PLAYER_DATA from '../../../data/players.json';

interface PlayerRecord {
	ign?: string;
	aliases?: string[];
}

export interface PlayerOption {
	/** Canonical player name, as used in the event data. */
	key: string;
	/** Searchable text built from the canonical name, IGN, and aliases. */
	search: string;
}

const records = PLAYER_DATA as Record<string, PlayerRecord | string>;

// Every player that has appeared in an MCC event, sorted for the picker.
export const PLAYERS: PlayerOption[] = Object.entries(records)
	.filter(([key, record]) => key !== '$schema' && typeof record === 'object')
	.map(([key, record]) => {
		const player = record as PlayerRecord;
		const search = [key, player.ign, ...(player.aliases ?? [])]
			.filter(Boolean)
			.join(' ')
			.toLowerCase();

		return { key, search };
	})
	.sort((a, b) => a.key.localeCompare(b.key));
