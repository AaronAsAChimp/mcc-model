import { useEffect, useId, useMemo, useRef, useState } from 'react';
import type { PlayerOption } from '../../data/players';
import styles from './styles.module.css';

/** Cap the rendered options so the DOM stays small with 40 pickers on the page. */
const MAX_RESULTS = 50;

export interface PlayerSelectProps {
	value: string | null;
	options: readonly PlayerOption[];
	/** Players already used by another slot. They stay visible but cannot be picked. */
	usedValues?: ReadonlySet<string>;
	onChange: (value: string | null) => void;
	/** Accessible name for the trigger, for example "Red Rabbits player 2". */
	label?: string;
}

export function PlayerSelect({ value, options, usedValues, onChange, label }: PlayerSelectProps) {
	const popoverRef = useRef<HTMLDivElement | null>(null);
	const inputRef = useRef<HTMLInputElement | null>(null);
	const listRef = useRef<HTMLDivElement | null>(null);
	const anchorName = `--player-select-${useId().replace(/[^a-zA-Z0-9_-]/g, '')}`;

	const [open, setOpen] = useState(false);
	const [query, setQuery] = useState('');
	const [highlighted, setHighlighted] = useState(0);

	const { filtered, matches } = useMemo(() => {
		const terms = query.trim().toLowerCase().split(/\s+/).filter(Boolean);
		const matches = terms.length === 0
			? options
			: options.filter(option => terms.every(term => option.search.includes(term)));

		return { filtered: matches.slice(0, MAX_RESULTS), matches: matches.length };
	}, [options, query]);

	// Keep keyboard navigation inside the visible part of the list.
	useEffect(() => {
		if (!popoverRef.current?.matches(':popover-open')) {
			return;
		}

		const option = popoverRef.current.querySelector<HTMLElement>('[data-highlighted="true"]');

		option?.scrollIntoView({ block: 'nearest' });
	}, [highlighted, query]);

	function isUsed(key: string) {
		return key !== value && (usedValues?.has(key) ?? false);
	}

	function toggle() {
		const popover = popoverRef.current;

		if (!popover) {
			return;
		}

		const opening = !popover.matches(':popover-open');

		if (opening) {
			setQuery('');
			setHighlighted(0);
		}

		popover.togglePopover();

		if (opening) {
			// The popover only accepts focus once it is in the top layer.
			requestAnimationFrame(() => {
				listRef.current?.scrollTo({ top: 0 });
				inputRef.current?.focus();
			});
		}
	}

	function select(key: string | null) {
		onChange(key);
		popoverRef.current?.hidePopover();
	}

	function moveHighlight(delta: number) {
		if (filtered.length === 0) {
			return;
		}

		let next = highlighted;

		for (let step = 0; step < filtered.length; step += 1) {
			next = (next + delta + filtered.length) % filtered.length;

			const option = filtered[next];

			if (option && !isUsed(option.key)) {
				break;
			}
		}

		setHighlighted(next);
	}

	function handleKeyDown(event: React.KeyboardEvent<HTMLInputElement>) {
		if (event.key === 'ArrowDown') {
			event.preventDefault();
			moveHighlight(1);
		} else if (event.key === 'ArrowUp') {
			event.preventDefault();
			moveHighlight(-1);
		} else if (event.key === 'Enter') {
			event.preventDefault();

			const option = filtered[highlighted];

			if (option && !isUsed(option.key)) {
				select(option.key);
			}
		} else if (event.key === 'Escape') {
			popoverRef.current?.hidePopover();
		}
	}

	const displayValue = value ?? 'Select a player';
	const truncated = matches > MAX_RESULTS;

	return <div className={styles.wrapper}>
		<button
			type="button"
			className={styles.trigger}
			onClick={toggle}
			style={{ anchorName }}
			aria-label={label ? `${label}: ${displayValue}` : undefined}
			aria-haspopup="dialog"
			aria-expanded={open}
		>
			<span className={styles.value}>{displayValue}</span>
			<span className={styles.chevron} aria-hidden="true">▾</span>
		</button>
		<div
			className={styles.menu}
			popover="auto"
			ref={popoverRef}
			style={{ positionAnchor: anchorName }}
			onToggle={event => setOpen(event.currentTarget.matches(':popover-open'))}
		>
			<input
				ref={inputRef}
				className={styles.search}
				type="search"
				placeholder="Search players"
				aria-label="Search players"
				value={query}
				onChange={event => {
					setQuery(event.target.value);
					setHighlighted(0);
				}}
				onKeyDown={handleKeyDown}
			/>
			<div className={styles.list} ref={listRef}>
				<button type="button" className={styles.item} onClick={() => select(null)}>
					<em>Clear selection</em>
				</button>
				{
					filtered.map((option, index) => {
						const disabled = isUsed(option.key);

						return <button
							key={option.key}
							type="button"
							className={styles.item}
							disabled={disabled}
							data-highlighted={index === highlighted}
							aria-current={value === option.key}
							onMouseEnter={() => setHighlighted(index)}
							onClick={() => select(option.key)}
						>
							{option.key}
						</button>
					})
				}
				{filtered.length === 0 && <div className={styles.empty}>No players found</div>}
			</div>
			{truncated && <div className={styles.note}>Showing the first {MAX_RESULTS}. Keep typing to narrow the list.</div>}
		</div>
	</div>
}
