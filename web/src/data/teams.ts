export interface Team {
	slug: string;
	name: string;
	color: string;
}

// The ten MCC team colours, in the order they are listed for an event.
export const TEAMS: Team[] = [
	{ slug: 'redrabbits', name: 'Red Rabbits', color: '#e63946' },
	{ slug: 'orangeocelots', name: 'Orange Ocelots', color: '#f77f00' },
	{ slug: 'yellowyaks', name: 'Yellow Yaks', color: '#e5b800' },
	{ slug: 'limellamas', name: 'Lime Llamas', color: '#6a994e' },
	{ slug: 'greengeckos', name: 'Green Geckos', color: '#2a9d5c' },
	{ slug: 'cyancoyotes', name: 'Cyan Coyotes', color: '#0096c7' },
	{ slug: 'aquaaxolotls', name: 'Aqua Axolotls', color: '#48bfe3' },
	{ slug: 'bluebats', name: 'Blue Bats', color: '#3a5bdb' },
	{ slug: 'purplepandas', name: 'Purple Pandas', color: '#8e44ad' },
	{ slug: 'pinkparrots', name: 'Pink Parrots', color: '#e0568c' }
];
