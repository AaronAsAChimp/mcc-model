"""Project learned player / game embeddings to 2D and render interactive plots.

Runs from the repo root:

    uv run python packages/player-embedding/visualize_embeddings.py

The player chart has a dropdown that toggles between coloring by skill
proxy and highlighting membership in a community group (e.g. Hermitcraft,
Dream SMP).
"""

import json

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go
from shared_core.data import PlayerDataFile
import torch

from typing import Iterable, TypedDict

from games import GAME_IDS
from main import TeamPlacementPredictor
from plotly.utils import PlotlyJSONEncoder

EXTRACTS_FILENAME = "data/extracts.sorted.jsonl"
MODEL_FILENAME = "packages/player-embedding/model/model.pt"
CONFIG_FILENAME = "packages/player-embedding/model/config.json"
OUTPUT_DIRECTORY = "packages/player-embedding/visualizations"
LABEL_TOP_PLAYERS = 12

NONMEMBER_COLOR = "#3a4150"
GROUP_COLORS = {
    "Hermitcraft": "#e3b341",
    "Dream SMP": "#4fa3ff",
    "Life series": "#c83235",
    "Lifesteal SMP": "#ce1818",
}


class EmbeddingView(TypedDict):
    title: str
    marker: dict[str, object]
    label_indices: list[int]
    text: list[str]

#
# Group memberships, using player names exactly as they appear in the
# trained model's config.json (packages/player-embedding/model/config.json).
#
# Hermitcraft membership covers everyone who has been whitelisted on the
# vanilla server across seasons. Dream SMP membership follows the cast list
# on the Dream SMP Wikipedia page.
#
GROUP_MEMBERS = {
    "Hermitcraft": {
        "BdoubleO100", "cubfan135", "Docm77", "EthosLab", "FalseSymmetry",
        "GeminiTay", "GoodTimesWithScar", "Grian", "hypnotizd", "iJevin",
        "impulseSV", "Keralis", "Mumbo Jumbo", "PearlescentMoon",
        "Rendog", "Skizzleman", "Smallishbeans", "TangoTek", "VintageBeef",
        "Welsknight", "xBCrafted", "Xisumavoid", "Zedaph", "ZombieCleo",
    },
    "Dream SMP": {
        "aimsey", "Antfrost", "Awesamdude", "BadBoyHalo", "CaptainPuffy",
        "ConnorEatsPants", "Dream", "Eret", "Foolish Gamers", "Fundy",
        "GeorgeNotFound", "hannahxxrose", "HBomb94", "JackManifoldTV",
        "Karl Jacobs", "LazarBeam", "Michaelmcchill", "Nihachu", "Ph1LzA",
        "Ponk", "Punz", "Purpled", "Quackity", "Ranboo", "Sapnap",
        "Seapeekay", "Skeppy", "Slimecicle", "Technoblade", "TinaKitten",
        "TommyInnit", "Tubbo", "Vikkstar123", "Wilbur Soot",
    },
    "Life series": {
        "BdoubleO100", "Bigbst4tz2", "EthosLab", "GeminiTay",
        "GoodTimesWithScar", "Grian", "impulseSV", "InTheLittleWood",
        "LDShadowLady", "MumboJumbo", "PearlescentMoon", "Rendog", "Skizzleman",
        "Smajor1995", "SmallishBeans", "Solidarity", "TangoTek", "ZombieCleo"
    },
    "Lifesteal SMP": {
        # Current
        "4CVIT",
        "ashswagg",
        "baconnwaffles0",
        "BranzyCraft",
        "CHIEFXD",
        "ClownPierce",
        "CookieBunsquat",
        "Derapchu",
        "Dtowncat",
        "ECorridor",
        "Epaxialfx",
        "Flowtives",
        "fruitberries",
        "hannahxxrose",
        "Infume",
        "Jaden_MAN",
        "Jepexx",
        "jojosolos",
        "JumperWho",
        "KatieGoBrr",
        "leekleek",
        "Leow0ok",
        "loppezz",
        "LuigiToan",
        "Mapicc",
        "MinuteTech",
        "MrCube6",
        "Only_A_Squid",
        "Pangi",
        "Peentar",
        "Poafa",
        "PrinceZam",
        "Reddoons",
        "rekrap2",
        "roshambogames",
        "SB737",
        "Spepticle",
        "Spongs",
        "TheRealSquiddo",
        "TheTerrain",
        "Vitalasy",
        "Vort3xDragon",
        "WithMayX",
        "Woogiex",
        "Yeah_Jaron",
        "yungyx",        
    }
}

GAME_TYPE = {
    "PvP": {
        "battlebox",
        "meltdown",
        "skybattle",
        "survivalgames"
    },
    "Movement": {
        "acerace",
        "holeinthewall",
        "parkourtag",
        "parkourwarrior",
        "rocketspleefrush",
        "tgttosawaf"
    },
    "Team": {
        "buildmart",
        "bingobutfast",
        "gridrunners",
        "railroadrush",
        "sandsoftime"
    }
}

#
# Friendly display names for game slugs.
# Source: mcchampionship.fandom.com
#
FRIENDLY_GAME_NAMES = {
    "acerace": "Ace Race",
    "battlebox": "Battle Box",
    "buildmart": "Build Mart",
    "bingobutfast": "Bingo but Fast",
    "gridrunners": "Grid Runners",
    "holeinthewall": "Hole in the Wall",
    "meltdown": "Meltdown",
    "parkourtag": "Parkour Tag",
    "parkourwarrior": "Parkour Warrior",
    "railroadrush": "Railroad Rush",
    "rocketspleefrush": "Rocket Spleef Rush",
    "sandsoftime": "Sands of Time",
    "skybattle": "Sky Battle",
    "survivalgames": "Survival Games",
    "tgttosawaf": "TGTTOSAWAF",
    "footrace": "Footrace",
    "lockoutbingo": "Lockout Bingo",
    "parkourwarriororig": "Parkour Warrior (Original)",
    "rocketspleef": "Rocket Spleef",
    "skyblockle": "Skyblockle",
    "parkourwarriorsurvivor": "Parkour Warrior Survivor",
    "buildmartirl": "Build Mart IRL",
}


def pca_2d(weights: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Project a weight matrix to 2D via PCA (SVD on the centered matrix).

    Returns (coordinates, explained_variance_ratio).
    """
    centered = weights - weights.mean(dim=0, keepdim=True)
    _, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
    coords = (centered @ vh[:2, :].T).numpy()
    explained = (singular_values[:2] ** 2) / (singular_values**2).sum()
    return coords, explained.numpy()


def skill_stats(
    extract_filename: str,
) -> dict[str, dict[str, float | int]]:
    """Per-player performance stats from the raw event data.

    Skill proxy is the model's own avg_placement feature: 1 - (mean placement - 1) / 9.
    """
    placements: dict[str, list[int]] = defaultdict(list)

    with open(extract_filename, "r") as fd:
        for json_line in fd:
            event = json.loads(json_line)

            for team in event["teams"].values():
                placement = team.get("placement")

                for player in team["players"]:
                    placements[player].append(placement)

    stats: dict[str, dict[str, float | int]] = {}

    for player, player_placements in placements.items():
        event_count = len(player_placements)
        wins = sum(1 for placement in player_placements if placement == 1)
        top_two = sum(1 for placement in player_placements if placement <= 2)
        mean_placement = sum(player_placements) / event_count
        skill = 1 - (mean_placement - 1) / 9

        stats[player] = {
            "skill": skill,
            "events": event_count,
            "wins": wins,
            "top_two": top_two,
            "avg_placement": mean_placement,
        }

    return stats


def dark_layout(title: str, variance: np.ndarray) -> dict:
    return {
        "title": title,
        "paper_bgcolor": "#0d1117",
        "plot_bgcolor": "#0d1117",
        "font": {"color": "#c9d1d9", "family": "Helvetica, Arial, sans-serif"},
        "xaxis": {
            "title": f"PC1 ({variance[0] * 100:.1f}% variance)",
            "gridcolor": "#21262d",
            "zerolinecolor": "#30363d",
        },
        "yaxis": {
            "title": f"PC2 ({variance[1] * 100:.1f}% variance)",
            "gridcolor": "#21262d",
            "zerolinecolor": "#30363d",
        },
        "showlegend": False,
    }


def build_views(
    player_names: list[str],
    skills: list[float],
) -> dict[str, EmbeddingView]:
    """Per-toggle colorings and the player indices to label in each view."""
    views: dict[str, EmbeddingView] = {}

    top_indices = sorted(
        range(len(player_names)), key=lambda index: skills[index], reverse=True
    )[:LABEL_TOP_PLAYERS]

    top_set = set(top_indices)

    views["Skill"] = {
        "title": "Skill proxy",
        "marker": {
            "size": 7,
            "color": skills,
            "colorscale": "viridis",
            "showscale": True,
            "colorbar": {"title": "Skill proxy", "x": 1.02},
            "line": {"width": 0.5, "color": "#0d1117"},
        },
        "label_indices": top_indices,
        "text": [
            player_names[index] if index in top_set else ""
            for index in range(len(player_names))
        ],
    }

    with PlayerDataFile('data/players.json') as player_data:
        index_by_name = {}

        for index, name in enumerate(player_names):
            index_by_name[player_data.normalize_name(name)] = index

        for group, members in GROUP_MEMBERS.items():
            member_indices = [
                index_by_name[player_data.normalize_name(name)] for name in members if name in index_by_name
            ]
            member_set = set(member_indices)
            accent = GROUP_COLORS[group]
            colors = [
                accent if index in member_set else NONMEMBER_COLOR
                for index in range(len(player_names))
            ]
            sizes = [
                9 if index in member_set else 5 for index in range(len(player_names))
            ]

            views[group] = {
                "title": group,
                "marker": {
                    "size": sizes,
                    "color": colors,
                    "showscale": False,
                    "line": {"width": 0.6, "color": "#0d1117"},
                },
                "label_indices": member_indices,
                "text": [
                    player_names[index] if index in member_set else ""
                    for index in range(len(player_names))
                ],
            }

    return views


def group_hover_suffix(name: str) -> str:
    groups = [group for group in GROUP_MEMBERS if name in GROUP_MEMBERS[group]]

    return " | ".join(groups) if groups else "—"


@dataclass
class PlayerData:
    player_names: list[str]
    skills: list[float]
    coords: np.ndarray
    variance: np.ndarray
    customdata: list[list[object]]
    views: dict[str, EmbeddingView]


def load_player_data() -> PlayerData:
    with open(CONFIG_FILENAME, "r") as fd:
        config = json.load(fd)

    player_count = int(config["player_count"])
    player_ids = {
        name: int(player_id) for name, player_id in config["player_ids"].items()
    }
    names_by_id = {player_id: name for name, player_id in player_ids.items()}

    model = TeamPlacementPredictor(player_count)
    model.load_state_dict(torch.load(MODEL_FILENAME, weights_only=True))
    model.eval()

    player_names = [names_by_id[player_id] for player_id in range(player_count)]
    stats = skill_stats(EXTRACTS_FILENAME)
    skills = [float(stats[name].get("skill", 0)) for name in player_names]

    coords, variance = pca_2d(model.player_embeddings.weight.detach())

    customdata = []

    for name in player_names:
        player_stats = stats.get(name, {"skill": 0, "events": 0, "wins": 0, "top_two": 0})
        customdata.append(
            [
                name,
                float(player_stats["skill"]),
                int(player_stats["events"]),
                int(player_stats["wins"]),
                int(player_stats["top_two"]),
                group_hover_suffix(name),
            ]
        )

    views = build_views(player_names, skills)

    return PlayerData(
        player_names=player_names,
        skills=skills,
        coords=coords,
        variance=variance,
        customdata=customdata,
        views=views,
    )


def build_player_figure(data: PlayerData) -> go.Figure:
    fig = go.Figure(
        data=[
            go.Scattergl(
                x=data.coords[:, 0],
                y=data.coords[:, 1],
                mode="markers+text",
                marker=data.views["Skill"]["marker"],
                text=data.views["Skill"]["text"],
                textposition="top center",
                textfont={"size": 11, "color": "#e6edf3"},
                customdata=data.customdata,
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Skill proxy: %{customdata[1]:.3f}<br>"
                    "Events: %{customdata[2]}<br>"
                    "Wins: %{customdata[3]}<br>"
                    "Top 2: %{customdata[4]}<br>"
                    "Group: %{customdata[5]}"
                    "<extra></extra>"
                ),
                name="players",
            )
        ]
    )
    fig.update_layout(**dark_layout("MCC Player Embeddings — Skill proxy", data.variance))
    return fig


def player_switcher_script(data: PlayerData, div_id: str) -> str:
    """Inline JS that overlays a view toggle and fully redraws the chart on
    selection over `Plotly.react`, because plotly.js 3.x restyle drops array
    updates to the scatter `text` attribute."""
    views_json = json.dumps(data.views, cls=PlotlyJSONEncoder)

    return f"""
(function() {{
    const graph = document.getElementById('{div_id}');
    const scenes = {views_json};

    const wrapper = document.createElement('div');
    wrapper.style.cssText =
        'position:absolute;top:10px;left:10px;z-index:50;';

    const select = document.createElement('select');
    select.id = 'mcc-view-select';
    select.style.cssText =
        'background:#161b22;color:#c9d1d9;border:1px solid #30363d;' +
        'border-radius:6px;padding:4px 8px;font-size:12px;' +
        'font-family:Helvetica,Arial,sans-serif;outline:none;';

    for (const label of Object.keys(scenes)) {{
        const option = document.createElement('option');
        option.value = label;
        option.textContent = label;
        select.appendChild(option);
    }}

    wrapper.appendChild(select);
    graph.style.position = 'relative';
    graph.appendChild(wrapper);

    select.addEventListener('change', function () {{
        const view = scenes[this.value];
        const trace = Object.assign({{}}, graph.data[0], {{
            marker: view.marker,
            text: view.text,
        }});
        const layout = Object.assign({{}}, graph.layout, {{
            title: {{ text: 'MCC Player Embeddings — ' + view.title }},
        }});
        Plotly.react(graph, [trace], layout);
    }});
}})();
"""


def build_game_marker(label: str, member_set: set[str], games: Iterable[str]) -> dict:
    color = [
        "#e3b341" if game in member_set else NONMEMBER_COLOR
        for game in games
    ]

    return {
        'args': [
            "marker",
            {
                "size": 9,
                "color": color,
                "line": {
                    "width": 0.5,
                    "color": "#0d1117"
                }
            }
        ],
        'label': label,
        'method': 'restyle'
    }


def build_game_figure() -> go.Figure:
    with open(CONFIG_FILENAME, "r") as fd:
        config = json.load(fd)

    model = TeamPlacementPredictor(int(config["player_count"]))
    model.load_state_dict(torch.load(MODEL_FILENAME, weights_only=True))
    model.eval()

    # Drop the padding index 0, so rows line up with GAME_IDS values.
    coords, variance = pca_2d(model.game_id_embedding.weight.detach()[1:])

    slugs_by_id = {game_id: slug for slug, game_id in GAME_IDS.items()}
    game_ids = sorted(slugs_by_id.keys())
    index_list = [game_id - 1 for game_id in game_ids]
    labels = [FRIENDLY_GAME_NAMES[slugs_by_id[game_id]] for game_id in game_ids]

    markers = go.Scattergl(
        x=coords[index_list, 0],
        y=coords[index_list, 1],
        mode="markers",
        marker={"size": 9, "color": "#e3b341", "line": {"width": 0.5, "color": "#0d1117"}},
        customdata=game_ids,
        hovertemplate="%{customdata}<br><extra></extra>",
        name="games",
    )

    labels_trace = go.Scatter(
        x=coords[index_list, 0],
        y=coords[index_list, 1],
        mode="text",
        text=labels,
        textposition="top center",
        textfont={"size": 12, "color": "#e6edf3"},
        hoverinfo="skip",
        name="labels",
    )

    buttons = [build_game_marker(type, games, GAME_IDS.keys()) for type, games in GAME_TYPE.items()]

    fig = go.Figure(data=[markers, labels_trace])
    fig.update_layout(**dark_layout("MCC Game Embeddings", variance))
    fig.update_layout(
        updatemenus=[
            dict(
                buttons=buttons,
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.1,
                xanchor="left",
                y=1.1,
                yanchor="top"
            ),
        ]
    )

    return fig


def main() -> None:
    import os

    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

    for group, members in GROUP_MEMBERS.items():
        print(f"{group}: {len(members)} members in dataset")

    data = load_player_data()

    player_path = f"{OUTPUT_DIRECTORY}/player_embeddings.html"
    build_player_figure(data).write_html(
        player_path,
        div_id="mcc-player-chart",
        post_script=[player_switcher_script(data, "mcc-player-chart")],
    )
    print(f"Wrote {player_path}")

    game_path = f"{OUTPUT_DIRECTORY}/game_embeddings.html"
    build_game_figure().write_html(game_path)
    print(f"Wrote {game_path}")


if __name__ == "__main__":
    main()