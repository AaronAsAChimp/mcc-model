import torch

from team import PLAYER_EMBEDDING_SIZE, PLAYERS_PER_TEAM, FEATURES_PER_PLAYER

# Build a model that, given the shooter and remaining playeres, give the probabilities of their target
# Do a weighted random choice of target
# Build a model that given a shooter and a target gives the probability of the shot being successful
# Do a weighted random choice of success
# Iterate between the two models to play out games


TEAM_INPUT_SIZE = PLAYERS_PER_TEAM + (PLAYERS_PER_TEAM * PLAYER_EMBEDDING_SIZE) + (PLAYERS_PER_TEAM * FEATURES_PER_PLAYER)

class DodgeboltPredictor(torch.nn.Module):
	_output: torch.nn.Linear

	def __init__(self):
		super().__init__()

		self._output = torch.nn.Linear(TEAM_INPUT_SIZE, PLAYERS_PER_TEAM)

	def forward(self,
        shooting_player_mask: torch.Tensor,
		shooting_team_features: torch.Tensor,
        shooting_player_ids: torch.Tensor,
        targeted_player_mask: torch.Tensor,
        targeted_team_features: torch.Tensor,
        targeted_player_ids: torch.Tensor):

		team_tensor = torch.concat((shooting_player_mask, shooting_team_features, shooting_player_ids))

		return self._output(team_tensor)

