from histolab import scorer
from histolab.tile import Tile

from ..unitutil import instance_mock


class DescribeScorers(object):
    def it_can_construct_randomscorer(self, request):
        tile = instance_mock(request, Tile)
        random_scorer = scorer.RandomScorer()

        score = random_scorer(tile)

        assert isinstance(random_scorer, scorer.RandomScorer)
        assert isinstance(random_scorer, scorer.Scorer)
        assert type(score) == float
