# encoding: utf-8

import os

from histolab.masks import BiggestTissueBoxMask
from histolab.scorer import RandomScorer
from histolab.slide import Slide
from histolab.tiler import GridTiler, RandomTiler, ScoreTiler

from ..fixtures import SVS


class Describe_TilerMppReuse:
    def it_reuses_grid_tiler_across_slides_with_mpp(self, tmpdir):
        slide_1 = Slide(SVS.CMU_1_SMALL_REGION, os.path.join(tmpdir, "out1"), True)
        slide_2 = Slide(SVS.CMU_1_SMALL_REGION, os.path.join(tmpdir, "out2"), True)
        tiler = GridTiler(
            tile_size=(128, 128), pixel_overlap=16, mpp=0.75, check_tissue=False
        )

        tiler.extract(slide_1, BiggestTissueBoxMask())
        tile_size_after_first = tiler.tile_size
        overlap_after_first = tiler.pixel_overlap

        tiler.extract(slide_2, BiggestTissueBoxMask())

        assert tiler.tile_size == tile_size_after_first
        assert tiler.pixel_overlap == overlap_after_first
        assert os.listdir(slide_2.processed_path)

    def it_reuses_random_tiler_across_slides_with_mpp(self, tmpdir):
        slide_1 = Slide(SVS.CMU_1_SMALL_REGION, os.path.join(tmpdir, "out1"), True)
        slide_2 = Slide(SVS.CMU_1_SMALL_REGION, os.path.join(tmpdir, "out2"), True)
        tiler = RandomTiler(
            tile_size=(128, 128), n_tiles=2, mpp=0.75, check_tissue=False
        )

        tiler.extract(slide_1, BiggestTissueBoxMask())
        tile_size_after_first = tiler.tile_size

        tiler.extract(slide_2, BiggestTissueBoxMask())

        assert tiler.tile_size == tile_size_after_first
        assert len(os.listdir(slide_2.processed_path)) == 2

    def it_reuses_score_tiler_across_slides_with_mpp(self, tmpdir):
        slide_1 = Slide(SVS.CMU_1_SMALL_REGION, os.path.join(tmpdir, "out1"), True)
        slide_2 = Slide(SVS.CMU_1_SMALL_REGION, os.path.join(tmpdir, "out2"), True)
        tiler = ScoreTiler(
            scorer=RandomScorer(),
            tile_size=(128, 128),
            n_tiles=2,
            pixel_overlap=16,
            mpp=0.75,
            check_tissue=False,
        )

        tiler.extract(slide_1, BiggestTissueBoxMask())
        tile_size_after_first = tiler.tile_size
        overlap_after_first = tiler.pixel_overlap

        tiler.extract(slide_2, BiggestTissueBoxMask())

        assert tiler.tile_size == tile_size_after_first
        assert tiler.pixel_overlap == overlap_after_first
        assert len(os.listdir(slide_2.processed_path)) == 2
