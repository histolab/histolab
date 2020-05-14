from src.histolab.filters import image_filters_functional as F


class Compose(object):
    """Composes several filters together.

    Parameters
    ----------
    filters : list of Filters
        List of filters to compose
    """

    def __init__(self, filters):
        self.filters = filters

    def __call__(self, img):
        for f in self.filters:
            img = f(img)
        return img


class Invert(object):
    """Invert an image."""

    def __call__(self, img):
        """
        Parameters
        ----------
        img : PIL.Image.Image
            Input image

        Returns
        -------
        PIL.Image.Image
            Inverted image
        """
        return F.invert(img)

    def __repr__(self):
        return self.__class__.__name__ + "()"
