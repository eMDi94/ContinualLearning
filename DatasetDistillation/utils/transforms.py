

class FlatTransform(object):

    def __call__(self, x):
        return x.view(-1)
