def create_weights_distribution(init, **kwargs):
    def decorator(tensor):
        init(tensor=tensor, **kwargs)
    return decorator
