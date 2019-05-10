def create_weights_distribution(init, **kwargs):
    def decorator(tensor):
        init(tensor=tensor, **kwargs if kwargs is not None else None)
    return decorator
