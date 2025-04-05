from configs import default as config


class MGTrainer( ):

    def __init__(self,
                model               = None,
                input_dims          = None,
                optim               = None,


                **kwargs
                ):
        
        self.model      = model
        self.input_dims = input_dims
        
        if optim == None:

            optim = config.DEFAULTS["optim"]

        ...
    ...