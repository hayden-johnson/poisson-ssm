from pvae.encoder import PL_PVAE

def build():
    load_encoder = False
    load_decoder = False
    load_decoder_dataset = False

    ## SET UP ENCODER
    if load_encoder:
        pass

    else:
        # load dataset
        # prep data
        # train encoder
        pass

    ## SET UP DECODER
    if load_decoder:
        pass

    else:
        if load_decoder_dataset:
            # load decoder dataset
            pass
        else:
            # create decoder dataset with encoder model
            pass

        # create decoder dataset
        # train decoder
        pass

    ## CREATE COMPOSITE MODEL

    # return model

    pass
        
def evaluate():
    pass

if __name__ == "__main__":
    build()
    evaluate()