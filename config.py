class CFG:
    DATA_PATH = "/groups/gcd50697/sakamoto/project/CLIP"
    DF_PATH = DATA_PATH+"/captions.csv"
    MODEL_SAVE_PATH = "./outs/weight.pth"

    wandb = {"project":"CLIP",
            "group":"test",
            "name":"test",
            "notes":"test"}

    EPOCH = 1
    IMG_SIZE = 224
    MAX_LENGTH = 200
    VAL_SET = 1

    projection_dim = 256
    temperature = 1

    image_encoder_model = "resnet50"
    image_embedding = 2048

    text_tokenizer = "distilbert-base-uncased"
    text_encoder_model = "distilbert-base-uncased"
    text_embedding = 768

    trainable = True
    inference = True
    test_query = "a girl jumping from swing"
    dropout = 0.1

    weight_decay = 1e-3
    image_encoder_lr = 1e-4
    text_encoder_lr = 1e-5
    head_lr =1e-3


    class dataloader:
        train = {
            "num_workers":10,
            "batch_size":32,
            "shuffle":True
            }

        val = {
            "num_workers":1,
            "batch_size":32,
            "shuffle":False
            }