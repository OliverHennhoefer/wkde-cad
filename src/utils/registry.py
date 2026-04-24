from oddball import Dataset
from pyod.models.abod import ABOD
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.inne import INNE
from pyod.models.loda import LODA

def get_dataset_enum(dataset_name: str) -> Dataset:
    return getattr(Dataset, dataset_name.upper())


def get_model_instance(model_name: str, random_state: int | None = None):
    model_key = model_name.lower()
    model_class = MODEL_MAPPING[model_key]
    model = model_class()
    if random_state is not None and hasattr(model, "random_state"):
        model.random_state = random_state
    return model


DATASET_MAPPING = {
    "aloi": Dataset.ALOI,
    "annthyroid": Dataset.ANNTHYROID,
    "backdoor": Dataset.BACKDOOR,
    "breastw": Dataset.BREASTW,
    "campaign": Dataset.CAMPAIGN,
    "cardio": Dataset.CARDIO,
    "cardiotocography": Dataset.CARDIOTOCOGRAPHY,
    "celeba": Dataset.CELEBA,
    "census": Dataset.CENSUS,
    "cover": Dataset.COVER,
    "donors": Dataset.DONORS,
    "fault": Dataset.FAULT,
    "fraud": Dataset.FRAUD,
    "glass": Dataset.GLASS,
    "hepatitis": Dataset.HEPATITIS,
    "http": Dataset.HTTP,
    "internetads": Dataset.INTERNETADS,
    "ionosphere": Dataset.IONOSPHERE,
    "landsat": Dataset.LANDSAT,
    "letter": Dataset.LETTER,
    "lymphography": Dataset.LYMPHOGRAPHY,
    "magic_gamma": Dataset.MAGIC_GAMMA,
    "mammography": Dataset.MAMMOGRAPHY,
    "mnist": Dataset.MNIST,
    "musk": Dataset.MUSK,
    "optdigits": Dataset.OPTDIGITS,
    "pageblocks": Dataset.PAGEBLOCKS,
    "pendigits": Dataset.PENDIGITS,
    "pima": Dataset.PIMA,
    "satellite": Dataset.SATELLITE,
    "satimage2": Dataset.SATIMAGE2,
    "shuttle": Dataset.SHUTTLE,
    "skin": Dataset.SKIN,
    "smtp": Dataset.SMTP,
    "spambase": Dataset.SPAMBASE,
    "speech": Dataset.SPEECH,
    "stamps": Dataset.STAMPS,
    "thyroid": Dataset.THYROID,
    "vertebral": Dataset.VERTEBRAL,
    "vowels": Dataset.VOWELS,
    "waveform": Dataset.WAVEFORM,
    "wbc": Dataset.WBC,
    "wdbc": Dataset.WDBC,
    "wilt": Dataset.WILT,
    "wine": Dataset.WINE,
    "wpbc": Dataset.WPBC,
    "yeast": Dataset.YEAST,
}

MODEL_MAPPING = {
    "iforest": IForest,
    "loda": LODA,
    "inne": INNE,
    "hbos": HBOS,
    "abod": ABOD,
}
