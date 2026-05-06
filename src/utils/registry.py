from oddball import Dataset
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.inne import INNE


def get_dataset_enum(dataset_name: str) -> Dataset:
    return getattr(Dataset, dataset_name.upper())


def get_model_instance(model_name: str, random_state: int | None = None):
    model_key = model_name.lower()
    try:
        model_factory = MODEL_MAPPING[model_key]
    except KeyError as exc:
        valid_options = "', '".join(sorted(MODEL_MAPPING))
        raise ValueError(
            f"Unknown model '{model_name}'. Valid options are: '{valid_options}'."
        ) from exc
    return model_factory(random_state)


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
    "iforest": lambda random_state=None: IForest(random_state=random_state),
    "inne": lambda random_state=None: INNE(random_state=random_state),
    "hbos": lambda random_state=None: HBOS(),
}
