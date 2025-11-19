from nonconform.utils.data import Dataset
from pyod.models.copod import COPOD
from pyod.models.ecod import ECOD
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.inne import INNE
from pyod.models.loda import LODA


def get_dataset_enum(dataset_name: str) -> Dataset:
    return getattr(Dataset, dataset_name.upper())


def get_model_instance(model_name: str):
    model_class = MODEL_MAPPING[model_name.lower()]
    return model_class()


DATASET_MAPPING = {
    "hepatitis": Dataset.HEPATITIS,
    "wbc": Dataset.WBC,
    "ionosphere": Dataset.IONOSPHERE,
    "breast": Dataset.BREAST,
    "cardio": Dataset.CARDIO,
    "musk": Dataset.MUSK,
    "mammography": Dataset.MAMMOGRAPHY,
    "shuttle": Dataset.SHUTTLE,
}

MODEL_MAPPING = {
    "iforest": IForest,
    "loda": LODA,
    "inne": INNE,
    "hbos": HBOS,
    "copod": COPOD,
    "ecod": ECOD,
}
