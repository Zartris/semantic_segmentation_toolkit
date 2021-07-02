import json

from ptsemseg.loader.dryharbour_4class_loader import DryHarbour4Loader
from ptsemseg.loader.dryharbour_5class_loader import DryHarbour5Loader
from ptsemseg.loader.dryharbour_5class_loader_test import DryHarbour5LoaderTest
from ptsemseg.loader.dryharbour_loader import DryHarbourLoader
from ptsemseg.loader.harbour_2class_loader import harbour2ClassLoader
from ptsemseg.loader.pascal_voc_loader import pascalVOCLoader
from ptsemseg.loader.camvid_loader import camvidLoader
from ptsemseg.loader.ade20k_loader import ADE20KLoader
from ptsemseg.loader.mit_sceneparsing_benchmark_loader import MITSceneParsingBenchmarkLoader
from ptsemseg.loader.cityscapes_loader import cityscapesLoader
from ptsemseg.loader.nyuv2_loader import NYUv2Loader
from ptsemseg.loader.sunrgbd_loader import SUNRGBDLoader
from ptsemseg.loader.mapillary_vistas_loader import mapillaryVistasLoader
from ptsemseg.loader.harbour_loader import harbourLoader


def get_loader(name):
    """get_loader

    :param name:
    """
    return {
        "pascal": pascalVOCLoader,
        "camvid": camvidLoader,
        "ade20k": ADE20KLoader,
        "mit_sceneparsing_benchmark": MITSceneParsingBenchmarkLoader,
        "cityscapes": cityscapesLoader,
        "nyuv2": NYUv2Loader,
        "sunrgbd": SUNRGBDLoader,
        "vistas": mapillaryVistasLoader,
        "harbour": harbourLoader,
        "harbour_2": harbour2ClassLoader,
        "dryharbour_6": DryHarbourLoader,
        "dryharbour_5": DryHarbour5Loader,
        "dryharbour_5_test": DryHarbour5LoaderTest,
        "dryharbour_4": DryHarbour4Loader
    }[name]
