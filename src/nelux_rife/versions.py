"""Every RIFE model vs-rife ships, as data.

This replaces the 200-line ``match model:`` block in vs-rife's ``rife()``. Each
entry names the IFNet module, the feature encoder ("head") that ``prepare``
runs, the padding modulo, whether ``ensemble`` is supported, and the released
weights with their SHA-256.
"""

import importlib
from collections.abc import Callable
from dataclasses import dataclass

import torch.nn as nn

WEIGHTS_URL = "https://github.com/HolyWu/vs-rife/releases/download/model/flownet_v{}.pkl"


def _head_16_4() -> nn.Module:
    return nn.Sequential(nn.Conv2d(3, 16, 3, 2, 1), nn.ConvTranspose2d(16, 4, 4, 2, 1))


def _head_32(out_channels: int) -> Callable[[], nn.Module]:
    def make() -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose2d(32, out_channels, 4, 2, 1),
        )

    return make


# Head kinds: None (no encoder; prepare passes the frame through), "module"
# (the Head class defined next to IFNet), or a function building vs-rife's inline nn.Sequential.
NO_HEAD = None
MODULE_HEAD = "module"


@dataclass(frozen=True)
class Version:
    module: str
    head: Callable[[], nn.Module] | str | None
    modulo: int
    ensemble: bool
    sha256: str

    def ifnet(self) -> type[nn.Module]:
        return importlib.import_module(f"nelux_rife.models.{self.module}").IFNet

    def make_head(self) -> nn.Module | None:
        if self.head is None:
            return None
        if self.head == MODULE_HEAD:
            return importlib.import_module(f"nelux_rife.models.{self.module}").Head()
        return self.head()


def _v(module, head, sha256, modulo=32, ensemble=True):
    return Version(f"IFNet_HDv3_v{module}", head, modulo, ensemble, sha256)


VERSIONS: dict[str, Version] = {
    "4.0": _v("4_0", NO_HEAD, "b234ad8d0acfcb6d9eaf2791eb7ddebdaab58bd862b229c703f8073e56d99c29"),
    "4.1": _v("4_1", NO_HEAD, "ae7018cabb9150f1d6e7ee1efc469e1d16440ad44bda868afc408549e272df62"),
    "4.2": _v("4_2", NO_HEAD, "c0aa37679b92a3809fdd16d6d1d50008070cdcb3c3ce7cbeb96e8c90699f5fe3"),
    "4.3": _v("4_3", NO_HEAD, "7f4bdd625488b55b6433e270751fa77619b733b03473489a1a495fe4fa50591a"),
    "4.4": _v("4_4", NO_HEAD, "a051572810b025c35013a35f044a08e6edb516ec1ffa1ac2878a48059f30d5fa"),
    "4.5": _v("4_5", NO_HEAD, "90d4d23aecd533820ad5dcd6420da24d19b046d5f4f76d93a6b7c8a3ed4b072e"),
    "4.6": _v("4_6", NO_HEAD, "008646e761f0e67cb77f0c6c44cfe3c3e5a05d9d9465311b9681ca650ce030db"),
    "4.7": _v("4_7", _head_16_4, "fcf3492b10f17fb035156ea4177ed87b1f517eae54fe4500e878f5d186043d5e"),
    "4.8": _v("4_8", _head_16_4, "7456a34db870a27db22bd010d644f6db4247ed7d3ac03faffd20fb078bb00871"),
    "4.9": _v("4_9", _head_16_4, "ef91580a020abb7ddfbd3a51573dc395cf2c2a9530ff653ef3f8a1fc6845857f"),
    "4.10": _v("4_10", _head_32(8), "056cfec383443884c84bd136ac4c2361981207619e4759fdcda7f2dfd212722a"),
    "4.11": _v("4_11", _head_32(8), "1bddce86425f21e25712ab84c62bcc780f6d558f6d4f856e0b5a8713cb473bce"),
    "4.12": _v("4_12", _head_32(8), "ac21bc88eee93aaf4bc820ad6bcab6c3f718a2596c2c3633d084c10d3d654d6f"),
    "4.12.lite": _v("4_12_lite", _head_32(4), "6644d4e130794a9a238c4ef0e04d4b62bead160998cc9c4c13e9b8e13a95598c"),
    "4.13": _v("4_13", MODULE_HEAD, "dcdae5083fb530891926eb8b762689c0a3ab75c687e48b6a4766623a42a0a58c"),
    "4.13.lite": _v("4_13_lite", _head_32(4), "aaff2d5360d25171d8b6ee3b0ef8407cd6d76232406957c35931600233395bd4"),
    "4.14": _v("4_14", MODULE_HEAD, "582c9c7ddf15eb9de56ae567cac24a47b2cb72853db5189b828a00ffbf183f8e"),
    "4.14.lite": _v("4_14_lite", MODULE_HEAD, "9035381a797cadd5762b13026c2c9b8c5e36f9113eba8e0c575d1cbe31cdac30"),
    "4.15": _v("4_15", MODULE_HEAD, "bf969affb3f6e9a09eea8c1da6c51bb22d6da45e92cafb9273689d05e2360dd2"),
    "4.15.lite": _v("4_15_lite", MODULE_HEAD, "54fe3fb51efb880f64ffe2048a88220ad2b0b3d03f7cb800742e90821dcfb24f"),
    "4.16.lite": _v("4_16_lite", MODULE_HEAD, "290cf343fb5a933f4503920ca07826c4d30561bd9b573b11b054b33ea0dee642"),
    "4.17": _v("4_17", MODULE_HEAD, "b1ee3186270312a38316e4d53c77b31a60062cfa5636e13d6f0a1dd89bb7b128"),
    "4.17.lite": _v("4_17_lite", MODULE_HEAD, "9303f59001bc40e801273828cc0da84b8f7696a9485cd7c45e7796f20ed2f086"),
    "4.18": _v("4_18", MODULE_HEAD, "955fc8b552417ae4808999f5d671d024438e904c0bebf766ae6fb61806ce021c"),
    "4.19": _v("4_19", MODULE_HEAD, "6f336b13f436897e697d07ccf79d3c04c2cee3a8c2cc51dcafe5b19b718bc957"),
    "4.20": _v("4_20", MODULE_HEAD, "f46e2a8b1794ec80c5ce84e6c6359ffa2d15f5ce6bd2b8f484252ff6ffba8e7e"),
    "4.21": _v("4_21", MODULE_HEAD, "56d90eabc24d9f394594a184ebff1becbcb9424ea60581d80444f87e0da7770c", ensemble=False),
    "4.22": _v("4_22", MODULE_HEAD, "549df85a6e5b55c4488864b21d4c9b35f415070bbb62c22c3cb6bf788c3d4af5", ensemble=False),
    "4.22.lite": _v(
        "4_22_lite", MODULE_HEAD, "8f6fb9105ba9e946762ee7190acbca3ca1cf14193eb81ca0955d492fb8558692", ensemble=False
    ),
    "4.23": _v("4_23", MODULE_HEAD, "b84b8515303657088cad027508ee37aa995540177ada714558da3cd24a68819f", ensemble=False),
    "4.24": _v("4_24", MODULE_HEAD, "453a550ff526cf8e9f4b6bb4e7e9f2dcc255694d9cf893a8685d74aabf483569"),
    "4.25": _v(
        "4_25", MODULE_HEAD, "6615790efd627772917205db291f51cd392528a157ecbb2ecaeec3bff8eb6de2", 64, ensemble=False
    ),
    "4.25.lite": _v(
        "4_25_lite", MODULE_HEAD, "81cdba223fe72a120130cc8552e5d2ecac824259d406f0c15323b3decf96b8b1", 128, ensemble=False
    ),
    "4.25.heavy": _v(
        "4_25_heavy", MODULE_HEAD, "5b761e18073f37f018ca498efc9d170b2a0fc81a6448bb95798d7327fcaa16ad", 64, ensemble=False
    ),
    "4.26": _v(
        "4_26", MODULE_HEAD, "45c7f74156704769dc9f85cfcaf8552e1e926f9399dcfa3a553dee88fac6f53f", 64, ensemble=False
    ),
    "4.26.heavy": _v(
        "4_26_heavy", MODULE_HEAD, "4cc518e172156ad6207b9c7a43364f518832d83a4325d484240493a9e2980537", 64, ensemble=False
    ),
}

SCALES = (0.25, 0.5, 1.0, 2.0, 4.0)
PRECISIONS = ("fp32", "fp16")
