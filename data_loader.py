import json
import os
import random
from pathlib import Path
from typing import Callable, Optional

import torch
import torchvision
import torchvision.transforms as transforms

torch.manual_seed(27)
random.seed(27)


MNIST_CLASSES = {
    0: "airplane",
    1: "car",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck",
}

TINY_CLASSES = {
    0: "goldfish, Carassius auratus",
    1: "European fire salamander, Salamandra salamandra",
    2: "bullfrog, Rana catesbeiana",
    3: "tailed frog, bell toad, ribbed toad, tailed toad, Ascaphus trui",
    4: "American alligator, Alligator mississipiensis",
    5: "boa constrictor, Constrictor constrictor",
    6: "trilobite",
    7: "scorpion",
    8: "black widow, Latrodectus mactans",
    9: "tarantula",
    10: "centipede",
    11: "goose",
    12: "koala, koala bear, kangaroo bear, native bear, Phascolarctos cinereus",
    13: "jellyfish",
    14: "brain coral",
    15: "snail",
    16: "slug",
    17: "sea slug, nudibranch",
    18: "American lobster, Northern lobster, Maine lobster, Homarus americanus",
    19: "spiny lobster, langouste, rock lobster, crawfish, crayfish, sea crawfish",
    20: "black stork, Ciconia nigra",
    21: "king penguin, Aptenodytes patagonica",
    22: "albatross, mollymawk",
    23: "dugong, Dugong dugon",
    24: "Chihuahua",
    25: "Yorkshire terrier",
    26: "golden retriever",
    27: "Labrador retriever",
    28: "German shepherd, German shepherd dog, German police dog, alsatian",
    29: "standard poodle",
    30: "tabby, tabby cat",
    31: "Persian cat",
    32: "Egyptian cat",
    33: "cougar, puma, catamount, mountain lion, painter, panther, Felis concolor",
    34: "lion, king of beasts, Panthera leo",
    35: "brown bear, bruin, Ursus arctos",
    36: "ladybug, ladybeetle, lady beetle, ladybird, ladybird beetle",
    37: "fly",
    38: "bee",
    39: "grasshopper, hopper",
    40: "walking stick, walkingstick, stick insect",
    41: "cockroach, roach",
    42: "mantis, mantid",
    43: "dragonfly, darning needle, devil's darning needle, sewing needle, snake feeder, snake doctor, mosquito hawk, skeeter hawk",
    44: "monarch, monarch butterfly, milkweed butterfly, Danaus plexippus",
    45: "sulphur butterfly, sulfur butterfly",
    46: "sea cucumber, holothurian",
    47: "guinea pig, Cavia cobaya",
    48: "hog, pig, grunter, squealer, Sus scrofa",
    49: "ox",
    50: "bison",
    51: "bighorn, bighorn sheep, cimarron, Rocky Mountain bighorn, Rocky Mountain sheep, Ovis canadensis",
    52: "gazelle",
    53: "Arabian camel, dromedary, Camelus dromedarius",
    54: "orangutan, orang, orangutang, Pongo pygmaeus",
    55: "chimpanzee, chimp, Pan troglodytes",
    56: "baboon",
    57: "African elephant, Loxodonta africana",
    58: "lesser panda, red panda, panda, bear cat, cat bear, Ailurus fulgens",
    59: "abacus",
    60: "academic gown, academic robe, judge's robe",
    61: "altar",
    62: "apron",
    63: "backpack, back pack, knapsack, packsack, rucksack, haversack",
    64: "bannister, banister, balustrade, balusters, handrail",
    65: "barbershop",
    66: "barn",
    67: "barrel, cask",
    68: "basketball",
    69: "bathtub, bathing tub, bath, tub",
    70: "beach wagon, station wagon, wagon, estate car, beach waggon, station waggon, waggon",
    71: "beacon, lighthouse, beacon light, pharos",
    72: "beaker",
    73: "beer bottle",
    74: "bikini, two-piece",
    75: "binoculars, field glasses, opera glasses",
    76: "birdhouse",
    77: "bow tie, bow-tie, bowtie",
    78: "brass, memorial tablet, plaque",
    79: "broom",
    80: "bucket, pail",
    81: "bullet train, bullet",
    82: "butcher shop, meat market",
    83: "candle, taper, wax light",
    84: "cannon",
    85: "cardigan",
    86: "cash machine, cash dispenser, automated teller machine, automatic teller machine, automated teller, automatic teller, ATM",
    87: "CD player",
    88: "chain",
    89: "chest",
    90: "Christmas stocking",
    91: "cliff dwelling",
    92: "computer keyboard, keypad",
    93: "confectionery, confectionary, candy store",
    94: "convertible",
    95: "crane",
    96: "dam, dike, dyke",
    97: "desk",
    98: "dining table, board",
    99: "drumstick",
    100: "dumbbell",
    101: "flagpole, flagstaff",
    102: "fountain",
    103: "freight car",
    104: "frying pan, frypan, skillet",
    105: "fur coat",
    106: "gasmask, respirator, gas helmet",
    107: "go-kart",
    108: "gondola",
    109: "hourglass",
    110: "iPod",
    111: "jinrikisha, ricksha, rickshaw",
    112: "kimono",
    113: "lampshade, lamp shade",
    114: "lawn mower, mower",
    115: "lifeboat",
    116: "limousine, limo",
    117: "magnetic compass",
    118: "maypole",
    119: "military uniform",
    120: "miniskirt, mini",
    121: "moving van",
    122: "nail",
    123: "neck brace",
    124: "obelisk",
    125: "oboe, hautboy, hautbois",
    126: "organ, pipe organ",
    127: "parking meter",
    128: "pay-phone, pay-station",
    129: "picket fence, paling",
    130: "pill bottle",
    131: "plunger, plumber's helper",
    132: "pole",
    133: "police van, police wagon, paddy wagon, patrol wagon, wagon, black Maria",
    134: "poncho",
    135: "pop bottle, soda bottle",
    136: "potter's wheel",
    137: "projectile, missile",
    138: "punching bag, punch bag, punching ball, punchball",
    139: "reel",
    140: "refrigerator, icebox",
    141: "remote control, remote",
    142: "rocking chair, rocker",
    143: "rugby ball",
    144: "sandal",
    145: "school bus",
    146: "scoreboard",
    147: "sewing machine",
    148: "snorkel",
    149: "sock",
    150: "sombrero",
    151: "space heater",
    152: "spider web, spider's web",
    153: "sports car, sport car",
    154: "steel arch bridge",
    155: "stopwatch, stop watch",
    156: "sunglasses, dark glasses, shades",
    157: "suspension bridge",
    158: "swimming trunks, bathing trunks",
    159: "syringe",
    160: "teapot",
    161: "teddy, teddy bear",
    162: "thatch, thatched roof",
    163: "torch",
    164: "tractor",
    165: "triumphal arch",
    166: "trolleybus, trolley coach, trackless trolley",
    167: "turnstile",
    168: "umbrella",
    169: "vestment",
    170: "viaduct",
    171: "volleyball",
    172: "water jug",
    173: "water tower",
    174: "wok",
    175: "wooden spoon",
    176: "comic book",
    177: "plate",
    178: "guacamole",
    179: "ice cream, icecream",
    180: "ice lolly, lolly, lollipop, popsicle",
    181: "pretzel",
    182: "mashed potato",
    183: "cauliflower",
    184: "bell pepper",
    185: "mushroom",
    186: "orange",
    187: "lemon",
    188: "banana",
    189: "pomegranate",
    190: "meat loaf, meatloaf",
    191: "pizza, pizza pie",
    192: "potpie",
    193: "espresso",
    194: "alp",
    195: "cliff, drop, drop-off",
    196: "coral reef",
    197: "lakeside, lakeshore",
    198: "seashore, coast, seacoast, sea-coast",
    199: "acorn",
}


def load_mnist_data(path: str):
    transform = torchvision.transforms.Compose(
        [
            transforms.Resize((28, 28)),
            torchvision.transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    mnist_data = torchvision.datasets.MNIST(
        root=path,
        download=True,
        transform=transform,
    )

    train_set_size = int(len(mnist_data) * 0.8)
    valid_set_size = int(len(mnist_data) * 0.2)

    train_dataset, test_dataset, _ = torch.utils.data.random_split(
        mnist_data,
        [
            train_set_size,
            valid_set_size,
            len(mnist_data) - train_set_size - valid_set_size,
        ],
        generator=torch.Generator().manual_seed(42),
    )

    return train_dataset, test_dataset


def load_cifar_data(path: str):
    train_dataset = torchvision.datasets.CIFAR10(
        root=path,
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        ),
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root=path,
        train=False,
        download=False,
        transform=transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        ),
    )

    # wrap the dataset in another dataset class for compatability with other dataset objects
    train_dataset = torch.utils.data.Subset(
        train_dataset, list(range(len(train_dataset)))
    )
    test_dataset = torch.utils.data.Subset(test_dataset, list(range(len(test_dataset))))

    return train_dataset, test_dataset


def load_image_net_data(
    path: str,
    subset: Optional[str] = None,
    add_subset: Optional[str] = None,
    pc: Optional[float] = 1.0,
    add_pc: Optional[float] = 1.0,
    extra_folders: Optional[list] = None,
    test_transforms: Optional[Callable] = None,
):
    data_transforms = transforms.Compose(
        [
            transforms.RandomResizedCrop((224, 224)),
            transforms.RandomRotation(20),
            transforms.RandomHorizontalFlip(0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    if test_transforms is None:
        test_transforms = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    train_dataset = torchvision.datasets.ImageFolder(
        os.path.join(path, "train"), transform=data_transforms
    )

    test_dataset = torchvision.datasets.ImageFolder(
        os.path.join(path, "val"), transform=test_transforms
    )

    train_datasets = []
    test_datasets = []

    # Load additional datasets from extra_folders if provided
    if extra_folders:
        train_datasets.append(
            torchvision.datasets.ImageFolder(extra_folders, transform=data_transforms)
        )

    if (subset is None) and (add_subset is None):
        train_dataset, _ = torch.utils.data.random_split(
            train_dataset,
            [
                int(len(train_dataset) * pc),
                len(train_dataset) - int(len(train_dataset) * pc),
            ],
            generator=torch.Generator().manual_seed(42),
        )
        test_dataset = torch.utils.data.Subset(
            test_dataset, list(range(len(test_dataset)))
        )
    else:
        if add_subset:
            add = True
            subset = add_subset
        else:
            add = False
        # load the subset of the dataset
        subset_path = Path(subset)
        with open(subset_path, "r") as f:
            subset_classes = json.load(f)
        subset_classes = {int(k): v for k, v in subset_classes.items()}
        subset_idx = {
            i
            for i in range(len(train_dataset))
            if train_dataset.targets[i] in subset_classes
        }
        if add:
            all_indices = set(range(len(train_dataset)))
            outside_indices = list(all_indices - subset_idx)

            subset_idx = list(subset_idx)
            random.shuffle(subset_idx)
            subset_idx = list(subset_idx)[: int(len(subset_idx) * add_pc)]

            # shuffle subset_class_idx
            random.shuffle(outside_indices)
            subset_idx = outside_indices[: int(len(outside_indices) * pc)] + list(
                subset_idx
            )
        else:
            # shuffle subset_class_idx
            subset_idx = list(subset_idx)
            random.shuffle(subset_idx)
            subset_idx = list(subset_idx)[: int(len(subset_idx) * pc)]

        train_dataset = torch.utils.data.Subset(
            train_dataset,
            list(subset_idx),
        )

        subset_test = list(range(len(test_dataset)))

        test_dataset = torch.utils.data.Subset(test_dataset, subset_test)

    train_datasets.append(train_dataset)
    test_datasets.append(test_dataset)

    if len(train_datasets) > 1:
        train_dataset = torch.utils.data.ConcatDataset(train_datasets)
    if len(test_datasets) > 1:
        test_dataset = torch.utils.data.ConcatDataset(test_datasets)

    return train_dataset, test_dataset


def load_tiny_image_net_data(path: str):
    data_transforms = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(64, 4),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    test_transforms = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    train_dataset = torchvision.datasets.ImageFolder(
        os.path.join(path, "train"), transform=data_transforms
    )
    test_dataset = torchvision.datasets.ImageFolder(
        os.path.join(path, "train"), transform=test_transforms
    )

    train_set_size = int(len(train_dataset) * 0.8)

    indices = torch.randperm(
        len(train_dataset), generator=torch.Generator().manual_seed(42)
    ).tolist()

    train_indices = indices[:train_set_size]
    test_indices = indices[train_set_size:]

    train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
    test_dataset = torch.utils.data.Subset(test_dataset, test_indices)
    return train_dataset, test_dataset
