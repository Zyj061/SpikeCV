from SpikeCV.spkProc.augment.augments import Assemble

from SpikeCV.spkProc.augment.augments import RandomHorizontalFlip
from SpikeCV.spkProc.augment.augments import RandomVerticalFlip

from SpikeCV.spkProc.augment.augments import Resize
from SpikeCV.spkProc.augment.augments import RandomResize

from SpikeCV.spkProc.augment.augments import SpikeQuant

from SpikeCV.spkProc.augment.augments import CenterCrop
from SpikeCV.spkProc.augment.augments import RandomCrop
from SpikeCV.spkProc.augment.augments import RandomResizedCrop

from SpikeCV.spkProc.augment.augments import SpatialPad
from SpikeCV.spkProc.augment.augments import TemporalPad
from SpikeCV.spkProc.augment.augments import RandomRotation
from SpikeCV.spkProc.augment.augments import RandomAffine

from SpikeCV.spkProc.augment.augments import RandomBlockErasing
from SpikeCV.spkProc.augment.augments import RandomSpikeErasing
from SpikeCV.spkProc.augment.augments import RandomSpikeAdding

__all__ = ["Assemble", "RandomHorizontalFlip", "RandomVerticalFlip", "Resize", "SpikeQuant", "RandomResize",
    "CenterCrop", "RandomCrop", "RandomResizedCrop",
    "SpatialPad", "TemporalPad",
    "RandomRotation", "RandomAffine",
    "RandomBlockErasing", "RandomSpikeErasing", "RandomSpikeAdding"
    # TODO "Normalize"
    ]