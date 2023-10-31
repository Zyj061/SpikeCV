from setuptools import find_packages
from setuptools import setup

with open("./requirements.txt", "r", encoding="utf-8") as fh:
    install_requires = fh.read()

with open("./README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='SpikeCV',
    version='0.1a',
    packages=['utils', 'metrics', 'spkData', 'spkProc', 'spkProc.filters', 'spkProc.tracking', 'spkProc.detection',
              'spkProc.augment', 'spkProc.recognition', 'spkProc.optical_flow', 'spkProc.optical_flow.SCFlow', 'spkProc.reconstruction',
              'spkProc.reconstruction.SSML_Recon', 'spkProc.depth_estimation', 'examples', 'visualization'],
    url="https://git.openi.org.cn/Cordium/SpikeCV.git",
    author='PKU, and other contributors',
    author_email='yj.zheng@pku.edu.cn, ...',
    description="An open-source framework for Spiking computer vision, which including datasets, algorithm libraries and software."
)
