from setuptools import setup

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="online_merging",
    version="1.0",
    description="Online Merging Optimizers for Boosting Rewards and Mitigating Tax in Alignment",
    url="",
    author="Keming Lu",
    author_email="lukeming.lkm@alibaba-inc.com",
    license="Apache 2.0",
    packages=["online_merging"],
    install_requires=required,
)