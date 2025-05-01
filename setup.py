from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name="game_of_life",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Conway's Game of Life with efficient matrix operations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/game_of_life",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "game-of-life=game_of_life.main:main",
        ],
    },
)
