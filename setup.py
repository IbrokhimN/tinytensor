from setuptools import setup, find_packages

setup(
    name="tinytensor",
    version="0.1.0",
    description="мини ИИ фреймворк от IbrokimN ( github/IbrokhimN )",
    packages=find_packages(include=["tinytensor", "tinytensor.*"]),
    install_requires=[
        "numpy>=1.23",
    ],
    extras_require={
        "dev": ["pytest>=7.0"],
    },
    python_requires=">=3.9",
)
