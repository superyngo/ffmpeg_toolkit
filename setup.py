from setuptools import setup, find_packages

setup(
    name="ffmpeg_toolkit",
    version="0.1.3",
    author="Wenyang Tai",
    author_email="superyngo@gmail.com",
    description="A toolkit for working with FFmpeg",
    long_description=open("README.MD").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/superyngo/ffmpeg_toolkit",
    packages=find_packages(where="src", exclude=["bin", "bin.*"]),
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        # "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    license="MIT",
    python_requires=">=3.9",
    install_requires=[
        "pydantic>=1.10.0",
    ],
)
