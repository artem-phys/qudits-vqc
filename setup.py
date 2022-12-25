import setuptools

with open("README.md") as file:
    read_me_description = file.read()

setuptools.setup(
    name="qudits_vqc",
    version="1.0.0",
    py_modules=['rpa']
    author="Artem Kuzmichev",
    author_email="artemkuzmichev.official@gmail.com",
    description="cirq extension with qudits",
    long_description=read_me_description,
    long_description_content_type="text/markdown",
    url="https://github.com/artem-phys/qudits-vqc",
    packages=setuptools.find_packages(where="src"),
    package_dir={'': 'src'},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5',
    install_requires=[
          'cirq',
      ],
)
