# ML project

### Set Up environment
```
virtualenv env
source ./venv/Scripts/activate
```
> To shorten path in terminal: PROMPT_DIRTRIM=1

To install dependencies in package: `python setup.py install`. This method is decrecated.

#### Recomendation
```
pip install .
```
This command assumes that you are in the root directory of the project where the `setup.py` file is located. The `.` specifies the current directory as the source for installation.

### Dataclass
A `dataclass` is a class that is designed to only hold data values. They aren't different from regular classes, but they usually don't have any other methods. They are typically used to store information that will be passed between different parts of a program or a system.

### doctest