"""Constant object class

Python doesn't have default constant type.
We implement in this file.
"""

class Constant(object):
    """Constant object class

    Attributes:
        Attributes are defined when the object is constructed.
        Please refer to Examples.

    Examples:
        ```
        const = Constant(fruit='apple', color='red')
        print(const.fruit) # 'apple'
        const.weight = 100 # First binding is fine
        const.weight = 200
        # Second binding raises 'Can't rebind const(xxx)'
        ```
    """
    def __init__(self, **kwargs): # kwargs = keyword arguments
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __setattr__(self, key, value):
        """
        Performs operation when setting attribure.
        """
        if key in self.__dict__:
            raise TypeError('Can\'t rebind const({key})'.format(key=key))
        self.__dict__[key] = value
