from agogos.transformer import Transformer


class ExampleTransformationBlock(Transformer):
    """This is an example transformation block, there are no init blocks in the base Transformer class.
    The transform method should be overridden in the child class."""

    def transform(self, x: int) -> int:
        """Transform the input data.

        :param x: The input data.
        :return: The transformed data."""
        print("Transforming the input data")
        return x * 2
    
    # More examples of how to use the Transformer class

    # Example of class with one argument
    """
    def transform(self, x: int, arg_1: int) -> int:
        return x * arg_1
    """

    # Example of class with multiple arguments
    """    
    def transform(self, x: int, arg_1: int, arg_2: int) -> int:
        return x * arg_1 * arg_2
    """
        
    # Example of class with new argument
    """    
    class X(Transformer):
        new_arg: int

        def transform(self, x: int) -> int:
            return x * self.new_arg
    """