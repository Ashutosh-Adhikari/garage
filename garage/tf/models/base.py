"""This file contains the abstraction class for models."""
from garage.tf.models import AutoPickable


class BaseModel:
    """Abstraction class for models."""

    def _build_model(self, input_var):
        """
        Build model.

        This function will build the whole graph for
        the model. All tensors should be created here.
        By calling this function, a copy of the same graph
        should be created with the same parameters.
        """
        raise NotImplementedError

    def __call__(self, inputs):
        return self.model(inputs)

    @property
    def input(self):
        """Tensor input of the Model."""
        return self.model.input

    @property
    def output(self):
        """Tensor output of the Model."""
        return self.model.output

    @property
    def inputs(self):
        """Tensor inputs of the Model."""
        return self.model.inputs

    @property
    def outputs(self):
        """Tensor outputs of the Model."""
        return self.model.outputs


class PickableModel(AutoPickable):
    """Abstraction class for autopickable models."""

    def _build_model(self, input_var):
        """
        Build model.

        This function will build the whole graph for
        the model. All tensors should be created here.
        By calling this function, a copy of the same graph
        should be created with the same parameters.
        """
        raise NotImplementedError

    def __call__(self, inputs):
        return self.model(inputs)

    @property
    def input(self):
        """Tensor input of the Model."""
        return self.model.input

    @property
    def output(self):
        """Tensor output of the Model."""
        return self.model.output

    @property
    def inputs(self):
        """Tensor inputs of the Model."""
        return self.model.inputs

    @property
    def outputs(self):
        """Tensor outputs of the Model."""
        return self.model.outputs
