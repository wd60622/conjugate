from dataclasses import asdict


class SliceMixin:
    """Mixin in order to slice the parameters"""

    @property
    def params(self):
        return asdict(self)

    def __getitem__(self, key):
        def take_slice(value, key):
            try:
                return value[key]
            except Exception:
                return value

        new_params = {k: take_slice(value=v, key=key) for k, v in self.params.items()}

        return self.__class__(**new_params)
