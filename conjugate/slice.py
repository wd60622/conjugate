from dataclasses import asdict


class SliceMixin:
    """Mixin in order to slice the parameters"""

    def __getitem__(self, key):
        params = asdict(self)

        def slice(value, key):
            try:
                return value[key]
            except Exception:
                return value

        new_params = {k: slice(value=v, key=key) for k, v in params.items()}

        return self.__class__(**new_params)
