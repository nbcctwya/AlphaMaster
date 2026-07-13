"""Custom data handlers for running the MASTER model on US (SP500) data.

Migrated without changing the feature substitution from
``qlib-MASTER/examples/benchmarks/MASTER/paper_baseline/us_handlers.py``.

Background
----------
MASTER expects ``d_feat=158`` (the Alpha158 factor set) plus 63 market-guided features.
On ``us_data`` there is **no ``$vwap`` bin**, so Alpha158's ``VWAP0`` feature
(``$vwap/$close``) is all-NaN on every US stock. Dropping the feature would shrink
``d_feat`` and break the model's ``gate_input_start_index=158`` assumption.

``Alpha158US`` swaps only the ``$vwap``-based feature for a typical-price proxy, so the
158-feature count and the MASTER architecture are preserved. The CSI300 run keeps using
the original ``Alpha158`` unchanged.
"""
from qlib.contrib.data.handler import Alpha158


class Alpha158US(Alpha158):
    """Alpha158 for US data: replace the ``$vwap`` feature with a typical-price proxy.

    Replacement (only for fields referencing ``$vwap``)::

        $vwap/$close  ->  ($high+$low+$close)/3/$close
    """

    # (high+low+close)/3 is the standard typical-price; normalized by close like VWAP0.
    _VWAP_PROXY = "($high+$low+$close)/3/$close"

    def get_feature_config(self):
        fields, names = super().get_feature_config()
        fields = [self._VWAP_PROXY if "$vwap" in str(f) else f for f in fields]
        return fields, names
