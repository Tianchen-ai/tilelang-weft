"""CPU loop requests retain their scope for the shared physical scheduler."""

from tilelang.language.loop import Pipelined as common_pipelined


def Pipelined(start, stop=None, num_stages=0, order=None, stage=None, annotations=None):
    if type(num_stages) is not int or num_stages not in (0, 1, 2, 3, 4):
        raise ValueError("Weft T.Pipelined num_stages must be one of 0, 1, 2, 3, 4")
    if order or stage:
        raise NotImplementedError("manual T.Pipelined order/stage requires a closed source-operation schedule")
    attributes = dict(annotations or {})
    if "weft.pipeline.depth" in attributes:
        raise ValueError("T.Pipelined num_stages already specifies the loop's physical depth")
    attributes["weft.pipeline.depth"] = num_stages
    return common_pipelined(start, stop, num_stages=num_stages, annotations=attributes)
