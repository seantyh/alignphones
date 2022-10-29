from dataclasses import dataclass
from typing import List, Optional, Union
from enum import Enum, auto

class AlignNodeType(Enum):
    Root = auto()
    Character = auto()
    EpiPhone = auto()
    AlloPhone = auto()

@dataclass
class AlignNode:
    nodetype: AlignNodeType
    label: str    
    children: List["AlignNode"]
    frame_start: Optional[float] = None
    frame_end: Optional[float] = None

    def __iadd__(self, other: "AlignNode"):
        if not isinstance(other, AlignNode):
            raise TypeError("AlignNode expected")
        if self == other:
            raise ValueError("Add to itself")
        self.children.extend(other.children)
        return self

    @property
    def start(self) -> Union[float, None]:
        if not self.frame_start:       
            starts = [x.start
                      for x in self.children
                      if x.start]
            return starts[0] if starts else None
        else:
            return self.frame_start
    
    @property
    def end(self) -> Union[float, None]:
        if not self.frame_end:
            ends = [x.end
                      for x in self.children
                      if x.end]
            return ends[0] if ends else None
        else:
            return self.frame_end        
        

def create_root() -> AlignNode:
    root = AlignNode(
        AlignNodeType.Root, 
        "root",
        [])
    return root
