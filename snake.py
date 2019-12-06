from __future__ import annotations

import sys
MIN_PYTHON = (3, 7) # requires at least python 3.7
assert sys.version_info >= MIN_PYTHON, f"requires Python {'.'.join([str(n) for n in MIN_PYTHON])} or newer"

import random
import queue
import functools
import itertools
import heapq
from tqdm.auto import tqdm
from enum import IntEnum, auto
from dataclasses import dataclass, field
import numpy as np

from scipy.ndimage.morphology import black_tophat
from scipy.ndimage import label
from skimage.morphology import flood_fill

from typing import List, Tuple, Iterable, Optional, Iterator
from PIL import Image
import uuid

# tile assets for drawing snake graphics
tile_names = {
    'd': Image.open('tiles/d.png'),
    'dd': Image.open('tiles/dd.png'),
    'dl': Image.open('tiles/dl.png'),
    'dr': Image.open('tiles/dr.png'),
    'e': Image.open('tiles/e.png'),
    'f': Image.open('tiles/f.png'),
    'h': Image.open('tiles/h.png'),
    'hd': Image.open('tiles/hd.png'),
    'hl': Image.open('tiles/hl.png'),
    'hr': Image.open('tiles/hr.png'),
    'hu': Image.open('tiles/hu.png'),
    'l': Image.open('tiles/l.png'),
    'ld': Image.open('tiles/ld.png'),
    'll': Image.open('tiles/ll.png'),
    'lu': Image.open('tiles/lu.png'),
    'r': Image.open('tiles/r.png'),
    'rd': Image.open('tiles/rd.png'),
    'rr': Image.open('tiles/rr.png'),
    'ru': Image.open('tiles/ru.png'),
    'u': Image.open('tiles/u.png'),
    'ul': Image.open('tiles/ul.png'),
    'ur': Image.open('tiles/ur.png'),
    'uu': Image.open('tiles/uu.png'),
    'x': Image.open('tiles/x.png'),
}
TH = tile_names['e'].height
TW = tile_names['e'].width

class Cell(IntEnum):
    E = 0 # empty
    F = auto() # fruit
    
    H = auto() # head
    
    # arrows pointing from tail to head:
    U = auto() # up
    D = auto() # down
    L = auto() # left
    R = auto() # right

    @property
    def vec(self):
        return {
            Cell.U: np.array([-1,  0]),
            Cell.D: np.array([ 1,  0]),
            Cell.L: np.array([ 0, -1]),
            Cell.R: np.array([ 0,  1]),
        }.get(self, np.array([ 0,  0]))
    
    @property
    def tile(self):
        return {
            Cell.E: "e",
            Cell.F: "f",
            Cell.U: "u",
            Cell.D: "d",
            Cell.L: "l",
            Cell.R: "r",
            Cell.H: "h",
        }.get(self, None)
    
    def __str__(self):
        return {
            Cell.E: " ",
            Cell.F: "@",
            Cell.U: "^",
            Cell.D: "v",
            Cell.L: "<",
            Cell.R: ">",
            Cell.H: "o",
        }.get(self, None)


class Game:
    # game parameters
    HEIGHT = 10
    WIDTH = 15
    LEN_PER_FRUIT = 3
    @classmethod
    def init_grid(cls) -> np.ndarray:
        g = np.full((cls.HEIGHT, cls.WIDTH), Cell.E)
        g[(0,0)] = Cell.H
        return g

class SnakeException(Exception):
    def __init__(self, snake:Snake, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.snake = snake
class Win(SnakeException):
    pass
class Lose(SnakeException):
    pass

@dataclass(eq=False, order=False, frozen=False)
@functools.total_ordering
class Snake:
    
    # authoritative parts of the system's state:
    grid: np.ndarray = field(default_factory=Game.init_grid) # shape=(HEIGHT, WIDTH), dtype=int
    rng_state: tuple = random.Random().getstate()
    grow: int = 2 # number of segments pending growth
    trace: List[Cell] = field(default_factory=list) # record of moves made
    
    # stored for quick referencing:
    head: Tuple[int] = (0,0)
    @property
    def nhead(self):
        return np.array(self.head)
    @nhead.setter
    def nhead(self, head):
        self.head = tuple(head)
    
    tail: Tuple[int] = (0,0) #np.ndarray = np.zeros(2, np.int32)
    @property
    def ntail(self):
        return np.array(self.tail)
    @ntail.setter
    def ntail(self, tail):
        self.tail = tuple(tail)

    fruit: Optional[Tuple[int]] = None
    @property
    def nfruit(self):
        return np.array(self.fruit) if self.fruit is not None else None
    @nfruit.setter
    def nfruit(self, fruit):
        self.fruit = tuple(fruit) if fruit is not None else None

    size: int = 1 # total segments not counting pending growth
    lose: bool = False
    win: bool = False
    
    def move(self, move: Cell, inplace:bool=False, exc_win:bool=False, exc_lose:bool=False) -> Snake:
        # accepts a Cell.U/D/L/R, returns the snake after the move
        if inplace:
            new = self
        else:
            new = self.copy()
        
        # add move to trace
        new.trace.append(move)
        
        # replace current head with movement and advance head pointer
        new.grid[new.head] = move
        new.nhead += move.vec
        
        # bounds check, if the new head is outside set a flag
        bound = True
        if not ((0 <= new.nhead).all() and (new.nhead < (Game.HEIGHT, Game.WIDTH)).all()):
            bound = False
        
        # increment growth if we're eating a fruit - removing/placing a new fruit happens later
        if bound and new.grid[new.head] == Cell.F:
            new.grow += Game.LEN_PER_FRUIT
        
        # retract tail or decrement remaining growth
        if new.grow > 0:
            new.grow -= 1
            new.size += 1
        else:
            tail_cell = Cell(new.grid[new.tail])
            new.grid[new.tail] = Cell.E
            new.ntail += tail_cell.vec
        
        # attempt to place new head
        if not bound:
            new.lose = True
            if exc_lose:
                raise Lose(new)
        elif new.grid[new.head] == Cell.F:
            new.grid[new.head] = Cell.H
            new.place_fruit(inplace=True)
        elif new.grid[new.head] == Cell.E:
            new.grid[new.head] = Cell.H
        else: #it's a UDLR segment - or possibly another H, but same result
            new.lose = True
            if exc_lose:
                raise Lose(new)
        
        if new.size >= Game.HEIGHT * Game.WIDTH:
            new.win = True
            if exc_win:
                raise Win(new)
            
        return new
    
    # play a sequence of moves onto this snake
    def replay(self, moves: Iterable[Cell]) -> Snake:
        new = self
        for m in moves:
            new = new.move(m)
        return new
    
    def place_fruit(self, inplace:bool=False, exc_win:bool=False) -> Snake:
        if inplace:
            new = self
        else:
            new = self.copy()
        
        # list of all coordinate pairs in the grid
        preference = list(np.ndindex(new.grid.shape))
        
        # extract a value from and advance the random state
        rng = random.Random(0)
        rng.setstate(new.rng_state)
        seed = rng.getrandbits(64)
        new.rng_state = rng.getstate()
        # use it to seed and shuffle the order to check fruits
        rng.seed(seed)
        rng.shuffle(preference)
        
        # get first empty cell in list
        new.tfruit = next(filter(lambda f:new.grid[f] == Cell.E, preference), None)

        # if there is none, we've won
        if new.tfruit is None:
            new.nfruit = None
            new.win = True
            if exc_win:
                raise Win(new)
        else:
            new.nfruit = np.array(new.tfruit)
            new.grid[new.tfruit] = Cell.F
        
        return new
    
    @functools.cached_property # can be pricy, so cache this
    def heuristic(self):
        return sum(self.heuristic_breakdown().values())
    
    def heuristic_breakdown(self):
        scores = {}
        if self.lose:
            scores['lose'] =+ 999999999 # huge penalty for losing, of course
        elif self.win:
            scores['win'] =- 999999999 # huge bonus if the given snake is a winner
            scores['age'] =+ len(self.trace) # tiny penalty for taking a long time
        else:
            scores['size'] =- 4000 * self.size # large bonus for having a longer snake = closer to winning
            scores['grow'] =- 2000 * self.grow # slightly smaller bonus for pending growth

            # distance is as fraction of possible distance, so larger grids aren't penalized aggressively
            scores['dist'] =+ 5000 * (sum(abs(self.nfruit - self.nhead)) if self.fruit else 0) // (Game.WIDTH + Game.HEIGHT)
            
            scores['age'] =+ len(self.trace) # tiny penalty for taking a long time
            
            if True: # penalize for number of canyons
                canyons = (self.grid > Cell.H).astype(int) # get empty spaces including head as zeroes
                canyons = (black_tophat(canyons, structure=[[1,1],[1,1]], mode='constant', cval=1)==1).astype(int) # black_tophat == 1 to isolate canyons
                num_canyons = label(canyons, output=canyons) # return value of inplace label is number of labels
                scores['canyon'] =+ 150 * num_canyons
                
            if True: # penalize for empty cells in regions disconnected from the head
                discon = (self.grid <= Cell.H).astype(int) # get empty spaces including head as ones
                flood_fill(discon, self.head, 0, connectivity=1, inplace=True) # flood fill empty space from head location
                num_discon = discon.sum() # anywhere that's still a one is disconnected
                scores['discon'] =+ 15 * num_discon

            if True: # penalize for number of disconnected empty segments
                segment = (self.grid <= Cell.F).astype(int) # get empty spaces excluding head as ones
                num_segment = label(segment, output=segment)
                scores['segment'] =+ 150 * num_segment
            
        return scores

    # these to are expensive and immutable, but we usually only use each on once so it's not worth caching
    #@functools.cached_property
    @property
    def u(self) -> Snake:
        return self.move(Cell.U)
    
    #@functools.cached_property
    @property
    def d(self) -> Snake:
        return self.move(Cell.D)
    
    #@functools.cached_property
    @property
    def l(self) -> Snake:
        return self.move(Cell.L)
    
    #@functools.cached_property
    @property
    def r(self) -> Snake:
        return self.move(Cell.R)
    
    @property
    def moves(self) -> List[Snake]:
        return [self.u, self.d, self.l, self.r]
    
    def __lt__(self, other:Snake) -> bool:
        # check for strict superiority before resorting to heuristic:

        # if two snakes are identical except for their trace, the younger one is _always_ better
        # (prevents duplicious squiggling for no reason)
        if ((self.grid == other.grid).all()
            and self.rng_state == other.rng_state
            and self.grow == other.grow):
            return len(self.trace) < len(other.trace)
        
        return self.heuristic < other.heuristic

    def __eq__(self, other:Snake) -> bool:
        return ((self.grid == other.grid).all()
            and self.rng_state == other.rng_state
            and self.grow == other.grow
            and self.trace == other.trace)
    
    def copy(self) -> Snake:
        return Snake(
            grid=self.grid.copy(),
            rng_state=self.rng_state,
            grow=self.grow,
            trace=self.trace.copy(),
            head=self.head,
            tail=self.tail,
            fruit=self.fruit,
            size=self.size,
            lose=self.lose,
            win=self.win,
        )
    
    ### graphical representations of a snake object

    def __str__(self) -> str:
        chars = ['+' + '-'*WIDTH + '+']
        for row in self.grid:
            rc = []
            for cell in row:
                rc.append(str(Cell(cell)))
            chars.append('|' + ''.join(rc) + '|')
        chars.append(chars[0])
        return  '\n'.join(chars)
    
    def as_image(self) -> Image:
        # first character of tile name is the letter corresponding to the cell itself
        first_char = np.vectorize(lambda c: Cell(c).tile)(self.grid).astype(object)

        # get matrix of offset values
        off_x = np.vectorize(lambda c: Cell(c).vec[0])(self.grid)
        off_y = np.vectorize(lambda c: Cell(c).vec[1])(self.grid)
        off = np.stack([off_x, off_y], axis=-1)

        # second character of tile name is the letter of any cell pointing into this cell (there should only be one)
        char = first_char.copy()
        for i in np.ndindex(off.shape[:-1]):
            if not (off[i] == (0,0)).all():
                char[tuple(i + off[i])] += first_char[i]

        # convert matrix of letter pairs into matrix of image objects
        tiles = np.vectorize(lambda n: tile_names.get(n,tile_names['x']), otypes=[object])(char).astype(object)

        # create full-size image of the full grid, and paste all the tiles into the correct locations
        new_im = Image.new('RGB', (Game.WIDTH * TW, Game.HEIGHT * TH), color=0)
        for i, img in np.ndenumerate(tiles):
            coord = (i[1] * TW, i[0] * TH)
            new_im.paste(img, coord, img)

        return new_im
    
    def _repr_png_(self) -> bytes:
        return self.as_image()._repr_png_()


class IMG_HTML(str):
    def _repr_html_(self):
        return f'<img src="{self}">'

class Search:
    def __init__(self, max_backtrack:int=2**20, n_best:int=16):
        self.initial = Snake().place_fruit()
        self.q = [self.initial]
        self.best = []
        self.max_backtrack = max_backtrack
        self.n_best = n_best
        
    def search_one(self) -> None:
        snake = heapq.heappop(self.q)
        moves = snake.moves
        random.shuffle(moves)
        for s in moves:
            if s.lose:
                continue
            heapq.heappush(self.q, s)

        del self.q[self.max_backtrack:] # limit backtracking to best 1 million entries

        self.best.append(snake)
        self.best = heapq.nsmallest(self.n_best, self.best)
        
    def search_n(self, n:int, progress:bool=True) -> None:
        it = range(n)
        if progress:
            it = tqdm(it, desc="searching")
        for i in it:
            self.search_one()
    
    def get_best(self, tag:bool=True, searchlist:bool=True) -> Snake:
        return next(self.dump_best(tag=tag,searchlist=searchlist))

    def dump_best(self, tag:bool=True, searchlist:bool=True) -> Iterable[Snake]:
        ranked_b = Search._dump_heap(self.best, tag="best" if tag else None)
        if searchlist:
            ranked_s = Search._dump_heap(self.q, tag="search" if tag else None)
        else:
            ranked_s = []
        
        ranked = heapq.merge(ranked_b, ranked_s)
        return ranked

    @classmethod
    def _dump_heap(cls, h, tag=None) -> Iterator[Snake]:
        dump = h.copy()
        while True:
            s = heapq.heappop(dump)
            if tag is not None:
                s.tag = tag
            yield s
            if len(dump) == 0: break
    
    def replay(self, snake:Optional[Snake]=None, trace:Optional[Iterable[Cell]]=None) -> Iterator[Snake]:
        # note, assumes the snake in question is a child of the root initial snake
        if trace is None:
            trace = snake.trace
        s = self.initial.copy()
        yield s
        for move in trace:
            s = s.move(move)
            yield s
    
    def report(self) -> Report:
        return Report(self)

class Report:
    def __init__(self, search:SnakeSearch, snake:Optional[Snake]=None):
        self.search = search
        self.snake = search.get_best() if snake is None else snake
    
    def save_png(self, filename=f'pngs/{str(uuid.uuid4())}.png', **kwargs):
        self.snake.as_image().save(filename, **kwargs)
        return IMG_HTML(filename)

    def save_gif(self, filename=f'gifs/{str(uuid.uuid4())}.gif', **kwargs):
        snakes = list(self.search.replay(snake=self.snake))
        im = [s.as_image() for s in tqdm(snakes, desc="rendering")]
        im[0].save(filename, save_all=True, append_images=([im[0]]*3+im+[im[-1]]*4), **kwargs)
        return IMG_HTML(filename)

    def _repr_html_(self):
        return f'''
            {self.save_png(loop=0)._repr_html_()}
            {self.save_gif(loop=0)._repr_html_()}
            <dl>
                <dt>Score</dt><dd>{self.snake.heuristic}</dd>
                <dt>Breakdown</dt><dd>{self.snake.heuristic_breakdown()}</dd>
                <dt>Size</dt><dd>{self.snake.size}</dd>
                <dt>Age</dt><dd>{len(self.snake.trace)}</dd>
            </dl>
            '''