class BaseRandom():
    mod = 5
    def __init__(self,rand_seed = 5):
        self.rand_seed = rand_seed
        self.mod = mod
        self.next_val = rand_seed * foo % mod
    def next(self):
        self.next_val = self.next_val * foo % mod
        return self.next_val

class Uniform():
    pass
class RandFloat():
    pass
class RandInt():
    pass
class Gussaion():
    pass

