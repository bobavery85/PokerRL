class OfflineArgs:

    def __init__(self, n_hands, hero_file_path, villain_file_path,
                 hero_use_canonical, villain_use_canonical):
        self.n_hands = n_hands
        self.hero_file_path = hero_file_path
        self.villain_file_path = villain_file_path
        self.hero_use_canonical = hero_use_canonical
        self.villain_use_canonical = villain_use_canonical