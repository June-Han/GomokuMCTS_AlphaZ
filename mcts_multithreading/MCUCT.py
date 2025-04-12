import time

import numpy as np

from TreeSearch import TreeSearch

import ipyparallel as ipp
from ipyparallel import Cluster
import time
from gomoku_board import GomokuBoard


class MCUCT(object):
    #Upper confidence bound applied to tree (UCT), a improved Monte Carlo Tree Search (MCTS)
    _ai_uid = 0

    def __init__(self, board_constructor, C=0.3, run_type='ipyparallel', min_num_sim=6e4):
        #The upper bound of the confidence is given by win_rate + C*sqrt(ln(n)/n_i)
        self.C = C
        self.min_num_sim = min_num_sim
        self.game_board = board_constructor()
        self.uid = MCUCT._ai_uid
        self.run_type = run_type
        if run_type != 'ipyparallel':
            TreeSearch.init_tree(self.uid, board_constructor, self.C)
        else:
            self._init_parallel_context(board_constructor)

    def update_state(self, move):
        if self.run_type == 'ipyparallel':
            self._update_state_parallel(move)
        else:
            self._update_state_single(move)

    def best_move(self):
        start_time = time.time()
        if self.run_type == 'ipyparallel':
            result = self._best_move_parallel()
        else:
            result = self._best_move_single()
        print('time spent', time.time() - start_time)
        return result
    

    def _init_parallel_context(self, board_constructor):
        try:
            print("[Init] Trying to connect to existing IPython cluster...")
            self.workers = ipp.Client(timeout=5)
            self.workers.wait()
            print(f"[Init] Connected to workers: {self.workers.ids}")
        except Exception as e:
            print(f"[Init] No cluster found. Launching one... ({e})")

            # Launch a local cluster
            self.cluster = Cluster(n=5)
            self.cluster.start_and_connect_sync()
            self.workers = self.cluster.connect_client_sync()
            print(f"[Init] New cluster started with workers: {self.workers.ids}")
            
        # Purge and initialize the workers
        try:
            self.workers.purge_everything()
            for wid in self.workers.ids:
                self.workers[wid].apply_async(
                    TreeSearch.init_tree, self.uid, board_constructor, self.C)
            self.workers.wait()
        except Exception as e:
            print(f"[Init] Failed to initialize workers: {e}")
            self.workers = None
            self.run_type = 'single'    

    def _update_state_parallel(self, move):
        self.game_board.update_state(move)
        for worker_id in self.workers.ids:
            self.workers[worker_id].apply_async(TreeSearch.update_state, self.uid, move)
        self.workers.wait()

    def _update_state_single(self, move):
        TreeSearch.update_state(self.uid, move)
        self.game_board.update_state(move)

    def _best_move_parallel(self):
        available_moves = self.game_board.available_moves()
        while True:
            stats_all_workers = []
            for worker_id in self.workers.ids:
                stats_all_workers.append(
                    self.workers[worker_id].apply_async(TreeSearch.next_move_stats, self.uid))
            self.workers.wait()
            stats = np.zeros(len(available_moves))
            for r in stats_all_workers:
                stats += r.get()
            if stats.sum() > self.min_num_sim:
                print ('total sim', stats.sum())
                break
        best_move_idx = stats.argmax()
        self.game_stats = stats
        return available_moves[best_move_idx]

    def _best_move_single(self):
        while True:
            stats = TreeSearch.next_move_stats(self.uid)
            if stats.sum() > self.min_num_sim:
                break
        best_move_idx = stats.argmax()
        self.game_stats = stats
        return self.game_board.available_moves()[best_move_idx]

    def __del__(self):
        if self.run_type == 'ipyparallel' and hasattr(self, 'workers'):
            for worker_id in self.workers.ids:
                self.workers[worker_id].apply_sync(TreeSearch.destroy_tree, self.uid)
