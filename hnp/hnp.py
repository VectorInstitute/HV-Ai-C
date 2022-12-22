import numpy as np

class HNP():
    def __init__(self, slow_continuous_idx) -> None:
        self.slow_continuous_idx = slow_continuous_idx

        n_slow_cont = len(self.slow_continuous_idx)
        if n_slow_cont > 0:
            portion_index_matrix = np.vstack(
                (np.zeros(n_slow_cont), np.ones(n_slow_cont))
            ).T
            self.all_portion_index_combos = np.array(
                np.meshgrid(*portion_index_matrix), dtype=int
            ).T.reshape(-1, n_slow_cont)

    def get_next_value(self, vtb, full_obs_index, cont_obs_index_floats):
        # If change first 5 lines of this function also
        if len(self.slow_continuous_idx) == 0:  # No HNP calculation needed
            return vtb[tuple(full_obs_index)]
        slow_cont_obs_index_floats = cont_obs_index_floats[
            : len(self.slow_continuous_idx)
        ]
        slow_cont_obs_index_int_below = np.floor(slow_cont_obs_index_floats).astype(
            np.int32
        )
        slow_cont_obs_index_int_above = np.ceil(slow_cont_obs_index_floats).astype(
            np.int32
        )

        vtb_index_matrix = np.vstack(
            (slow_cont_obs_index_int_below, slow_cont_obs_index_int_above)
        ).T
        all_vtb_index_combos = np.array(np.meshgrid(*vtb_index_matrix)).T.reshape(
            -1, len(slow_cont_obs_index_int_above)
        )

        portion_below = slow_cont_obs_index_int_above - slow_cont_obs_index_floats
        portion_above = 1 - portion_below
        portion_matrix = np.vstack((portion_below, portion_above)).T

        non_hnp_index = full_obs_index[len(self.slow_continuous_idx) :]
        next_value = 0
        for i, combo in enumerate(self.all_portion_index_combos):
            portions = portion_matrix[np.arange(len(slow_cont_obs_index_floats)), combo]
            value_from_vtb = vtb[
                tuple(np.hstack((all_vtb_index_combos[i], non_hnp_index)).astype(int))
            ]
            next_value += np.prod(portions) * value_from_vtb

        return next_value