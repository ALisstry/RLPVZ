import unittest

import numpy as np

from models.ddqn import checkpoint as ck
from models.ddqn.ddqn import PrioritizedReplayBuffer


class DDQNCheckpointRoundTripTest(unittest.TestCase):
    def test_replay_buffer_round_trip_preserves_entries(self):
        buf = PrioritizedReplayBuffer(memory_size=4, burn_in=1)
        for i in range(2):
            buf.append(
                np.ones(3, dtype=np.float32),
                1,
                0.5,
                False,
                np.ones(3, dtype=np.float32),
                np.ones(3, dtype=bool),
                np.ones(3, dtype=bool),
            )

        data = ck._serialize_buffer(buf)
        restored = ck._deserialize_buffer(data)

        self.assertEqual(len(restored), len(buf))
        self.assertEqual(restored.memory_size, buf.memory_size)
        self.assertEqual(restored._write_ptr, buf._write_ptr)
        self.assertEqual(restored.sum_tree.n_entries, buf.sum_tree.n_entries)

    def test_replay_buffer_round_trip_preserves_wraparound_order(self):
        buf = PrioritizedReplayBuffer(memory_size=4, burn_in=1)
        for i in range(5):
            state = np.array([float(i + 1)], dtype=np.float32)
            buf.append(
                state,
                i,
                float(i),
                False,
                state + 1,
                np.array([True], dtype=bool),
                np.array([True], dtype=bool),
            )

        data = ck._serialize_buffer(buf)
        restored = ck._deserialize_buffer(data)

        self.assertTrue(
            np.array_equal(data["states"][0], np.array([2.0], dtype=np.float32)),
            msg="serialized buffer should preserve chronological order",
        )
        self.assertEqual(restored._write_ptr, buf._write_ptr)
        self.assertTrue(
            np.array_equal(restored.replay_memory[1].state, np.array([2.0], dtype=np.float32)),
            msg="restored ring buffer should preserve the wrapped slot layout",
        )


if __name__ == "__main__":
    unittest.main()
