import multiprocessing
import numpy as np
from utils.spacemouse import pyspacemouse
# import pyspacemouse
from typing import Tuple


class SpaceMouseExpert:
    """
    This class provides an interface to the SpaceMouse.
    It continuously reads the SpaceMouse state and provides
    a "get_action" method to get the latest action and button state.
    """

    def __init__(self):
        pyspacemouse.open()

        # Manager to handle shared state between processes
        self.manager = multiprocessing.Manager()
        self.latest_data = self.manager.dict()
        self.latest_data["action"] = [0.0] * 6  # Using lists for compatibility
        self.latest_data["buttons"] = [0, 0, 0, 0]

        # Start a process to continuously read the SpaceMouse state
        self.process = multiprocessing.Process(target=self._read_spacemouse)
        self.process.daemon = True
        self.process.start()

    def _deal_spacemouse(self, action):
        # spacemouse_threshold = 0.4
        # spacemouse_pos_coef = 0.08
        # spacemouse_rot_coef = 0.25

        spacemouse_threshold = 0
        spacemouse_pos_coef = 1
        spacemouse_rot_coef = 1
        space_state = np.array(action)

        cond1 = space_state > spacemouse_threshold
        cond2 = space_state < -spacemouse_threshold
        cond3 = np.logical_and(space_state > -spacemouse_threshold, space_state < spacemouse_threshold)
        space_state = np.where(cond1, space_state - spacemouse_threshold, space_state)
        space_state = np.where(cond2, space_state + spacemouse_threshold, space_state)
        space_state = np.where(cond3, 0, space_state)
        space_state[:3] *= spacemouse_pos_coef
        space_state[3:] *= spacemouse_rot_coef

        action = space_state.tolist()

        return action

    def _read_spacemouse(self):
        def process_button_states(button_states):
            return [button_states[0], button_states[-1]]

        while True:
            state = pyspacemouse.read_all()
            action = [0.0] * 6
            buttons = [0, 0, 0, 0]

            if len(state) == 2:
                raise NotImplementedError("Dual SpaceMouse control is not supported yet.")
                action = [
                    -state[0].y, state[0].x, state[0].z,
                    -state[0].roll, -state[0].pitch, -state[0].yaw,
                    -state[1].y, state[1].x, state[1].z,
                    -state[1].roll, -state[1].pitch, -state[1].yaw
                ]
                buttons = process_button_states(state[0].buttons) + process_button_states(state[1].buttons)
            elif len(state) == 1:
                # # left
                action = [
                    state[0].y, -state[0].x, state[0].z,
                    state[0].roll, state[0].pitch, -state[0].yaw
                ]
                # right
                # action = [
                #     -state[0].y, state[0].x, state[0].z,
                #     -state[0].roll, -state[0].pitch, -state[0].yaw
                # ]
                buttons = process_button_states(state[0].buttons)
            else:
                raise ValueError(f"Invalid state length, expected 1 or 2, got {len(state)}")

            # action = self._deal_spacemouse(action)

            # Update the shared state
            self.latest_data["action"] = action
            self.latest_data["buttons"] = buttons

    def get_action(self) -> Tuple[np.ndarray, list]:
        """Returns the latest action and button state of the SpaceMouse."""
        action = self.latest_data["action"]
        buttons = self.latest_data["buttons"]
        return np.array(action), buttons

    def close(self):
        # pyspacemouse.close()
        self.process.terminate()
