from typing import Union
import numbers
import numpy as np
import scipy.interpolate as si

def anything_distance(start: np.ndarray, end: np.ndarray):
    dist = np.linalg.norm(end - start)

    return dist

def zip_xs(xs: np.ndarray, x_max_len: int) -> np.ndarray:
    xs = np.array(xs)

    if xs.ndim == 1:
        xs = xs.reshape(1, -1)  # reshape to 2D if it's 1D

    assert xs.ndim == 2, f"Input array must be 2D, got {xs.ndim}D"
    assert xs.shape[1] <= x_max_len, \
        f"Input array shape {xs.shape} exceeds the maximum allowed size of {x_max_len}."

    x_real_len = xs.shape[1]

    return np.concatenate([
        xs,
        np.zeros((xs.shape[0], x_max_len - x_real_len,), dtype=xs.dtype)
    ], axis=1) # (T, x_max_len)

def zip_x(x: np.ndarray, x_max_len: int) -> np.ndarray:
    return zip_xs(np.array([x]), x_max_len)[0]

def unzip_x(x: np.ndarray, x_real_len: int) -> np.ndarray:
    assert x.ndim == 1, f"__call__ return value must be 1D, got {x.ndim}D"

    x = x[:x_real_len]
    return x

class AnythingTrajectoryInterpolator:
    x_max_len = 10

    def __init__(self, times: np.ndarray, xs: np.ndarray):

        assert len(times) >= 1
        assert len(xs) == len(times)
        if not isinstance(times, np.ndarray):
            times = np.array(times)
        if not isinstance(xs, np.ndarray):
            xs = np.array(xs)

        self._times = times
        self._xs = xs

        if len(times) == 1:
            # special treatment for single step interpolation
            self.single_step = True
        else:
            self.single_step = False
            assert np.all(times[1:] >= times[:-1])

            self.x_interp = si.interp1d(times, xs, axis=0, assume_sorted=True)

    @property
    def times(self) -> np.ndarray:
        if self.single_step:
            return self._times
        else:
            return self.x_interp.x

    @property
    def xs(self) -> np.ndarray:
        if self.single_step:
            return self._xs
        else:
            n = len(self.times)
            xs = np.zeros((n, self.x_max_len))
            xs = self.x_interp.y
            return xs

    def trim(self,
            start_t: float, end_t: float
            ) -> "AnythingTrajectoryInterpolator":
        assert start_t <= end_t
        times = self.times
        should_keep = (start_t < times) & (times < end_t)
        keep_times = times[should_keep]
        all_times = np.concatenate([[start_t], keep_times, [end_t]])
        # remove duplicates, Slerp requires strictly increasing x
        all_times = np.unique(all_times)
        # interpolate
        all_xs = self(all_times)
        return AnythingTrajectoryInterpolator(times=all_times, xs=all_xs)

    def drive_to_waypoint(self,
            x, time, curr_time,
            max_pos_speed=np.inf,
            max_rot_speed=np.inf
        ) -> "AnythingTrajectoryInterpolator":
        assert(max_pos_speed > 0)
        assert(max_rot_speed > 0)
        time = max(time, curr_time)

        curr_x = self(curr_time)
        pos_dist, rot_dist = anything_distance(curr_x, x)
        pos_min_duration = pos_dist / max_pos_speed
        rot_min_duration = rot_dist / max_rot_speed
        duration = time - curr_time
        duration = max(duration, max(pos_min_duration, rot_min_duration))
        assert duration >= 0
        last_waypoint_time = curr_time + duration

        # insert new x
        trimmed_interp = self.trim(curr_time, curr_time)
        times = np.append(trimmed_interp.times, [last_waypoint_time], axis=0)
        xs = np.append(trimmed_interp.xs, [x], axis=0)

        # create new interpolator
        final_interp = AnythingTrajectoryInterpolator(times, xs)
        return final_interp

    def schedule_waypoint(self,
            x, time,
            max_x_speed=np.inf,
            curr_time=None,
            last_waypoint_time=None
        ) -> "AnythingTrajectoryInterpolator":
        assert(max_x_speed > 0)
        if last_waypoint_time is not None:
            assert curr_time is not None

        # trim current interpolator to between curr_time and last_waypoint_time
        start_time = self.times[0]
        end_time = self.times[-1]
        assert start_time <= end_time

        if curr_time is not None:
            if time <= curr_time:
                # if insert time is earlier than current time
                # no effect should be done to the interpolator
                return self
            # now, curr_time < time
            start_time = max(curr_time, start_time)

            if last_waypoint_time is not None:
                # if last_waypoint_time is earlier than start_time
                # use start_time
                if time <= last_waypoint_time:
                    end_time = curr_time
                else:
                    end_time = max(last_waypoint_time, curr_time)
            else:
                end_time = curr_time

        end_time = min(end_time, time)
        start_time = min(start_time, end_time)
        # end time should be the latest of all times except time
        # after this we can assume order (proven by zhenjia, due to the 2 min operations)

        # Constraints:
        # start_time <= end_time <= time (proven by zhenjia)
        # curr_time <= start_time (proven by zhenjia)
        # curr_time <= time (proven by zhenjia)

        # time can't change
        # last_waypoint_time can't change
        # curr_time can't change
        assert start_time <= end_time
        assert end_time <= time
        if last_waypoint_time is not None:
            if time <= last_waypoint_time:
                assert end_time == curr_time
            else:
                assert end_time == max(last_waypoint_time, curr_time)

        if curr_time is not None:
            assert curr_time <= start_time
            assert curr_time <= time

        trimmed_interp = self.trim(start_time, end_time)
        # after this, all waypoints in trimmed_interp is within start_time and end_time
        # and is earlier than time

        # determine speed
        duration = time - end_time
        end_x = trimmed_interp(end_time)
        x_dist = anything_distance(x, end_x)
        x_min_duration = x_dist / max_x_speed
        duration = max(duration, x_min_duration)
        assert duration >= 0
        last_waypoint_time = end_time + duration

        # insert new x
        times = np.append(trimmed_interp.times, [last_waypoint_time], axis=0)
        xs = np.append(trimmed_interp.xs, [x], axis=0)

        # create new interpolator
        final_interp = AnythingTrajectoryInterpolator(times, xs)
        return final_interp


    def __call__(self, t: Union[numbers.Number, np.ndarray]) -> np.ndarray:
        is_single = False
        if isinstance(t, numbers.Number):
            is_single = True
            t = np.array([t])

        x = np.zeros((len(t), self.x_max_len))
        if self.single_step:
            x[:] = self._xs[0]
        else:
            start_time = self.times[0]
            end_time = self.times[-1]
            t = np.clip(t, start_time, end_time)

            x = np.zeros((len(t), self.x_max_len))
            x[:] = self.x_interp(t)

        if is_single:
            x = x[0]
        return x


if __name__ == "__main__":
    # Example usage
    times = np.array([0, 1, 2, 3])
    xs = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])

    interpolator = AnythingTrajectoryInterpolator(times, xs)
    print("Times:", interpolator.times)
    print("Xs:", interpolator.xs)
