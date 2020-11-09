import bisect
import numpy as np
from l5kit.data import ChunkedDataset, get_frames_slice_from_scenes
from functools import partial
from l5kit.rasterization import Rasterizer
from pathlib import Path
from zarr import convenience
from l5kit.sampling.slicing import get_future_slice, get_history_slice
from l5kit.data import get_agents_slice_from_frames
from l5kit.data.filter import filter_agents_by_frames

MIN_FRAME_HISTORY = 99
MIN_FRAME_FUTURE = 50


class HandwriteDataset(object):
    def __init__(
            self,
            cfg: dict,
            zarr_dataset: ChunkedDataset,
            rasterizer: Rasterizer,
            agents_mask_path: str,
    ):
        self.cfg = cfg
        self.dataset = zarr_dataset
        self.rasterizer = rasterizer
        # import agents mask and agents track id in frames
        file = np.load(agents_mask_path)
        agents_mask = file['agents_mask']
        agents_track_id_in_frames = file['agents_track_id']
        # import num_index
        if 'num_index' in file:
            self.num_index = file['num_index']
        self.agents_track_id_in_frames = agents_track_id_in_frames
        self.cumulative_sizes_agents = self.dataset.frames["agent_index_interval"][:, 1]
        self.cumulative_sizes = self.dataset.scenes["frame_index_interval"][:, 1]
        self.agents_indices = np.nonzero(agents_mask)[0]
        self.sample_function = partial(
            self.generate_agent_sample,
            history_num_frames=cfg["model_params"]["history_num_frames"],
            history_step_size=cfg["model_params"]["history_step_size"],
            future_num_frames=cfg["model_params"]["future_num_frames"],
            future_step_size=cfg["model_params"]["future_step_size"],
        )

    def get_frame(self, scene_index: int, state_index: int, frame_index: int) -> dict:

        frames = self.dataset.frames[get_frames_slice_from_scenes(self.dataset.scenes[scene_index])]
        data = self.sample_function(state_index, frames, self.dataset.agents)
        timestamp = frames[state_index]["timestamp"]

        history_num_frames = len(data["history_agents"])
        num_agents = int(data["history_agents"][history_num_frames - 1][-1]["track_id"] + 1)
        history_agents_position = np.zeros((num_agents, history_num_frames, 2))
        agents_available_frames = np.zeros((num_agents, 7))
        for i in range(len(data["history_agents"][history_num_frames - 1])):
            select_agent = data["history_agents"][history_num_frames - 1][i]
            agents_available_frames[select_agent["track_id"], 2] = 1
            agents_available_frames[select_agent["track_id"], 3] = int(
                np.argmax(select_agent["label_probabilities"]) != 1)
        agents_available_frames[0, 2], agents_available_frames[0, 3], agents_available_frames[0, 4] = 1, 1, 1

        for i in range(history_num_frames):
            history_agents_position[0, history_num_frames - 1 - i, 0] = \
                self.dataset.frames[frame_index - i]["ego_translation"][0]
            history_agents_position[0, history_num_frames - 1 - i, 1] = \
                self.dataset.frames[frame_index - i]["ego_translation"][1]
            agents_available_frames[0, 0] += 1
            for j in range(len(data["history_agents"][i])):
                track_id = data["history_agents"][i][j]["track_id"]
                position = data["history_agents"][i][j]["centroid"]
                if track_id < num_agents:
                    history_agents_position[track_id, i, 0] = position[0]
                    history_agents_position[track_id, i, 1] = position[1]
                    agents_available_frames[track_id, 0] += 1

        future_num_frames = len(data["future_agents"])
        future_agents_position = np.zeros((num_agents, future_num_frames, 2))
        for i in range(future_num_frames):
            future_agents_position[0, i, 0] = self.dataset.frames[frame_index + i + 1]["ego_translation"][0]
            future_agents_position[0, i, 1] = self.dataset.frames[frame_index + i + 1]["ego_translation"][1]
            agents_available_frames[0, 1] += 1
            for j in range(len(data["future_agents"][i])):
                track_id = data["future_agents"][i][j]["track_id"]
                position = data["future_agents"][i][j]["centroid"]
                if track_id < num_agents:
                    future_agents_position[track_id, i, 0] = position[0]
                    future_agents_position[track_id, i, 1] = position[1]
                    agents_available_frames[track_id, 1] += 1

        for i in range(len(data["history_agents"][history_num_frames - 1])):
            select_agent = data["history_agents"][history_num_frames - 1][i]
            history_position = history_agents_position[select_agent["track_id"], :, :]
            # calculate change of coordinates
            hist_num = int(MIN_FRAME_HISTORY - agents_available_frames[select_agent["track_id"], 0] + 1)
            # obtain maximum and minimum coordinate of future and history
            if hist_num != MIN_FRAME_HISTORY+1:
                max_x = max(history_position[hist_num:, 0])
                min_x = min(history_position[hist_num:, 0])
                max_y = max(history_position[hist_num:, 1])
                min_y = min(history_position[hist_num:, 1])
            else:
                max_x, min_x, max_y, min_y = 0, 0, 0, 0
            agents_available_frames[select_agent["track_id"], 5] = max_x - min_x
            agents_available_frames[select_agent["track_id"], 6] = max_y - min_y
            if max_x - min_x > 2 or max_y - min_y > 2:
                agents_available_frames[select_agent["track_id"], 4] = 1

        select_track_id = self.agents_track_id_in_frames[frame_index, :]
        select_track_id = np.delete(select_track_id, np.argwhere(select_track_id == 0))


        result = {
            "history_agents_position": history_agents_position,
            "future_agents_position": future_agents_position,
            "agents_available_frames": agents_available_frames,
            "timestamp": timestamp,
            "frame_index": frame_index,
            "select_track_id": select_track_id,
        }
        return result

    def __len__(self) -> int:
        return len(self.agents_indices)

    def __getitem__(self, index: int) -> dict:
        if index < 0:
            index = len(self) + index
        index = self.agents_indices[index]
        frame_index = bisect.bisect_right(self.cumulative_sizes_agents, index)
        scene_index = bisect.bisect_right(self.cumulative_sizes, frame_index)

        if scene_index == 0:
            state_index = frame_index
        else:
            state_index = frame_index - self.cumulative_sizes[scene_index - 1]
        return self.get_frame(scene_index, state_index, frame_index)

    def load_agents_mask(self) -> np.ndarray:
        agent_prob = self.cfg["raster_params"]["filter_agents_threshold"]
        agents_mask_path = Path(self.dataset.path) / f"agents_mask/{agent_prob}"
        agents_mask = convenience.load(str(agents_mask_path))
        return agents_mask

    def get_index(self, agents_mask: np.ndarray):
        record = np.zeros(len(self.dataset.frames))
        for i in range(len(agents_mask)):
            if agents_mask[i] != 0:
                if record[bisect.bisect_right(self.cumulative_sizes_agents, i)] == 0:
                    record[bisect.bisect_right(self.cumulative_sizes_agents, i)] += 1
                else:
                    agents_mask[i] = 0
        return np.nonzero(agents_mask)[0]

    @staticmethod
    def generate_agent_sample(state_index: int,
                              frames: np.ndarray,
                              agents: np.ndarray,
                              history_num_frames: int,
                              history_step_size: int,
                              future_num_frames: int,
                              future_step_size: int,
                              ) -> dict:
        history_slice = get_history_slice(state_index, history_num_frames, history_step_size,
                                          include_current_state=True)
        future_slice = get_future_slice(state_index, future_num_frames, future_step_size)
        history_frames = frames[history_slice].copy()
        future_frames = frames[future_slice].copy()
        sorted_frames = np.concatenate((history_frames[::-1], future_frames))
        agent_slice = get_agents_slice_from_frames(sorted_frames[0], sorted_frames[-1])
        agents = agents[agent_slice].copy()
        history_frames["agent_index_interval"] -= agent_slice.start
        future_frames["agent_index_interval"] -= agent_slice.start
        history_agents = filter_agents_by_frames(history_frames, agents)
        future_agents = filter_agents_by_frames(future_frames, agents)
        return {
            "history_agents": history_agents[::-1],
            "future_agents": future_agents,
        }
