"""
生成一个符合要求的agent以及他周围物体在过去指定帧的绝对位置构成的矩阵（2*帧数+3*帧数*neighbor_agents个数）
生成语义图image
"""
import bisect
import warnings
import numpy as np
from typing import Optional
from l5kit.data import ChunkedDataset, get_frames_slice_from_scenes
from l5kit.kinematic import Perturbation
from l5kit.dataset import AgentDataset
from functools import partial
from My_agent_sampling import My_generate_agent_sample
from l5kit.rasterization import Rasterizer, RenderContext

# 根据agents_mask筛选符合要求的my_agent（agents_mask里两列分别表示可用的过去帧和未来帧个数）
MIN_FRAME_HISTORY = 99  # 要求该agent过去帧不少于MIN_FRAME_HISTORY
MIN_FRAME_FUTURE = 50  # 要求该agent未来帧不少于MIN_FRAME_FUTURE


class MyDataset(AgentDataset):
    def __init__(
            self,
            cfg: dict,
            zarr_dataset: ChunkedDataset,
            rasterizer: Rasterizer,
            perturbation: Optional[Perturbation] = None,
            agents_mask: Optional[np.ndarray] = None,
            min_frame_history: int = MIN_FRAME_HISTORY,
            min_frame_future: int = MIN_FRAME_FUTURE,
    ):
        # 记录画图所用的信息
        render_context = RenderContext(
            raster_size_px=np.array(cfg["raster_params"]["raster_size"]),
            pixel_size_m=np.array(cfg["raster_params"]["pixel_size"]),
            center_in_raster_ratio=np.array(cfg["raster_params"]["ego_center"]),
        )
        # partial表示固定函数My_generate_agent_sample的部分参数来生成一个新函数
        # My_generate_agent_sample用来返回history_neighbor_agents的所有信息（"centroid","extent","yaw","velocity","track_id","label_probabilities"）
        # 和语义图信息
        self.My_sample_function = partial(
            My_generate_agent_sample,
            render_context=render_context,
            history_num_frames=cfg["model_params"]["history_num_frames"],
            history_step_size=cfg["model_params"]["history_step_size"],
            future_num_frames=cfg["model_params"]["future_num_frames"],
            future_step_size=cfg["model_params"]["future_step_size"],
            filter_agents_threshold=cfg["raster_params"]["filter_agents_threshold"],
            rasterizer=rasterizer,
            perturbation=perturbation,
        )

        super(MyDataset, self).__init__(cfg, zarr_dataset, rasterizer, perturbation, agents_mask, min_frame_history,
                                        min_frame_future)

    def get_frame_sub(self, scene_index: int, state_index: int, frame_index: int,
                      track_id: Optional[int] = None) -> dict:
        """
        Args:
            scene_index (int): 在zarr中的第几个场景
            state_index (int): 在这一场景中的第几帧
            frame_index (int): 在所有frame数据集中的当前帧数
            track_id (Optional[int]): agent在这一场景中的track_id（ego为none）
        Returns:
            agent以及他周围物体在过去指定帧的绝对位置构成的矩阵（2*帧数+3*帧数*neighbor_agents个数）
        """
        # 截取这一场景中的所有帧
        frames = self.dataset.frames[get_frames_slice_from_scenes(self.dataset.scenes[scene_index])]

        tl_faces = self.dataset.tl_faces
        try:
            if self.cfg["raster_params"]["disable_traffic_light_faces"]:
                tl_faces = np.empty(0, dtype=self.dataset.tl_faces.dtype)  # completely disable traffic light faces
        except KeyError:
            warnings.warn(
                "disable_traffic_light_faces not found in config, this will raise an error in the future",
                RuntimeWarning,
                stacklevel=2,
            )
        data = self.My_sample_function(state_index, frames, self.dataset.agents, tl_faces,
                                       track_id)  # 这里的frames已经是一个scene中的所有frames
        image = data["image"].transpose(2, 0, 1)

        target_positions = np.array(data["target_positions"], dtype=np.float32)
        target_yaws = np.array(data["target_yaws"], dtype=np.float32)

        history_positions = np.array(data["history_positions"], dtype=np.float32)
        history_yaws = np.array(data["history_yaws"], dtype=np.float32)

        timestamp = frames[state_index]["timestamp"]
        track_id = np.int64(-1 if track_id is None else track_id)  # always a number to avoid crashing torch

        all_agents_track_id = (data["history_neighbor_agents"][len(data["history_neighbor_agents"]) - 1]["track_id"])
        neighbor_agents_history_position = np.zeros((len(all_agents_track_id), len(data["history_neighbor_agents"]), 3))
        My_agent_history_position = np.zeros((len(data["history_neighbor_agents"]), 2))
        for k in range(len(all_agents_track_id)):
            for i in range(len(data["history_neighbor_agents"])):
                for j in range(len(data["history_neighbor_agents"][i])):
                    if data["history_neighbor_agents"][i][j]["track_id"] == all_agents_track_id[k]:
                        if data["history_neighbor_agents"][i][j]["track_id"] < track_id:
                            neighbor_agents_history_position[k, i, 2] = 1
                            neighbor_agents_history_position[k, i, 0] = \
                                data["history_neighbor_agents"][i][j]["centroid"][0]
                            neighbor_agents_history_position[k, i, 1] = \
                                data["history_neighbor_agents"][i][j]["centroid"][1]
                            break
                        elif data["history_neighbor_agents"][i][j]["track_id"] > track_id:
                            neighbor_agents_history_position[k - 1, i, 2] = 1
                            neighbor_agents_history_position[k - 1, i, 0] = \
                                data["history_neighbor_agents"][i][j]["centroid"][0]
                            neighbor_agents_history_position[k - 1, i, 1] = \
                                data["history_neighbor_agents"][i][j]["centroid"][1]
                            break
                        else:
                            My_agent_history_position[i, 0] = data["history_neighbor_agents"][i][j]["centroid"][0]
                            My_agent_history_position[i, 1] = data["history_neighbor_agents"][i][j]["centroid"][1]
                            break

        for i in range(len(data["history_neighbor_agents"])):
            neighbor_agents_history_position[
                len(all_agents_track_id) - 1, len(data["history_neighbor_agents"]) - 1 - i, 2] = 1
            neighbor_agents_history_position[
                len(all_agents_track_id) - 1, len(data["history_neighbor_agents"]) - 1 - i, 0] = \
                self.dataset.frames[frame_index - i]["ego_translation"][0]
            neighbor_agents_history_position[
                len(all_agents_track_id) - 1, len(data["history_neighbor_agents"]) - 1 - i, 1] = \
                self.dataset.frames[frame_index - i]["ego_translation"][1]

        neighbor_agents_track_id = np.append(np.delete(all_agents_track_id, np.where(all_agents_track_id == track_id)),
                                             -1)

        result = {
            "image": image[200:203],
            "My_agent_track_id": track_id,
            "neighbor_agents_track_id": neighbor_agents_track_id,
            "neighbor_agents_history_position": neighbor_agents_history_position[:, :, 0:2],
            "My_agent_history_position": My_agent_history_position,
            "target_positions": target_positions,
            # "target_yaws": target_yaws,
            # "target_availabilities": data["target_availabilities"],
            # "history_positions": history_positions,
            # "history_yaws": history_yaws,
            # "history_availabilities": data["history_availabilities"],
            # "world_to_image": data["raster_from_world"],
            # "raster_from_world": data["raster_from_world"],
            # "raster_from_agent": data["raster_from_agent"],
            # "agent_from_world": data["agent_from_world"],
            # "world_from_agent": data["world_from_agent"],
            "timestamp": timestamp,
            # "centroid": data["centroid"],
            # "yaw": data["yaw"],
            # "extent": data["extent"],
        }
        return result

    def __getitem__(self, index: int) -> dict:
        """
        Differs from parent by iterating on agents and not AV.
        """
        if index < 0:
            if -index > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            index = len(self) + index

        index = self.agents_indices[index]
        track_id = self.dataset.agents[index]["track_id"]
        frame_index = bisect.bisect_right(self.cumulative_sizes_agents, index)
        scene_index = bisect.bisect_right(self.cumulative_sizes, frame_index)

        if scene_index == 0:
            state_index = frame_index
        else:
            # 在这一场景中是第几帧
            state_index = frame_index - self.cumulative_sizes[scene_index - 1]
        return self.get_frame_sub(scene_index, state_index, frame_index, track_id=track_id)
